

//=============================||  PARAMETERS  ||=============================//


#version 330 core

in vec3 FragPos;
out vec4 FragColor;

uniform float aspect_ratio;
uniform vec3 camera_pos;
// The current time
uniform float current_time;
// Number of primitives sent from outside
uniform int primitive_count;
// The current frame
uniform int frame;

// We can output each frame onto a texture. Then in the first pass = 0
// we can accumulate the output of this frame with the previous frames'
// outputs, and then in the second pass = 1 we can display the
// result in real time.
uniform sampler2D last_frame;
uniform int pass;

#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define LIMIT 999999
#define SAMPLE_COUNT 64
#define BOUNCE_COUNT 4
// Amount of base light added to all objects to avoid black objects
#define AMBIENT_LIGHT 0.025
// Maximum possible BVH tree height
#define MAX_BVH_HEIGHT 64
// Maximum possible number of primitives
#define MAX_SCENE_SIZE 256

#define ENABLE_ANTI_ALIASING true
#define USE_AVERAGE true


//===============================||  OBJECTS  ||==============================//


// A triangle primitive
struct Triangle {
    // We assume that the vertices are given in the counter clockwise
    // order.
    vec3 v0;
    vec3 v1;
    vec3 v2;
    // Area, in case we want it saved to avoid recalculations.
    float area;
    // Optional parameters, in case the normals used at each
    // vertex are different from the face normal.
    // Assumed to be normalized.
    vec3 n0;
    vec3 n1;
    vec3 n2;

    // Graphics data fields
    vec3 color;
    bool is_light_source;
    float roughness;
    vec3 reflectance;
};


// A plane, which can be written parametrically as
// alpha * dir1 + beta * dir2 + point on plane, where the
// directions are linearly independent.
struct Plane_parametric {
    vec3 dir1;
    vec3 dir2;
    vec3 point;
};


// A plane can also implicitey be defined as n . (p - q) = 0 where 
// q is a point on the plane, and n is 
// the normal = normalize(dir1 cross dir2).
struct Plane_implicit{
    vec3 normal;
    vec3 point;
};


// A ray, given parametrically as r(t) = o + td
struct Ray {
    // The direction of the ray must always be normalized.
    vec3 dir;
    vec3 origin;
};


// Records a ray hit
struct Ray_hit {
    bool hit;
    // Coordinate of the hit
    vec3 coordinate;
    // The place along the ray where the hit took place
    float t;
    // Intersected bject index
    int index;
};


// Records the result of a bounce: the ouput T^k L_e,
// and the accumulated product used in the next bounce.
struct Bounce_data{
    vec3 radiance;
    vec3 throughput;
};


// This is used for the BVH. The BVH is used to accelerate the
// intersection of the ray with the list of n primitives, from O(n)
// to O(log(n)).
// The tree is stored as an array, with the children given by indexes.
// The leafs of the BVH also contain objects (primitives), which
// are reference by an index (used in the Scene[] array). 
struct BVH_node{
    // BVH volume
    vec3 center;
    vec3 radius;

    // Children
    int left;
    int right;

    // Referenced object (-1 if not a leaf or a leaf with no object)
    int object;
};


//==========================|| CHANGE OF MEASURE ||===========================//


// Returns the triangle normal.
// This assumes that the triangle's vertices are given in the counter
// clockwise order.
vec3 triangle_normal(Triangle triangle, vec3 point) {
    vec3 a = triangle.v1 - triangle.v0;
    vec3 b = triangle.v2 - triangle.v0;
    return normalize(cross(a, b));
}


// Returns a point as a linear combination of the three vertices
// of a triangle.
vec3 barycentric_coordinates(Triangle triangle, vec3 point) {
    vec3 v0 = triangle.v1 - triangle.v0;
    vec3 v1 = triangle.v2 - triangle.v0;
    vec3 v2 = point - triangle.v0;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    float denom = d00 * d11 - d01 * d01;
    if (denom == 0.0) {
        // Degenerate triangle
        return vec3(0.0);
    }

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0 - v - w;

    return vec3(u, v, w);
}


// Returns the triangle normal.
// Uses the normals of each vertex (in case they are different),
// coupled with the barycentric weights of the point;
vec3 triangle_barycentric_normal(Triangle triangle, vec3 point) {
    vec3 weights = barycentric_coordinates(triangle, point);
    return normalize(triangle.n0 * weights.x 
        + triangle.n1 * weights.y + triangle.n2 * weights.z);
}


// Returns the triangle surface area
float triangle_area(Triangle triangle) {
    vec3 a = triangle.v1 - triangle.v0;
    vec3 b = triangle.v2 - triangle.v0;
    return 0.5 * length(cross(a, b));
}


// Generates a point on a triangle using parameters between 0 and 1.
// We can do that by generating a random point inside the parallelogram
// defined by (v1 - v0) and (v2 - v0), and the folding the other triangle
// in to avoid points not in the triangle (for u + v <= 1).
vec3 triangle_transform(Triangle triangle, float u, float v){
    vec3 a = triangle.v1 - triangle.v0;
    vec3 b = triangle.v2 - triangle.v0;
    if(u+v > 1){
        return (1 - u) * a + (1 - v) * b + triangle.v0;
    }
    else{
        return u * a + v * b + triangle.v0;
    }
}


// The change of measure term, or sqrt(det(J^TJ)) where J is the
// Jacobian matrix of the transformation, is twice the surface
// area of the triangle.
// Note that it actually does not depend on u and v.
float change_of_measure(Triangle triangle, float u, float v){
    // If the area is set and defined already, we don't
    // recompute it.
    if(triangle.area > 0.0){
        return 2.0 * triangle.area;
    }
    else{
        return 2.0 * triangle_area(triangle);
    }
}


//==============================||  COLLISIONS  ||============================//


// Checks for an intersection between a ray and a plane.
// The plane is parametric: alpha * t1 + beta * t2 + point on plane.
// So to find the intersection we set:
// alpha * t1 + beta * t2 + point on plane = dt + o
// alpha * t1 + beta * t2 -dt = o - point on plane
// [t1 t2 -d][alpha, beta, t] = o - point on plane
// which is a linear system we can solve.
// Also returns the parameters alpha and beta of the intersection.
Ray_hit ray_plane_intersection(
    Plane_parametric plane, Ray ray, 
    inout vec2 parameters
){

    Ray_hit result;
    mat3 matrix = mat3(plane.dir1, plane.dir2, -ray.dir);
    vec3 b = ray.origin - plane.point;

    // First we check if there even is an intersection
    if (abs(determinant(matrix)) > EPSILON){
        vec3 v = inverse(matrix) * b;
        float t = v.z;
        if(t > 0){
            result.hit = true;
            result.coordinate = ray.dir * t + ray.origin;
            result.t = t;
            parameters.x = v.x;
            parameters.y = v.y;
            return result;
        }
    }

    result.hit = false;
    return result;
}


// Checks for an intersection between a ray and a plane,
// this time given as an implicit equation.
// This is easier and in fact much faster, as we can simply
// solve for t in n . (r(t) - q) = 0.
// Where we will find that t = n . (q - o) / n . d.
// However, we don't automatically get the coordinates
// of the intersection in the plane basis, that is, the
// parameters alpha and beta that multiply the two directions dir1
// and dir2, which we will be needing.
Ray_hit ray_plane_intersection(Plane_implicit plane, Ray ray){
    
    Ray_hit result;
    float denominator = dot(plane.normal, ray.dir);
    float numerator = dot(plane.normal, plane.point - ray.origin);

    // This means there is no intersection or the whole plane
    // is an intersection, since the normal of the plane and
    // direction are orthogonal (the plane and ray are parallel).
    if(abs(denominator) > EPSILON){
        float t = numerator / denominator;
        if(t > 0){
            result.hit = true;
            result.coordinate = ray.dir * t + ray.origin;
            result.t = t;
            return result;
        }
    }

    result.hit = false;
    return result;
}


// Intersects a triangle with a ray
Ray_hit ray_triangle_intersection(Triangle triangle, Ray ray){
    Plane_parametric plane;
    plane.dir1 = (triangle.v1 - triangle.v0);
    plane.dir2 = (triangle.v2 - triangle.v0);
    plane.point = triangle.v0;

    // The point of intersection in the basis dir1 dir2;
    vec2 parameters;
    Ray_hit hit = ray_plane_intersection(plane, ray, parameters);
    if(hit.hit){
        if(parameters.x > 0.0 && parameters.y > 0.0
        && parameters.x + parameters.y < 1.0){
            return hit;
        }
    }
    
    Ray_hit false_hit;
    false_hit.hit = false;
    return false_hit;
}


//==========================|| SCENE INTERSECTION ||==========================//


// A funtion that casts a ray into the scene, and returns the closest
// intersection. It is O(n) for n objects.
Ray_hit ray_tracing_function(
    inout Triangle scene[MAX_SCENE_SIZE],
    Ray ray, 
    int scene_size
) {
    Ray_hit hit;
    hit.hit = false;
    hit.t = LIMIT;
    
    for(int i = 0; i < scene_size; i++){

        Ray_hit hit_temp = ray_triangle_intersection(scene[i], ray);
        hit_temp.index = i;

        // If there is a hit, and the hit occurs before reaching y
        // (with epsilon to avoid self intersection)
        if(hit_temp.hit && hit_temp.t < hit.t){
            hit = hit_temp;
        }
    }

    return hit;
}


// A function that also checks for the closest intersection
// between a ray and the scene, but uses the BVH of the scene to
// accelerate the search. It is O(log(n)) for n objects.
Ray_hit ray_tracing_function(
    inout Triangle scene[MAX_SCENE_SIZE],
    inout BVH_node root,
    Ray ray, 
    int scene_size
) {
    Ray_hit hit;
    hit.hit = false;
    hit.t = LIMIT;
    
    // We will traverse the BVH. If a parent node's sphere does not
    // intersect the ray, we won't bother with its children.
    // If a lead node is determined to be intersecting the ray, then
    // we check if its referenced primitive (triangle) intersects
    // the ray, and then return the closest of these hits.

    // Recursion is best avoided in GLSL, so we will instead be using
    // a stack, with a size equal to that of the maximum height
    // of the BVH tree.

    

    return hit;
}


//=========================|| VISIBILITY FUNCTION ||==========================//


// In order to check if a point x is visible to y, we need to ensure
// there are no collisions when casting a ray from x to y.
// Note that we only check for collisions when t is between 0
// and the distance between x and y, as we only care for collisions
// between x and y.
float visibility_term(
    inout Triangle scene[MAX_SCENE_SIZE],
    vec3 x, vec3 y,
    int scene_size
) {
    Ray ray;
    // We add a small epsilon to avoid self intersection
    ray.dir = normalize(y - x);
    ray.origin = x + ray.dir * EPSILON;

    float dist = length(x - y);

    // We cast a ray from x to y, and check for any collision
    for(int i = 0; i < scene_size; i++){
        Ray_hit hit = ray_triangle_intersection(scene[i], ray);

        // If there is a hit, and the hit occurs before reaching y
        // (with epsilon to avoid self intersection)
        if(hit.hit && hit.t < dist - EPSILON){
            return 0.0;
        }
    }

    return 1.0;
}


//==========================|| GEOMETRY FUNCTION ||===========================//


// This is the geometry term. The arguments nx and ny are the normals
// at x and y, assumed to be normalized.
float geometry_term(vec3 x, vec3 y, vec3 nx, vec3 ny){
    float product = dot((x-y), (x-y));

    // We ensure the output can't be negative.
    return max(dot(y - x, nx), 0.0) * max(dot(x - y, ny), 0.0) 
        / (product * product);
}


//==============================|| PSEUDO-RNG ||==============================//


// We want our PRNG to output to be completely uncorrelated between
// pixels, frames, bounces, and samples, and parameter number, 
// so our seed takes them all into account.
vec4 seed(vec2 pixel, float frame, int bounce, int sample, int parameter){
    return vec4(pixel.x, pixel.y, frame, 
        parameter * 10000 + bounce * 100 + sample);
}


// Returns which bucket between 0 and n-1 a float between 0 and 1
// corresponds to.
int bucket_index(int n, float x) {
    x = clamp(x, 0.0, 1.0 - EPSILON);
    return int(x * float(n));
}


// A pseudo random number generator for a float between 0.0 and 1.0.
float random(vec4 seed) {
    const vec4 k = vec4(12.9898, 78.233, 45.164, 94.673);
    float t = dot(seed, k);
    return fract(sin(t) * 43758.5453);
}


//============================|| BRDF FUNCTION ||=============================//


// Fresnel function: here the reflectance depends on the material
// and is set by the user, and omega_i is the incidant radiance 
// direction, assumed to be normalized.
// Here h is the halfway vector, assumed to be normalized.
vec3 fresnel(vec3 omega_i, vec3 h, vec3 reflectance){
    float product = clamp(dot(omega_i, h), 0.0, 1.0);
    return reflectance + 
        (vec3(1.0) - reflectance) * pow((1.0 - product), 5.0);
}


// GGX normal distribution function: here n is the normal vector
// at the point of contact and h is the halfway vector, both of which
// are assumed to be normalized. 
// Alpha is equal to the roughness squared, which depends on the material
// and is set by the user.
float ggx_normal(vec3 n, vec3 h, float alpha){
    float product = clamp(dot(n, h), 0.0, 1.0);
    float denominator = 
        PI * pow((pow(product, 2) * (alpha - 1) + 1), 2);
    return alpha / denominator;
}


// The geometry attenuation function for one direction,
// or more specifically, the Smith masking-shadowing function.
// Here omega is some normalized direction, n a normal at the point
// of contact, also normalized, and alpha is the roughness squared,
// a property of the material.
float geometry_one_d(vec3 omega, vec3 n, float alpha){
    float product = clamp(dot(n, omega), 0.0, 1.0);
    float denominator = product + 
        sqrt(alpha + (1-alpha) * pow(product, 2));
    return 2 * product / denominator;
}


// The full geometry term for the microfacet BRDF.
// Takes as input the incident and exitant directions, both normalized.
float full_geometry(vec3 omega_i, vec3 omega_o, vec3 n, float alpha){
    return geometry_one_d(omega_i, n, alpha) * 
        geometry_one_d(omega_o, n, alpha);
}


// The full Cook-Torrance microfacet BRDF.
// Takes as input the incident and exitant directions (normalized).
// Also takes in the reflectance, roughness parameters, as well
// as the color of the scene at x_{n} (texture or solid color)
// and the normal vector at x_{n}.
vec3 cook_torrance_BRDF(
    vec3 omega_i, vec3 omega_o, vec3 n,
    vec3 color, vec3 reflectance, float roughness
){
    vec3 h = normalize(omega_i + omega_o);
    float alpha = roughness * roughness;

    float product_i = clamp(dot(n, omega_i), 0.0, 1.0);
    float product_o = clamp(dot(n, omega_o), 0.0, 1.0);

    // The diffuse term of the BRDF
    vec3 fresnel_term = fresnel(omega_i, h, reflectance);
    vec3 diffuse_term = (color / PI) * (vec3(1.0) - fresnel_term);
    
    // The specular term of the BRDF
    float denominator = max(4.0 * product_i * product_o, EPSILON);
    vec3 specular_term = (fresnel_term * ggx_normal(n, h, alpha) * 
        full_geometry(omega_i, omega_o, n, alpha)) 
        / denominator;

    return diffuse_term + specular_term;
}


//=============================|| RAY TRACING ||==============================//


// This is the L_e(x, omega) emitted light function.
// Here x is the point on the surface that emits the light, and omega
// is the direction in which the light is being emitted.
vec3 emitted_light(
    vec3 x, vec3 omega_o, 
    vec3 light_color, float light_intensity
){
    return light_intensity * light_color;
}


// Calculates one Monte-Carlo sample of the kth light bounce.
// We have:
//  T^kL_e = int_{M^k} [ 
//      L_e(x_{k+1} from x_k) * product_{k=1}^{n} [ 
//          f(x_{k-1}, x_k, x_{k+1})
//          * G(x_k, x_{k+1}) * V(x_k, x_{k+1}) dA(x_{k+1}) 
//      ]
//  ]
// We know that M = S_1 union S_2 union ... S_n
// So the integral over M is a sum of integrals over each surface:
// T^kL_e = Sum_{i = 1}^m int_{S_i^K} ... 
// Then, each surface can be parameterized between  0 and 1, so that
// we have:
// int_M f(x) dx = int_0^1 int_0^1 f(T(t)) * change_of_measure dt
// Same for int_{M^k}.
// We can then get a Monte_Carlo sample:
// f(T(t)) * change_of_measure
// Or in our case:
//  L_e(T(x_{k+1}) from T(x_k)) * product_{k=1}^{n} [ 
//          f(T(x_{k-1}), T(x_k), T(x_{k+1}))
//      * G(T(x_k), T(x_{k+1})) * V(T(x_k), T(x_{k+1})) 
//  ]
// Note that the product term is accumulated and saved from the last
// bounce, so it does not need to be recomputed. It is sometimes called
// the throughput.
// Note that this function should only be used for k > 1, since
// k = 0 does not involve an integral term. 
// Note that the sampling is uniform.
// Note that uv is the same as x2, but given as local parameters of
// the surface on which x2 lies.
Bounce_data light_bounce(
    inout Triangle scene[MAX_SCENE_SIZE], int scene_index_1, 
    int scene_index_2, vec3 throughput, 
    vec3 x0, vec3 x1, vec3 x2, vec2 uv,
    vec3 light_color, float light_intensity,
    int scene_size
){
    Bounce_data result;

    vec3 omega_i = -normalize(x1 - x0);
    vec3 omega_o = -normalize(x1 - x2);
    vec3 n1 = triangle_normal(scene[scene_index_1], x1);
    vec3 n2 = triangle_normal(scene[scene_index_2], x2);

    vec3 reflectance = scene[scene_index_1].reflectance;
    float roughness = scene[scene_index_1].roughness;

    // Shifting points slightly to ensure no self collision with the
    // inside of the triangle.
    x1 = x1 + n1 * EPSILON;
    x2 = x2 + n2 * EPSILON;

    // If the throughput is zero, then at some point the visibility term
    // must have becomes= 0; the ray after that vanishes, as it 
    // is obstructed.
    if(throughput == vec3(0.0)){ 
        result.throughput = vec3(0.0);
        result.radiance = vec3(0.0);
        return result;
    }

    result.throughput = throughput;
    result.throughput *= geometry_term(x1, x2, n1, n2);
    result.throughput *= cook_torrance_BRDF(omega_i, omega_o, n1, 
        scene[scene_index_1].color, reflectance, roughness);
    result.throughput *= change_of_measure(scene[scene_index_2], uv.x, uv.y);
    result.throughput *= visibility_term(scene, x1, x2, scene_size);

    // If the current sampled point is not on a light source, it
    // contributes no radiance, but future bounce may still 
    // contribute some, so we update the througput but keep the radiance
    // at 0.0. Otherwise we update the radiance.
    if(scene[scene_index_2].is_light_source){
        result.radiance = result.throughput 
            * emitted_light(x2, omega_o, light_color, light_intensity);
    }
    else{
        result.radiance = vec3(0.0);
    }
    
    return result;
}


// Generates several Monte-Carlo samples, that is, several rays of light
// whose bounces we trace throughout the scene.
vec3 monte_carlo_sampling(
    inout Triangle scene[MAX_SCENE_SIZE], float total_area,
    int sample_count, int bounce_count,
    vec3 camera_pos, vec3 screen_pos, float frame,
    vec3 light_color, float light_intensity,
    int scene_size
){
    vec3 final_result = vec3(0.0);;

    // We will first generate the primary ray. We just return a vector
    // of 0 values if it doesn't intersect the scene, since this is a
    // background.
    // The primary ray is in principle the same for all samples,
    // but we will instead recalculate it for each sample, which allows
    // us to introduce random jitter to the screen_position, which in
    // turn gives us an anti-aliasing effect.

    // Otherwise we generate cample_count monte-carlo samples, 
    // each with bouncd_count light bounces.
    for(int sample = 0; sample < sample_count; sample++){

        // The Neumann expansion for one bounce
        vec3 sample_result = vec3(0.0);

        // Primary ray calculation and tracing
        Ray ray;
        ray.origin = camera_pos;

        // We can add anti-aliasing here
        if(ENABLE_ANTI_ALIASING){
        // We can introduce anti aliasing by adding some jitter
        // each frame to the primary ray.
            vec3 jitter_val = vec3(
                random(seed(screen_pos.xy, frame, 0, sample, 0)),
                random(seed(screen_pos.xy, frame, 0, sample, 1)),
                random(seed(screen_pos.xy, frame, 0, sample, 2))
            ) * 0.005 - vec3(0.0025);
            ray.dir = normalize(screen_pos + jitter_val - camera_pos);
        }
        else{
            ray.dir = normalize(screen_pos - camera_pos);
        }

        Ray_hit hit = ray_tracing_function(scene, ray, scene_size);
        if(!hit.hit){
            continue;
        }

        // First, for bounce 0, we just want to sample the emitted
        // light L_e on the object.
        // This is T^0 L_e
        if(scene[hit.index].is_light_source){
            vec3 omega_o = normalize(hit.coordinate - camera_pos);
            sample_result += emitted_light(hit.coordinate, omega_o,
                light_color, light_intensity);
        }

        int scene_index_1 = hit.index;
        int scene_index_2;

        vec3 x0 = camera_pos;
        vec3 x1 = hit.coordinate;
        vec3 x2;  

        vec3 throughput = vec3(1.0);

        // Then we do each of T^k L_e
        for(int bounce = 1; bounce < bounce_count; bounce++){

            // For a bounce, we first we randomly choose an object.
            // To ensure no bias, we choose objects with a probability
            // proportional to their surphace area.
            float rand = 
                random(seed(screen_pos.xy, frame, bounce, sample, 0));

            float accum = 0.0;
            int chosen_index = 0;
            for(int i = 0; i < scene_size; i++) {
                accum += triangle_area(scene[i]) / total_area;
                if (rand < accum) {
                    chosen_index = i;
                    break;
                }
            }
            scene_index_2 = chosen_index;

            vec2 uv;
            uv.x = random(seed(screen_pos.xy, frame, bounce, sample, 1));
            uv.y = random(seed(screen_pos.xy, frame, bounce, sample, 2));
            x2 = triangle_transform(scene[scene_index_2], uv.x, uv.y);

            Bounce_data data = light_bounce(
                scene, scene_index_1, scene_index_2, throughput, 
                x0, x1, x2, uv, light_color, light_intensity,
                scene_size
            );

            sample_result += data.radiance;

            // A small amount of ambient light
            sample_result += scene[hit.index].color * AMBIENT_LIGHT;

            throughput = data.throughput;
            x0 = x1;
            x1 = x2;
            scene_index_1 = scene_index_2;
        }
        
        final_result += sample_result;
    }

    final_result /= sample_count;
    return final_result;
}


//============================|| MAIN FUNCTION ||=============================//


layout(std140) uniform CubeData {
    // 12 triangles × 3 vertices = 36 vec4
    vec4 vertices[36];
};


void main() {

    if(pass == 0){

        float time = current_time;

        // The screen position is basically the current grid point on the
        // pixel grid that we are at. We get it from the FragPos (which 
        // is a fragment) on the quad (two triangles) whose vertices we 
        // sent to the vertex shader. We use the aspect ratio to ensure the 
        // screen size doesn't stretch the scene. 
        // Shooting a ray from the camera position to the screen position
        // gives us our first (primary) ray into the scene, which then
        // intersects the scene to determine where the objects that will
        // be rendered are.
        vec3 screen_pos = vec3(FragPos.x * aspect_ratio, FragPos.y, 
            FragPos.z);
        vec3 view_pos = camera_pos;

        // The total number of primitives is the number of primitives sent
        // the CPU and the ones generated in the shader (the light).
        int scene_size = primitive_count + 2;

        // Light
        Triangle s1, s2;

        s1.v1 = vec3(2.0, 2.0, -4.0);
        s1.v0 = vec3(-2.0, 2.0, -4.0);
        s1.v2 = vec3(-2.0, 2.0, 4.0);
        s1.color = vec3(0.0, 0.0, 1.0);
        s1.is_light_source = true;
        s1.roughness = 0.8;
        s1.reflectance = vec3(0.1);

        s2.v1 = vec3(2.0, 2.0, 4.0);
        s2.v0 = vec3(2.0, 2.0, -4.0);
        s2.v2 = vec3(-2.0, 2.0, 4.0);
        s2.color = vec3(0.0, 0.0, 0.0);
        s2.is_light_source = true;
        s2.roughness = 0.8;
        s2.reflectance = vec3(0.1);

        // The scene contains all of the geometry
        Triangle scene[MAX_SCENE_SIZE];
        scene[0] = s1;
        scene[1] = s2;
        for(int i = 0; i < scene_size; i++){
            Triangle s;
            s.v0 = vertices[0 + i*3].xyz * 0.5 + vec3(0, -0.5, 0);
            s.v1 = vertices[1 + i*3].xyz * 0.5 + vec3(0, -0.5, 0);
            s.v2 = vertices[2 + i*3].xyz * 0.5 + vec3(0, -0.5, 0);
            s.color = vec3(1.0, 0.0, 0.0);
            s.roughness = 0.1;
            s.reflectance = vec3(0.1);
            scene[i + 2] = s;
        }

        // Total scene area
        float total_area = 0;
        for(int i = 0; i < scene_size; i++){
            total_area += triangle_area(scene[i]);
        }

        float light_intensity = 1.0;
        // White
        vec3 light_color = vec3(1.0);

        vec3 radiance = monte_carlo_sampling(
            scene, total_area, SAMPLE_COUNT, BOUNCE_COUNT,
            view_pos, screen_pos, time,
            light_color, light_intensity, scene_size
        );
        
        // The output is saved onto a texture that becomes
        // the next frame's input, as well as the texture displayed in the
        // next pass this frame.
        if(USE_AVERAGE){
            // Mix with the texture containing accumulated colors.
            vec2 uv = FragPos.xy * 0.5 + vec2(0.5);
            vec3 average_radiance = (radiance + 
                texture(last_frame, uv).rgb * frame) / (frame + 1);
            FragColor = vec4(average_radiance, 1.0);
        }
        else{
            FragColor = vec4(radiance, 1.0);
        }

    }
    else if(pass == 1){

        vec2 uv = FragPos.xy * 0.5 + vec2(0.5);
        FragColor = texture(last_frame, uv);
    }
}

