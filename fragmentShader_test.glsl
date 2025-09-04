

//=============================||  PARAMETERS  ||=============================//


#version 430 core

in vec3 FragPos;
out vec4 FragColor;

uniform float aspect_ratio;
uniform vec3 camera_pos;
uniform float current_time;
uniform int frame;

// We can output each frame onto a texture. Then in the first pass = 0
// we can accumulate the output of this frame with the previous frames'
// outputs, and then in the second pass = 1 we can display the
// result in real time.
uniform sampler2D last_frame;
uniform int pass;

#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define SCENE_SIZE 4
#define LIMIT 999999
#define SAMPLE_COUNT 256
#define BOUNCE_COUNT 4
#define ENABLE_ANTI_ALIASING true
#define USE_AVERAGE true


//===============================||  OBJECTS  ||==============================//


// A sphere
struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    bool is_light_source;
    float roughness;
    vec3 reflectance;
};


// A ray
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


//==============================||  COLLISIONS  ||============================//


// Sphere (implict) and ray (parametric) intersection.
// Leads to a quadratic; if there are one or more than one real 
// and positive solutions are found then we have a hit.
// If two hits are found, we return the closer one (smaller positive t).
Ray_hit ray_sphere_intersection(
    Sphere sphere,
    Ray ray
){
    Ray_hit result;
    result.hit = false;

    float a = dot(ray.dir, ray.dir);
    float b = 2 * dot(ray.dir, (ray.origin - sphere.center));
    float c = dot(ray.origin - sphere.center, ray.origin - sphere.center) 
        - sphere.radius * sphere.radius;
    float discriminant = b*b - 4*a*c;
    
    // If discriminant is equal to 0, or close enough to 0
    if (abs(discriminant) < EPSILON){
        float t = -b / (2*a);
        if(t > 0){
            result.coordinate = ray.dir * t + ray.origin;
            result.hit = true;
            result.t = t;
        }
    }
    else if (discriminant > 0){
        // We check the closer one first, then the further one only
        // if the closer one fails.
        float t = (-b - sqrt(discriminant))/ (2*a);
        if(t >= 0){
            result.coordinate = ray.dir * t + ray.origin;
            result.hit = true;
            result.t = t;
        }
        else{
            // If not check the further one
            t = (-b + sqrt(discriminant))/ (2*a);
            if(t >= 0){
                result.coordinate = ray.dir * t + ray.origin;
                result.hit = true;
                result.t = t;
            }
        }
    }

    return result;
}


// A funtion that casts a ray into the scene, and returns the closest
// intersection.
Ray_hit ray_tracing_function(
    inout Sphere scene[SCENE_SIZE],
    Ray ray
) {
    Ray_hit hit;
    hit.hit = false;
    hit.t = LIMIT;
    
    for(int i = 0; i < SCENE_SIZE; i++){

        Ray_hit hit_temp = ray_sphere_intersection(scene[i], ray);
        hit_temp.index = i;

        // If there is a hit, and the hit occurs before reaching y
        // (with epsilon to avoid self intersection)
        if(hit_temp.hit && hit_temp.t < hit.t){
            hit = hit_temp;
        }
    }

    return hit;
}


//==========================|| CHANGE OF MEASURE ||===========================//


// Returns the sphere normal, assuming the point is on the sphere
vec3 sphere_normal(Sphere sphere, vec3 point) {
    return normalize(point - sphere.center);
}


// Returns the sphere surface area
float sphere_area(Sphere sphere) {
    return 4 * PI * sphere.radius * sphere.radius;
}


// Generates a point on a sphere using parameters between 0 and 1.
vec3 sphere_transform(Sphere sphere, float u, float v){
    return  sphere.center + sphere.radius * vec3(
        sin(PI * u) * cos(2 * PI * v),
        sin(PI * u) * sin(2 * PI * v),
        cos(PI * u)
    );
}


// The change of measure term, or sqrt(det(J^TJ)) where J is the
// Jacobian matrix of the transformation.
float change_of_measure(Sphere sphere, float u, float v){
    return 2 * PI * PI * sphere.radius * sphere.radius * sin(PI * u);
}


// The inverse transform, that takes a point in and returns two
// parameters u and v between 0 and 1.
vec2 get_sphere_parameters(Sphere sphere, vec3 coordinate){
    
    // The direction of the point relative to the center
    vec3 p = normalize(coordinate - sphere.center);
    float theta = acos(clamp(p.z, -1.0, 1.0));
    float phi = atan(p.y, p.x);
    if (phi < 0.0) {
        phi += 2.0 * PI;
    }
    float u = theta / PI;
    float v = phi / (2.0 * PI);

    return vec2(u, v);
}


//=========================|| VISIBILITY FUNCTION ||==========================//


// In order to check if a point x is visible to y, we need to ensure
// there are no collisions when casting a ray from x to y.
// Note that we only check for collisions when t is between 0
// and the distance between x and y, as we only care for collisions
// between x and y.
float visibility_term(
    inout Sphere scene[SCENE_SIZE],
    vec3 x, vec3 y
) {
    Ray ray;
    // We add epsilon to avoid self intersection
    ray.dir = normalize(y - x);
    ray.origin = x + ray.dir * EPSILON;

    float dist = length(x - y);

    // We cast a ray from x to y, and check for any collision
    for(int i = 0; i < SCENE_SIZE; i++){
        Ray_hit hit = ray_sphere_intersection(scene[i], ray);

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
vec4 seed(vec2 pixel, float frame, int bounce, int mt_sample, int parameter){
    return vec4(pixel.x, pixel.y, frame, 
        parameter * 10000 + bounce * 100 + mt_sample);
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


// Calculates one Monte-Carlo mt_sample of the kth light bounce.
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
// We can then get a Monte_Carlo mt_sample:
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
    inout Sphere scene[SCENE_SIZE], int sphere_index_1, 
    int sphere_index_2, vec3 throughput, 
    vec3 x0, vec3 x1, vec3 x2, vec2 uv,
    vec3 light_color, float light_intensity
){
    Bounce_data result;

    vec3 omega_i = -normalize(x1 - x0);
    vec3 omega_o = -normalize(x1 - x2);
    vec3 n1 = sphere_normal(scene[sphere_index_1], x1);
    vec3 n2 = sphere_normal(scene[sphere_index_2], x2);

    vec3 reflectance = scene[sphere_index_1].reflectance;
    float roughness = scene[sphere_index_1].roughness;

    // Shifting points slightly to ensure no self collision with the
    // inside of the sphere.
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
        scene[sphere_index_1].color, reflectance, roughness);
    result.throughput *= change_of_measure(scene[sphere_index_2], uv.x, uv.y);
    result.throughput *= visibility_term(scene, x1, x2);

    // If the current sampled point is not on a light source, it
    // contributes no radiance, but future bounce may still 
    // contribute some, so we update the througput but keep the radiance
    // at 0.0. Otherwise we update the radiance.
    if(scene[sphere_index_2].is_light_source){
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
    inout Sphere scene[SCENE_SIZE], float total_area,
    int sample_count, int bounce_count,
    vec3 camera_pos, vec3 screen_pos, float frame,
    vec3 light_color, float light_intensity
){
    vec3 final_result = vec3(0.0);;

    // We will first generate the primary ray. We just return a vector
    // of 0 values if it doesn't intersect the scene, since this is a
    // background.
    // The primary ray is in principle the same for all samples,
    // but we will instead recalculate it for each mt_sample, which allows
    // us to introduce random jitter to the screen_position, which in
    // turn gives us an anti-aliasing effect.

    // Otherwise we generate cample_count monte-carlo samples, 
    // each with bouncd_count light bounces.
    for(int mt_sample = 0; mt_sample < sample_count; mt_sample++){

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
                random(seed(screen_pos.xy, frame, 0, mt_sample, 0)),
                random(seed(screen_pos.xy, frame, 0, mt_sample, 1)),
                random(seed(screen_pos.xy, frame, 0, mt_sample, 2))
            ) * 0.005 - vec3(0.0025);
            ray.dir = normalize(screen_pos + jitter_val - camera_pos);
        }
        else{
            ray.dir = normalize(screen_pos - camera_pos);
        }

        Ray_hit hit = ray_tracing_function(scene, ray);
        if(!hit.hit){
            continue;
        }

        // First, for bounce 0, we just want to mt_sample the emitted
        // light L_e on the object.
        // This is T^0 L_e
        if(scene[hit.index].is_light_source){
            vec3 omega_o = normalize(hit.coordinate - camera_pos);
            sample_result += emitted_light(hit.coordinate, omega_o,
                light_color, light_intensity);
        }

        int sphere_index_1 = hit.index;
        int sphere_index_2;

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
                random(seed(screen_pos.xy, frame, bounce, mt_sample, 0));
            float accum = 0.0;
            int chosen_index = 0;
            for(int i = 0; i < SCENE_SIZE; i++) {
                accum += sphere_area(scene[i]) / total_area;
                if (rand < accum) {
                    chosen_index = i;
                    break;
                }
            }
            sphere_index_2 = chosen_index;

            vec2 uv;
            uv.x = random(seed(screen_pos.xy, frame, bounce, mt_sample, 1));
            uv.y = random(seed(screen_pos.xy, frame, bounce, mt_sample, 2));
            x2 = sphere_transform(scene[sphere_index_2], uv.x, uv.y);

            Bounce_data data = light_bounce(
                scene, sphere_index_1, sphere_index_2, throughput, 
                x0, x1, x2, uv, light_color, light_intensity
            );

            sample_result += data.radiance;

            throughput = data.throughput;
            x0 = x1;
            x1 = x2;
            sphere_index_1 = sphere_index_2;
        }
        
        final_result += sample_result;
    }

    final_result /= sample_count;
    return final_result;
}


//============================|| MAIN FUNCTION ||=============================//


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

        // The third sphere is the light source
        Sphere s1, s2, s3, s4;

        s1.center = vec3(0.0, -0.4, 0.0);
        s1.radius = 0.5;
        s1.color = vec3(1.0, 0.0, 0.0);
        s1.roughness = 0.2;
        s1.reflectance = vec3(0.2);

        s2.center = vec3(0.0, 0.4, 0.0);
        s2.radius = 0.25;
        s2.color = vec3(0.0, 1.0, 0.0);
        s2.roughness = 0.2;
        s2.reflectance = vec3(0.2);
    
        s3.center = vec3(0.15, 0.2, -0.3);
        s3.radius = 0.15;
        s3.color = vec3(0.0, 0.0, 1.0);
        s3.roughness = 0.2;
        s3.reflectance = vec3(0.2);

        s4.center = vec3(0.0, 10.0, 0.0);
        s4.radius = 8.0;
        s4.color = vec3(0.0, 0.0, 0.0);
        s4.is_light_source = true;
        s4.roughness = 0.8;
        s4.reflectance = vec3(0.1);

        // The scene contains all of the geometry
        Sphere scene[SCENE_SIZE];
        scene[0] = s1;
        scene[1] = s2;
        scene[2] = s3; 
        scene[3] = s4; 

        // Total scene area
        float total_area = 0;
        for(int i = 0; i < SCENE_SIZE; i++){
            total_area += sphere_area(scene[i]);
        }

        float light_intensity = 1.0;
        // White
        vec3 light_color = vec3(1.0);

        vec3 radiance = monte_carlo_sampling(
            scene, total_area, SAMPLE_COUNT, BOUNCE_COUNT,
            view_pos, screen_pos, time,
            light_color, light_intensity
        );
        
        // The ouput is saved onto a texture that becomes
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

