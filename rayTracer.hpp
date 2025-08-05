#define GLEW_STATIC

#include <GL/glew.h>
#include <GL/gl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <GL/gl.h>
#include <cfloat>
#include <limits>
#include <iostream>
#include <vector>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

#include "openglUtil/glfwWindowWrapper.hpp"
#include "openglUtil/shader.hpp"
#include "openglUtil/vertexBuffer.hpp"


void runRayTracer(){

    GlfwWindowWrapper window(1000, 750, 4, "Window", true);
    float aspectRatio =  (float)window.getWidth() / (float)window.getHeight();

    Shader shader("vertexShader.glsl", "fragmentShader.glsl");
    /*
        We want to define the whole geometry inside the fragment shader,
        so we can just send two triagles to the vertex shader as a
        background.
        We can think of this as the grid of pixels onto which the
        scene gets captured. 
    */
    VertexBuffer emptyBuffer(6, std::vector<unsigned int>{3}, 3, GL_STATIC_DRAW);
    // Note that the grid of pixels we shoot a ray into has to be
    // within the clip space (-1, -1, -1) to (1, 1, 1).
    emptyBuffer.setData(std::vector<float>{
        1, 1, -1, -1, 1, -1, -1, -1, -1,        // Triangle 1 (CCW)
        1, 1, -1, -1, -1, -1, 1, -1, -1         // Triangle 2 (CCW)
    });
    /*
        The camera can be imagined as being a point sort of behind
        the grid of pixels onto which the scene is flattened.
        There is no camera matrix, just a camera position (ray origin)
        from where we shoot rays onto the pixel grid (quad defined
        by the vertices above). 
    */
    shader.setUniform("camera_pos", glm::vec3(0, 0, -2));
    shader.setUniform("aspect_ratio", aspectRatio);

    // This is just a texture, used to save the output and then
    // display it. We can't directly display teh output because we need
    // it preserved as an input for the next frame's computation.
    int width = window.getWidth(), height = window.getHeight();
    GLuint texture0, texture1;
    GLuint fbo0, fbo1;
    Util::initializeTexture(texture0, fbo0, width, height);
    Util::initializeTexture(texture1, fbo1, width, height);

    // This is just framerate management
    float lastTime = glfwGetTime();
    float deltaTime = 0.0;
    float framesPerSecond = 60;
    float frameRate = 1.0 / framesPerSecond;

    float time = 0.0;
    int frame = 0;

    glfwSetFramebufferSizeCallback(
        window.getWindow(),
        window.framebuffer_size_callback
    );
    while (!glfwWindowShouldClose(window.getWindow())) {

        float currentTime = glfwGetTime();
        deltaTime += (currentTime - lastTime);
        time += (currentTime - lastTime);
        lastTime = currentTime;

        // Event polling
        glfwPollEvents();
        window.processInput();

        // Events
        if (glfwGetKey(window.getWindow(), GLFW_KEY_A) == GLFW_PRESS) {
            
        }

        // Time sent to shader
        shader.setUniform("current_time", time);
        shader.setUniform("frame", frame);

        if (deltaTime >= frameRate) {

            // First render onto texture1, to save this frame's output
            glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                GL_TEXTURE_2D, texture1, 0);
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);

            shader.setUniform("pass", 0);
            // Use the last frame's input, texture0
            shader.setTextureUniform("last_frame", texture0, GL_TEXTURE_2D, 0);
            shader.render(emptyBuffer);   

            // Second render onto the screen
            // This part just tells Opengl where to render to
            // (the window's framebuffer)
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, window.getWidth(), window.getHeight());
            // This is the default background color
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            shader.setUniform("pass", 1);
            // When displaying, use the newly accumulated texture
            shader.setTextureUniform("last_frame", texture1, GL_TEXTURE_2D, 0);
            shader.render(emptyBuffer);

            // Then switch the textures for ping-pong rendering.
            std::swap(texture0, texture1);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                GL_TEXTURE_2D, texture1, 0);

            glfwSwapBuffers(window.getWindow());

            // Sets delta time back to 0.0
            deltaTime = 0.0f;
            frame++;
        }
    }
}