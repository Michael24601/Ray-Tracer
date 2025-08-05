/*
    A Utility class.
*/

#ifndef UTIL_H
#define UTIL_H

// Macros for STB
#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <GL/glew.h>
#include <GL/gl.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stb_image_write.h>
#include <stb_image.h>

class Util{

public:

    // Reads a file to a string
    static std::string readFileToString(const std::string& filePath) {

        std::ifstream file(filePath);
        std::stringstream buffer;

        if (file) {
            buffer << file.rdbuf();
            return buffer.str();
        }
        else {
            std::cerr << "Error: Unable to open file " << filePath << "\n";
            exit(1);
        }
    }

    // Loads an opengl texture
    static GLuint loadTexture(const std::string& imagePath, 
        int& w, int& h) {

        int width, height, channels;
        /*
            Here we have to flip the image vertically. This is necessary
            because OpenGL expects the bottom-left corner to be the origin.
        */
        stbi_set_flip_vertically_on_load(true);
        unsigned char* pixels = stbi_load(
            imagePath.c_str(), 
            &width, &height, 
            &channels, 0
        );
        if (!pixels) {
            std::cerr << "Error: Failed to load image: " << imagePath << "\n";
            return 0;
        }

        // Resetting the flip setting to avoid affecting other image loads
        stbi_set_flip_vertically_on_load(false);

        // png generally has 4 channels, while jpeg generally has 3 (no opacity)
        GLenum textureFormat;
        if (channels == 3) {
            textureFormat = GL_RGB;
        }
        else if (channels == 4) {
            textureFormat = GL_RGBA;
        }
        else {
            // If the format is unsupported, the memory allocated is freed
            std::cerr << "Error: Unsupported number of channels: " << channels << "\n";
            stbi_image_free(pixels);
            return 0;
        }


        // This part adds the image data to the texture
        GLuint textureID;
        glGenTextures(1, &textureID);

        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            textureFormat,
            width,
            height,
            0,
            textureFormat,
            GL_UNSIGNED_BYTE,
            pixels
        );

        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        h = height;
        w = width;

        return textureID;
    }


    static void initializeTexture(
        GLuint& texture, GLuint& fbo, int width, int height
    ){
        // Just setting up texture2 and binding it to a buffer object.
        // Reference: https://learnopengl.com/Advanced-OpenGL/Framebuffers
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
            GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
            GL_TEXTURE_2D, texture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Error.\n";
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }


    static void initializeTexture2(
        GLuint& texture, GLuint& fbo, int width, int height
    ){
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F,
            width, height, 0,
            GL_RGBA, GL_FLOAT,
            nullptr
        );

        // Necessary options for the spectral method
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0
        );

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Error.\n";
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

#endif