/*
	Class for a shader object.
    Allows users to send vertex data, define uniforms...
*/

#ifndef SHADER_H
#define SHADER_H

#include "shaderProgram.hpp"
#include "vertexBuffer.hpp"
#include "util.hpp"

class Shader {

protected:

    ShaderProgram shaderProgram;

    /*
        Below are functions that allow us to set uniforms of any type.
    */


public:

    Shader(
        // The vertex and fragment shader code
        const std::string& vertexShaderSource,
        const std::string& fragmentShaderSource
    ) : shaderProgram(
        Util::readFileToString(vertexShaderSource),
        Util::readFileToString(fragmentShaderSource)
    ){}


    ~Shader() {
        shaderProgram.~ShaderProgram();
        glUseProgram(0);
    }

    
    GLuint getShaderProgram() {
        return shaderProgram.getShaderProgram();
    }


    void useShaderProgram() const {
        glUseProgram(shaderProgram.getShaderProgram());
    }

    /*
        Funcion that renders the given VBO object using the shader.
        Just ensure the uniforms are set before the function is called.
        Also it is the responsibility of the caller to ensure that the
        vertex buffer data (attribute type and attribute count)
        actually match those expected by the shader,
        or there will be errors.
    */
    void render(const VertexBuffer& buffer) const {

        glUseProgram(shaderProgram.getShaderProgram());

        buffer.render();
        /*
            Goes back to using no program,(in case other shaders need
            to draw).
        */
        glUseProgram(0);
    }


      // mat4
    void setUniform(const std::string& name, const glm::mat4& value) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // vec2
    void setUniform(const std::string& name, const glm::vec2& value) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform2fv(location, 1, glm::value_ptr(value));
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // vec3
    void setUniform(const std::string& name, const glm::vec3& value) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform3fv(location, 1, glm::value_ptr(value));
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // vec4
    void setUniform(const std::string& name, const glm::vec4& value) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform4fv(location, 1, glm::value_ptr(value));
        }
        else {
            /// Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // mat4 array
    void setUniform(const std::string& name, const glm::mat4* value, int size) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniformMatrix4fv(location, size, GL_FALSE, glm::value_ptr(value[0]));
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // vec3 array
    void setUniform(const std::string& name, const glm::vec3* arr, int size) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform3fv(location, size, glm::value_ptr(arr[0]));
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // vec4 array
    void setUniform(const std::string& name, const glm::vec4* arr, int size) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform4fv(location, size, glm::value_ptr(arr[0]));
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // float
    void setUniform(const std::string& name, float value) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform1f(location, value);
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program.\n";
        }
        glUseProgram(0);
    }

    // int
    void setUniform(const std::string& name, int value) {
        glUseProgram(shaderProgram.getShaderProgram());
        GLint location = glGetUniformLocation(shaderProgram.getShaderProgram(), name.c_str());
        if (location != -1) {
            glUniform1i(location, value);
        }
        else {
            // Error handling if the uniform was not found
            std::cerr << "Uniform '" << name << "' not found in shader program. \n";
        }
        glUseProgram(0);
    }


    /*
        A function that sets a texture, be it a cube map or a normal
        2D texture.
    */
    void setTextureUniform(
        const std::string& name, 
        GLuint textureId, 
        GLenum textureType, 
        GLuint textureUnit
    ) {

        glUseProgram(shaderProgram.getShaderProgram());

        glActiveTexture(GL_TEXTURE0 + textureUnit);
        glBindTexture(textureType, textureId);
        
        GLint location = glGetUniformLocation(
            shaderProgram.getShaderProgram(), 
            name.c_str()
        );
        if (location != -1) {
            glUniform1i(location, textureUnit);
        }
        else {
            std::cerr << "Sampler uniform '" << name 
                << "' not found in shader program.\n";
        }

        /*
            We then always reset to use texture unit 0 after setting
            the texture.
        */
        glActiveTexture(GL_TEXTURE0);
        glUseProgram(0);
    }

};

#endif