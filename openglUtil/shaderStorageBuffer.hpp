#ifndef SSBO_BUFFER_H
#define SSBO_BUFFER_H

#include <GL/glew.h>
#include <vector>
#include <iostream>

class SSBOBuffer {
private:
    GLuint ssbo;
    size_t dataSize;
    GLenum usage;

    void checkGLError() const {
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }

public:

    SSBOBuffer() : ssbo(0), dataSize(0), usage(GL_STATIC_DRAW) {}

    ~SSBOBuffer() {
        if (ssbo != 0) {
            glDeleteBuffers(1, &ssbo);
        }
    }

    // Creates and allocate the buffer with given size, in bytes,
    // and a draw type (static or dynamic)
    void allocate(int sizeInBytes, GLenum drawType) {
        this->usage = usage;
        this->dataSize = sizeInBytes;

        glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeInBytes, nullptr, usage);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        checkGLError();
    }

    // Upload raw data (you can template this)
    void uploadData(std::vector<float> data) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.size(), data.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        checkGLError();
    }

    // Binds the buffer to a binding point (matches layout(binding = N))
    void bindToBindingPoint(GLuint bindingPoint) const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, ssbo);
    }

    // Maps the SSBO to CPU memory
    void* map(GLenum access = GL_READ_WRITE) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        void* ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, access);
        return ptr;
    }

    void unmap() {
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    GLuint getBufferID() const { return ssbo; }

    size_t getSize() const { return dataSize; }
};

#endif
