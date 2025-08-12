#ifndef UBO_BUFFER_H
#define UBO_BUFFER_H

#include <GL/glew.h>
#include <vector>
#include <iostream>

class UBOBuffer {
private:
    GLuint ubo;
    size_t dataSize;
    GLenum usage;

    void checkGLError() const {
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }

public:
    UBOBuffer() : ubo(0), dataSize(0), usage(GL_STATIC_DRAW) {}

    ~UBOBuffer() {
        if (ubo != 0) {
            glDeleteBuffers(1, &ubo);
        }
    }

    void allocate(int sizeInBytes, GLenum drawType) {
        this->usage = drawType;
        this->dataSize = sizeInBytes;

        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeInBytes, nullptr, usage);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        checkGLError();
    }

    void uploadData(const std::vector<float>& data) {
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, data.size() * sizeof(float), data.data());
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        checkGLError();
    }

    void bindToBindingPoint(GLuint bindingPoint) const {
        glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, ubo);
    }

    GLuint getBufferID() const { return ubo; }

    size_t getSize() const { return dataSize; }
};

#endif
