/*
* This portion of code originally made by BennyQBD
* https://github.com/BennyQBD/ModernOpenGLTutorial/blob/master/
*/

#include <iostream>
#include <string>
#include <GL/glew.h>

#include "stb_image.h"

class Texture
{
public:
	Texture(const std::string& fileName)
    {
        int width, height, numComponents;
        unsigned char* data = stbi_load((fileName).c_str(), &width, &height, &numComponents, 4);

        if(data == NULL)
            std::cerr << "Unable to load texture: " << fileName << std::endl;
            
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
            
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    };

	void Bind() { glBindTexture(GL_TEXTURE_2D, texture); };
	virtual ~Texture() { glDeleteTextures(1, &texture); };
protected:
private:
	Texture(const Texture& texture) {}
	void operator=(const Texture& texture) {}

	GLuint texture;
};