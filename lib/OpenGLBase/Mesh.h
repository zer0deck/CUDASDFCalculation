/*
* This portion of code originally made by BennyQBD
* https://github.com/BennyQBD/ModernOpenGLTutorial/blob/master/
*/

#include <GL/glew.h>

#include "ObjectLoader.h"

struct Vertex
{
public:
	Vertex(const glm::vec3& pos, const glm::vec2& texCoord, const glm::vec3& normal)
	{
		this->pos = pos;
		this->texCoord = texCoord;
		this->normal = normal;
	}

	inline glm::vec3* GetPos() { return &pos; }
	inline glm::vec2* GetTexCoord() { return &texCoord; }
	inline glm::vec3* GetNormal() { return &normal; }

private:
	glm::vec3 pos;
	glm::vec2 texCoord;
	glm::vec3 normal;
};

enum MeshBufferPositions
{
	POSITION_VB,
	TEXCOORD_VB,
	NORMAL_VB,
	INDEX_VB
};

class Mesh
{
public:

    Mesh(const std::string& fileName) { 
        model = OBJModel(fileName).ToIndexedModel();
        InitMesh(); 
        };
	Mesh(Vertex* vertices, unsigned int numVertices, unsigned int* indices, unsigned int numIndices)
    {
        for(unsigned int i = 0; i < numVertices; i++)
        {
            model.positions.push_back(*vertices[i].GetPos());
            model.texCoords.push_back(*vertices[i].GetTexCoord());
            model.normals.push_back(*vertices[i].GetNormal());
        }
        
        for(unsigned int i = 0; i < numIndices; i++)
            model.indices.push_back(indices[i]);

        InitMesh();
    };

    inline IndexedModel getModel() { return model; };

	void Draw()
    {
        glBindVertexArray(vertexArrayObject);
        glDrawElementsBaseVertex(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0, 0);
        glBindVertexArray(0);
    };

	virtual ~Mesh()
    {
        glDeleteBuffers(NUM_BUFFERS, vertexArrayBuffers);
	    glDeleteVertexArrays(1, &vertexArrayObject);
    };
protected:
private:
    IndexedModel model;
	static const unsigned int NUM_BUFFERS = 4;
	void operator=(const Mesh& mesh) {}
	Mesh(const Mesh& mesh) {}

    void InitMesh()
    {
        numIndices = model.indices.size();

        glGenVertexArrays(1, &vertexArrayObject);
        glBindVertexArray(vertexArrayObject);

        glGenBuffers(NUM_BUFFERS, vertexArrayBuffers);
        
        glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffers[POSITION_VB]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(model.positions[0]) * model.positions.size(), &model.positions[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffers[TEXCOORD_VB]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(model.texCoords[0]) * model.texCoords.size(), &model.texCoords[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffers[NORMAL_VB]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(model.normals[0]) * model.normals.size(), &model.normals[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexArrayBuffers[INDEX_VB]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(model.indices[0]) * model.indices.size(), &model.indices[0], GL_STATIC_DRAW);

        glBindVertexArray(0);
    };

	GLuint vertexArrayObject;
	GLuint vertexArrayBuffers[NUM_BUFFERS];
	unsigned int numIndices;
};