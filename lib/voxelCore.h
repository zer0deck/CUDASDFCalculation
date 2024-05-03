/*
* Created by Aleksei Grandilevskii (zer0deck) in 2024;
* Please, refer to me if you are using this code in your work.
*/

// :MARK: Includes

#include <stdio.h>
#include <stdlib.h>

#include <Mesh.h>

class VoxelMesh
{
public:
    VoxelMesh(int voxelsPerSide, int numVoxels, float4 *voxelsInput, float sdfDepth)
    {
        voxelSize = 1.f/voxelsPerSide;
        indices = (unsigned int *)malloc(sizeof(unsigned int) * 36 * numVoxels);
        vertices = (Vertex *)malloc(sizeof(Vertex) * 24 * numVoxels);

        int counterA = 0;
        int counterB = 0;
        for (int i = 0; i < numVoxels; ++i)
        {
            if (abs(voxelsInput[i].w) > sdfDepth)
                continue;
            for (Vertex val : createCube(voxelsInput[i]))
            {
                vertices[counterA] = val;
                counterA++;
            }

            for  (unsigned int val : createIndicesBatch(i))
            {
                indices[counterB] = val;
                counterB++;
            }
        }

        mesh = new Mesh(vertices, 24 * numVoxels, indices, 36 * numVoxels);
    
    }

    void Draw()
    {
        mesh->Draw();
    }

private:
    float voxelSize;
    Mesh *mesh;
    unsigned int *indices;
    Vertex *vertices;

    std::array<Vertex, 24> createCube(float4 location)
    {
        std::array<Vertex, 24> cube = {
        {
            Vertex(glm::vec3(location.x - voxelSize,    location.y - voxelSize,     location.z - voxelSize), glm::vec2(1, 0), glm::vec3(0, 0, -1)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y + voxelSize,     location.z - voxelSize), glm::vec2(0, 0), glm::vec3(0, 0, -1)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y + voxelSize,     location.z - voxelSize), glm::vec2(0, 1), glm::vec3(0, 0, -1)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y - voxelSize,     location.z - voxelSize), glm::vec2(1, 1), glm::vec3(0, 0, -1)),

            Vertex(glm::vec3(location.x - voxelSize,    location.y - voxelSize,     location.z + voxelSize), glm::vec2(1, 0), glm::vec3(0, 0, 1)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y + voxelSize,     location.z + voxelSize), glm::vec2(0, 0), glm::vec3(0, 0, 1)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y + voxelSize,     location.z + voxelSize), glm::vec2(0, 1), glm::vec3(0, 0, 1)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y - voxelSize,     location.z + voxelSize), glm::vec2(1, 1), glm::vec3(0, 0, 1)),

            Vertex(glm::vec3(location.x - voxelSize,    location.y - voxelSize,     location.z - voxelSize), glm::vec2(0, 1), glm::vec3(0, -1, 0)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y - voxelSize,     location.z + voxelSize), glm::vec2(1, 1), glm::vec3(0, -1, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y - voxelSize,     location.z + voxelSize), glm::vec2(1, 0), glm::vec3(0, -1, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y - voxelSize,     location.z - voxelSize), glm::vec2(0, 0), glm::vec3(0, -1, 0)),

            Vertex(glm::vec3(location.x - voxelSize,    location.y + voxelSize,     location.z - voxelSize), glm::vec2(0, 1), glm::vec3(0, 1, 0)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y + voxelSize,     location.z + voxelSize), glm::vec2(1, 1), glm::vec3(0, 1, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y + voxelSize,     location.z + voxelSize), glm::vec2(1, 0), glm::vec3(0, 1, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y + voxelSize,     location.z - voxelSize), glm::vec2(0, 0), glm::vec3(0, 1, 0)),

            Vertex(glm::vec3(location.x - voxelSize,    location.y - voxelSize,     location.z - voxelSize), glm::vec2(1, 1), glm::vec3(-1, 0, 0)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y - voxelSize,     location.z + voxelSize), glm::vec2(1, 0), glm::vec3(-1, 0, 0)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y + voxelSize,     location.z + voxelSize), glm::vec2(0, 0), glm::vec3(-1, 0, 0)),
            Vertex(glm::vec3(location.x - voxelSize,    location.y + voxelSize,     location.z - voxelSize), glm::vec2(0, 1), glm::vec3(-1, 0, 0)),

            Vertex(glm::vec3(location.x + voxelSize,    location.y - voxelSize,     location.z - voxelSize), glm::vec2(1, 1), glm::vec3(1, 0, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y - voxelSize,     location.z + voxelSize), glm::vec2(1, 0), glm::vec3(1, 0, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y + voxelSize,     location.z + voxelSize), glm::vec2(0, 0), glm::vec3(1, 0, 0)),
            Vertex(glm::vec3(location.x + voxelSize,    location.y + voxelSize,     location.z - voxelSize), glm::vec2(0, 1), glm::vec3(1, 0, 0))
        }
        };
        return cube;

    }

    std::array<unsigned int, 36> createIndicesBatch(unsigned int multiplier)
    {
        unsigned int i = multiplier * 24;
        std::array<unsigned int, 36> batch = {
            {
            i+0,    i+1,    i+2,    i+0,    i+2,    i+3,    i+6,    i+5,    i+4, 
            i+7,    i+6,    i+4,    i+10,   i+9,    i+8,    i+11,   i+10,   i+8, 
            i+12,   i+13,   i+14,   i+12,   i+14,   i+15,   i+16,   i+17,   i+18, 
            i+16,   i+18,   i+19,   i+22,   i+21,   i+20,   i+23,   i+22,   i+20 
            }
        };
        return batch;
    }

};
