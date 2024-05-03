/*
* Created by Aleksei Grandilevskii (zer0deck) in 2024;
* Please, refer to me if you are using this code in your work.
* Link: github.com/zer0deck
* Email: zer0deck.work@icloud.com
* My nvcc path /usr/local/cuda-12.4/bin/nvcc
* Check MAKEFILE is you have another
*/

// :MARK: Includes

#include <stdio.h>
#include <stdlib.h>
// #include <chrono>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <Display.h>
#include <Shader.h>
#include <Texture.h>

#include <Float3OpsOverload.h>
#include <voxelCore.h>

// :MARK: Space setup

struct polygon3 {
    float3 a,b,c;
    polygon3(float3 v0, float3 v1, float3 v2)
        : a(v0), b(v1), c(v2) {}
};

// using namespace std;

// :MARK: Constants defenition

__constant__ int TEXTURE_SIZE;
__constant__ int NUM_VOXELS;
__constant__ int NUM_POLYGONS;
__constant__ float EPSILON;
__constant__ float3 MIN_VERTEX, MAX_VERTEX;

static const int DISPLAY_H = 900;
static const int DISPLAY_W = 1200;

// :MARK: Functions declaration

__device__ int calculateID();
__device__ float3 convertTo3D(int id);
__device__ float3 convertToPos(float3 id);
__device__ int getNearestpolygon(float3 pos, polygon3 *deviceInput);
__device__ float calculateSDF(float3 position, polygon3 polygon);
__device__ int IntersectionCount(float3 pos, polygon3 *deviceInput);
__device__ int RayIntersectsTriangle(float3 pos, polygon3 polygon);
__device__ float calculateBrightness(float3 pos, int nearestTriangleID, polygon3 *deviceInput);
__global__ void SignedDistanceField(polygon3 *deviceInput, float4 *deviceOutput);