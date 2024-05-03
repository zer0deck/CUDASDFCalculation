/*
* Created by Aleksei Grandilevskii (zer0deck) in 2024;
* Please, refer to me if you are using this code in your work.
*/

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Add
__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ float3 operator+(const float3 &a, const int b) {
  return make_float3(a.x+b, a.y+b, a.z+b);
}
__device__ float3 operator+(const float3 &a, const float b) {
  return make_float3(a.x+b, a.y+b, a.z+b);
}

// Substract 
__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float3 operator-(const float3 &a, const int b) {
  return make_float3(a.x-b, a.y-b, a.z-b);
}
__device__ float3 operator-(const float3 &a, const float b) {
  return make_float3(a.x-b, a.y-b, a.z-b);
}

// Multiply
__device__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float3 operator*(const float3 &a, const int b) {
  return make_float3(a.x*b, a.y*b, a.z*b);
}
__device__ float3 operator*(const float3 &a, const float b) {
  return make_float3(a.x*b, a.y*b, a.z*b);
}

// Divide
__device__ float3 operator/(const float3 &a, const float3 &b) {
  return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}
__device__ float3 operator/(const float3 &a, const int b) {
  return make_float3(a.x/b, a.y/b, a.z/b);
}
__device__ float3 operator/(const float3 &a, const float b) {
  return make_float3(a.x/b, a.y/b, a.z/b);
}

// Dot product
__device__ float operator^(const float3 &a, const float3 &b)
{
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

// Cross product
// Reference: https://registry.khronos.org/OpenGL-Refpages/gl4/html/cross.xhtml
__device__ float3 operator|(const float3 &a, const float3 &b)
{
  return make_float3(a.y*b.z - b.y*a.z, a.z*b.x - b.z*a.x, a.x*b.y - b.x*a.y);
}

__device__ int signCUDA(float x)
{
	return x > 0.f ? 1 : x < 0.f ? -1 : 0;
}
__device__ float clampCUDA(float x)
{
	return x > 1.f ? 1 : x < -1.f ? -1 : x;
}
__device__ float dot2CUDA(float3 x)
{
  return x^x;
}
__device__ float3 normalizeCUDA(float3 x)
{
  float z = sqrtf(powf(x.x, 2) + powf(x.y, 2) + powf(x.z, 2)); 
  return x/z;
}