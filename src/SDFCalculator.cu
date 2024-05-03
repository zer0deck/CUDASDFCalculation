/*
* Created by Aleksei Grandilevskii (zer0deck) in 2024;
* Please, refer to me if you are using this code in your work.
* Link: github.com/zer0deck
* Email: zer0deck.work@icloud.com
* My nvcc path /usr/local/cuda-12.4/bin/nvcc
* Check MAKEFILE is you have another
*/

#include "SDFCalculator.h"

// :MARK: CUDA functions

__device__ int calculateID()
{
	int blockID = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	return blockID * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ float3 get3dID(int threadID) {
	return make_float3 (threadID % TEXTURE_SIZE, 
		(threadID / TEXTURE_SIZE) % TEXTURE_SIZE, 
		(threadID / (TEXTURE_SIZE * TEXTURE_SIZE)) % TEXTURE_SIZE);
}

__device__ float3 convertToPos(int threadID) { 
	return ( get3dID(threadID) + make_float3(.5f, .5f, .5f) ) / 
	TEXTURE_SIZE * ( MAX_VERTEX - MIN_VERTEX ) +
	MIN_VERTEX;
}

__device__ int getNearestPolygon(float3 pos, polygon3 *deviceInput) {
	int id = 0;
	float bestDistance = calculateSDF(pos, deviceInput[id]);
	for (int t = 0; t < NUM_POLYGONS; t++) {
		float currentDistance = calculateSDF(pos, deviceInput[t]);
		if (currentDistance < bestDistance) {
			id = t;
			bestDistance = currentDistance;
		}
	}
	return id;
}

__device__ int intersectionCheck(float3 pos, polygon3 polygon) {

	float3 e0, e1, e2, h, q;
	float a, u, v, t;

	e0 = pos - polygon.a;
	e1 = polygon.b - polygon.a;
	e2 = polygon.c - polygon.a;

	h = make_float3(0.f, 1.f, 0.f)|e2;
	a = e1^h;

	if (fabsf(a) < EPSILON)
		return 0;

	u = (e0^h) / a;

	if (u < 0.0 || u > 1.0)
		return 0;

	q = e0|e1;
	v = (make_float3(0.f, 1.f, 0.f)^q) / a;

	if (v < 0.0 || u + v > 1.0)
		return 0;

	t = (e2^q) / a;

	return t >= 0.0;
}

__device__ int calculateDirection(float3 pos, polygon3 *deviceInput) {
	int count = 0;

	for (int t = 0; t < NUM_POLYGONS; t++) {
		count += intersectionCheck(pos, deviceInput[t]);
	}

	return (count % 2 == 0) ? 1 : -1;
}

__device__ float calculateBrightness(float3 pos, int nPolygonID, polygon3 *deviceInput)
{
	return calculateDirection(pos, deviceInput) * calculateSDF(pos, deviceInput[nPolygonID]);
}

/* Returns the unsigned distance between the input position and triangle.
Developed by Inigo Quilez. */
__device__ float calculateSDF(float3 position, polygon3 polygon) {
	float3 ba = polygon.b - polygon.a;
	float3 cb = polygon.c - polygon.b;
	float3 ac = polygon.a - polygon.c;
	float3 pa = position - polygon.a;
	float3 pb = position - polygon.b;
	float3 pc = position - polygon.c;

	float3 nor = normalizeCUDA(ba|ac);

	if (((signCUDA((ba|nor) ^ pa) + signCUDA((cb|nor) ^ pb) + signCUDA((ac|nor) ^ pc)) ) < 2.f) {
		float x = dot2CUDA(ba * clampCUDA((ba^pa) / dot2CUDA(ba)) - pa);
		float y = dot2CUDA(cb * clampCUDA((cb^pb) / dot2CUDA(cb)) - pb);
		float z = dot2CUDA(ac * clampCUDA((ac^pc) / dot2CUDA(ac)) - pc);
		return sqrtf(min(min(x, y), z));
	}

	return sqrtf((nor^pa)*(nor^pa) / (nor^nor));
}

/* 
Special thanks for Daniel Shervheim and his Unity SDF calculator for some cool solutions.
*/
__global__ void SignedDistanceField(polygon3 *deviceInput, float4 *deviceOutput) 
{
	int threadID = calculateID();
	if (threadID >= NUM_VOXELS)
		return;

	float3 pos = convertToPos(threadID);
	int nPolygonID = getNearestPolygon(pos, deviceInput);

	// float3 currentNormal = normalizeCUDA(
	// 	(deviceInput[nPolygonID].b - deviceInput[nPolygonID].a) | 
	// 	(deviceInput[nPolygonID].a - deviceInput[nPolygonID].c)
	// );

	deviceOutput[threadID] = make_float4(
		pos.x, pos.y, pos.z, 
		calculateBrightness(pos, nPolygonID, deviceInput));
}

// :MARK: System functions

__host__ polygon3 vec3ToPolygon(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
	return polygon3(
		make_float3(a.r, a.g, a.b),
		make_float3(b.r, b.g, b.b),
		make_float3(c.r, c.g, c.b)
	);
}

__host__ void checkCUDAError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "CUDA error (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

// :MARK: Main executable fuction

__host__ int main(int argc, char *argv[])
{
	int voxelsPerSide = 32;
	float sdfDepth = 0.1f;
	bool enableRotation = false;
	std::string meshPath;

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-v") == 0)
        {
			int _tV = atoi(value.c_str());
			if ((_tV < 8) || !(_tV % 8 == 0))
			{
				std::cerr << "voxelPerSide should be equal or greater than [8] and [8] multiple." << std::endl;
				return EXIT_FAILURE; 
			}
#if !defined(SKIP_WARNINGS)
			if (_tV > 64)
			{
				char o;
				std::wcerr << "[This warning can be suppressed via -DSKIP_WARNINGS command]" << std::endl << "voxelsPerSide parameter is set too high. This may cause long execution (exponential cubic)." << std::endl;
				std::cout << "Are you sure want to continue? [y/n, n will set voxelsPerSide to 64] ";
				std::cin >> o;
				_tV = o == 'n' ? 64 : _tV;
			}
#endif
			voxelsPerSide = _tV;
        } else if (option.compare("-d") == 0)
        {
            sdfDepth = atof(value.c_str());
        } else if (option.compare("-r") == 0)
		{
			enableRotation = static_cast<bool>(atoi(value.c_str()));
		} else if(option.compare("-f") == 0)
        {
            meshPath = value;
    	}
	}
#if !defined(TEST_MODE)
	if (meshPath.empty())
	{
		std::cerr << "Error: No file paths specified. Use '-f' to specify path or enable test mode with flag '-t'" << std::endl;
		return EXIT_FAILURE;
	}
#endif

#ifdef DEBUG_MODE
	std::cout << "Display creation..." <<std::endl;
#endif
	Display display(DISPLAY_W, DISPLAY_H, "OpenGL SDF");

#ifdef TEST_MODE
	Vertex vertices[] =
	{
		Vertex(glm::vec3(-1, -1, -1), glm::vec2(1, 0), glm::vec3(0, 0, -1)),
		Vertex(glm::vec3(-1, 1, -1), glm::vec2(0, 0), glm::vec3(0, 0, -1)),
		Vertex(glm::vec3(1, 1, -1), glm::vec2(0, 1), glm::vec3(0, 0, -1)),
		Vertex(glm::vec3(1, -1, -1), glm::vec2(1, 1), glm::vec3(0, 0, -1)),

		Vertex(glm::vec3(-1, -1, 1), glm::vec2(1, 0), glm::vec3(0, 0, 1)),
		Vertex(glm::vec3(-1, 1, 1), glm::vec2(0, 0), glm::vec3(0, 0, 1)),
		Vertex(glm::vec3(1, 1, 1), glm::vec2(0, 1), glm::vec3(0, 0, 1)),
		Vertex(glm::vec3(1, -1, 1), glm::vec2(1, 1), glm::vec3(0, 0, 1)),

		Vertex(glm::vec3(-1, -1, -1), glm::vec2(0, 1), glm::vec3(0, -1, 0)),
		Vertex(glm::vec3(-1, -1, 1), glm::vec2(1, 1), glm::vec3(0, -1, 0)),
		Vertex(glm::vec3(1, -1, 1), glm::vec2(1, 0), glm::vec3(0, -1, 0)),
		Vertex(glm::vec3(1, -1, -1), glm::vec2(0, 0), glm::vec3(0, -1, 0)),

		Vertex(glm::vec3(-1, 1, -1), glm::vec2(0, 1), glm::vec3(0, 1, 0)),
		Vertex(glm::vec3(-1, 1, 1), glm::vec2(1, 1), glm::vec3(0, 1, 0)),
		Vertex(glm::vec3(1, 1, 1), glm::vec2(1, 0), glm::vec3(0, 1, 0)),
		Vertex(glm::vec3(1, 1, -1), glm::vec2(0, 0), glm::vec3(0, 1, 0)),

		Vertex(glm::vec3(-1, -1, -1), glm::vec2(1, 1), glm::vec3(-1, 0, 0)),
		Vertex(glm::vec3(-1, -1, 1), glm::vec2(1, 0), glm::vec3(-1, 0, 0)),
		Vertex(glm::vec3(-1, 1, 1), glm::vec2(0, 0), glm::vec3(-1, 0, 0)),
		Vertex(glm::vec3(-1, 1, -1), glm::vec2(0, 1), glm::vec3(-1, 0, 0)),

		Vertex(glm::vec3(1, -1, -1), glm::vec2(1, 1), glm::vec3(1, 0, 0)),
		Vertex(glm::vec3(1, -1, 1), glm::vec2(1, 0), glm::vec3(1, 0, 0)),
		Vertex(glm::vec3(1, 1, 1), glm::vec2(0, 0), glm::vec3(1, 0, 0)),
		Vertex(glm::vec3(1, 1, -1), glm::vec2(0, 1), glm::vec3(1, 0, 0)),
	};

	unsigned int indices[] = {0, 1, 2,
							  0, 2, 3,

							  6, 5, 4,
							  7, 6, 4,

							  10, 9, 8,
							  11, 10, 8,

							  12, 13, 14,
							  12, 14, 15,

							  16, 17, 18,
							  16, 18, 19,

							  22, 21, 20,
							  23, 22, 20
	                          };

	Mesh originalMesh(vertices, sizeof(vertices)/sizeof(vertices[0]), indices, sizeof(indices)/sizeof(indices[0]));
#else
	Mesh originalMesh(meshPath);
#endif

#if defined DEBUG_MODE || defined TEST_MODE
	std::cout << "Remap data for CUDA." <<std::endl;
#endif
	int numPolygons = originalMesh.getModel().indices.size() / 3;
	int numVoxels = pow(voxelsPerSide, 3);

	polygon3 *hostInput = (polygon3 *)malloc(sizeof(polygon3) * numPolygons);
	float4 *hostOutput = (float4 *)malloc(sizeof(float4) * numVoxels);

	for (int i=0; i<numPolygons; ++i)
	{
		hostInput[i] = vec3ToPolygon(
			originalMesh.getModel().positions[originalMesh.getModel().indices[3*i]],
			originalMesh.getModel().positions[originalMesh.getModel().indices[3*i+1]],
			originalMesh.getModel().positions[originalMesh.getModel().indices[3*i+2]]);
	}

	float3 minVertex = make_float3(0,0,0);
	float3 maxVertex = make_float3(0,0,0);
	for(glm::vec3 &v : originalMesh.getModel().positions) 
	{
		minVertex.x = minVertex.x < v.r ? minVertex.x : v.r;
		minVertex.y = minVertex.y < v.g ? minVertex.y : v.g;
		minVertex.z = minVertex.z < v.b ? minVertex.z : v.b;

		maxVertex.x = maxVertex.x > v.r ? maxVertex.x : v.r;
		maxVertex.y = maxVertex.y > v.g ? maxVertex.y : v.g;
		maxVertex.z = maxVertex.z > v.b ? maxVertex.z : v.b;
	}

#if defined DEBUG_MODE || defined TEST_MODE
	std::cout << "Allocation and copying to CUDA memory." <<std::endl;
#endif
	cudaError_t err;

	float eps = .0000001f;

#ifdef DEBUG_MODE
	std::cout << "numVoxels: " << numVoxels <<std::endl;
	std::cout << "eps: " << eps <<std::endl;
	std::cout << "numPolygons: " << numPolygons <<std::endl;
	std::cout << "sdfDepth: " << sdfDepth <<std::endl;
#endif

	err = cudaMemcpyToSymbol(EPSILON, &eps, sizeof(float), 0, cudaMemcpyHostToDevice);
	checkCUDAError(err);
	err = cudaMemcpyToSymbol(TEXTURE_SIZE, &voxelsPerSide, sizeof(int), 0, cudaMemcpyHostToDevice);
	checkCUDAError(err);
	err = cudaMemcpyToSymbol(NUM_POLYGONS, &numPolygons, sizeof(int), 0, cudaMemcpyHostToDevice);
	checkCUDAError(err);
	err = cudaMemcpyToSymbol(NUM_VOXELS, &numVoxels, sizeof(int), 0, cudaMemcpyHostToDevice);
	checkCUDAError(err);
	err = cudaMemcpyToSymbol(MIN_VERTEX, &minVertex, sizeof(float3), 0, cudaMemcpyHostToDevice);
	checkCUDAError(err);
	err = cudaMemcpyToSymbol(MAX_VERTEX, &maxVertex, sizeof(float3), 0, cudaMemcpyHostToDevice);
	checkCUDAError(err);

	polygon3 *deviceInput;
	float4 *deviceOutput;

	err = cudaMalloc(&deviceInput, sizeof(polygon3) * numPolygons);
	checkCUDAError(err);
	err = cudaMalloc(&deviceOutput, sizeof(float4) * numVoxels);
	checkCUDAError(err);

    err = cudaMemcpy(deviceInput, hostInput, sizeof(polygon3) * numPolygons, cudaMemcpyHostToDevice);
    checkCUDAError(err);

#if defined DEBUG_MODE || defined TEST_MODE
	std::cout << "CUDA calculation." <<std::endl;
#endif
	int gridDim = voxelsPerSide/8;
	dim3 grid(gridDim, gridDim, gridDim);
	dim3 block(8, 8, 8);
    SignedDistanceField<<<grid, block>>>(deviceInput, deviceOutput);

    err = cudaMemcpy(hostOutput, deviceOutput, sizeof(float4) * numVoxels, cudaMemcpyDeviceToHost);
    checkCUDAError(err);


#ifdef DEBUG_MODE 
	std::cout << "Voxel mesh construction..." <<std::endl;
#endif

	VoxelMesh newMesh(voxelsPerSide, numVoxels, hostOutput, sdfDepth);


#ifdef DEBUG_MODE
	std::cout << "Initializing shaders and texture..." <<std::endl;
#endif
	Shader shader("./data/shaders/bs");
	Texture texture("./data/textures/bricks.jpg");
	Transform transform;
	Camera camera(glm::vec3(0.0f, 0.0f, -4.0f), 70.0f, (float)DISPLAY_W/(float)DISPLAY_H, 0.1f, 100.0f);

	int counter = 0.0f;
	bool flag = false;

	transform.GetRot()->y = 3.7f;
	while(!display.getIsClosed())
	{
		display.Clear(0.0f, 0.0f, 0.0f, 1.0f);

		if (enableRotation)
		{		
			transform.GetRot()->y = 3.7f + counter * 0.0002f;
			transform.GetRot()->z = counter * 0.0002f;
		}

		shader.Bind();
		texture.Bind();
		shader.Update(transform, camera);
		if (flag)
			newMesh.Draw();
		else
			originalMesh.Draw();

		display.Update();
		SDL_Delay(1);
		counter += 5;
		if (counter % 2000 ==0)
		{
#ifdef DEBUG_MODE
			std::cout << "Mesh changed." << std::endl;
#endif
			flag = !flag;
		}
	}

    return EXIT_SUCCESS;
}