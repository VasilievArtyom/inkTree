#include <cmath>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <omp.h>

#define NUMOFPTCLS 16384 //should correspond to the block size on device
#define BLCSIZE 256 //block size on device
#define SIMTIME 1000.0f
#define MODELINGTIME 1000.0f
#define STEP 0.01f
#define MODEL_PARAM 0.04f // MODEL_PARAM = a_0/R * 0.75
#define MINDIST 0.0001f // minimal dist 


__global__ void onCernelCalc(float *X, float *Y, float *Z,
							 float *UX, float *UY, float *UZ,
							 int blcs_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float ux_i = 0.0f; float uy_i = 0.0f; float uz_i = 0.0f;
	float Xi = X[id]; float Yi = Y[id]; float Zi = Z[id]; //current element coord-s
	float hx, hy, hz; // (r_i - r_j)
	float r; // |r_i - r_j|
	const int SMBS = 256; //shared memory block size
	__shared__ float Xs[SMBS]; __shared__ float Ys[SMBS]; __shared__ float Zs[SMBS]; //copy of state in shared mem
	for (int block = 0; block < blcs_num; ++block)
	{
		Xs[threadIdx.x] = X[threadIdx.x + block * blockDim.x]; //parallel copy
		Ys[threadIdx.x] = Y[threadIdx.x + block * blockDim.x];
		Zs[threadIdx.x] = Z[threadIdx.x + block * blockDim.x];
		__syncthreads();

		for (int j = 0; j < blockDim.x; ++j)
		{
			if ((j + block * blockDim.x) != id)
			{
				hx = Xi - Xs[j];
				hy = Yi - Ys[j];
				hz = Zi - Zs[j];
				r = sqrtf(hx*hx + hy*hy + hz*hz) + MINDIST;
				ux_i += -hx * (hz / (r*r*r));
				uy_i += -hy * (hz / (r*r*r));
				uz_i += -1.0f / (r*r) - hz * (hz / (r*r*r));
			}
		}
		__syncthreads();
	}

	UX[id] = MODEL_PARAM * ux_i;
	UY[id] = MODEL_PARAM * uy_i;
	UZ[id] = MODEL_PARAM * uy_i - 1.0f;
}

void viCalc(float *X, float *Y, float *Z, float *VX, float *VY, float *VZ,
			float *devX, float *devY, float *devZ, float *devVX, float *devVY, float *devVZ)
{
	unsigned int array_size = sizeof(float) * NUMOFPTCLS;
	//copy state from host to device
	cudaMemcpy(devX, X, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devY, Y, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devZ, Z, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVX, VX, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVY, VY, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVZ, VZ, array_size, cudaMemcpyHostToDevice);

	int numOfThreads = BLCSIZE;
	int numOfBlocks = NUMOFPTCLS / BLCSIZE;

	onCernelCalc <<<numOfBlocks, numOfThreads>>> (devX, devY, devZ, devVX, devVY, devVZ, numOfBlocks);

	cudaMemcpy(devVX, VX, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(devVY, VY, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(devVZ, VZ, array_size, cudaMemcpyDeviceToHost);
}

void init(float *X, float *Y, float *Z, float *VX, float *VY, float *VZ)
{
	for (int i = 0; i < NUMOFPTCLS; ++i)
	{
		VX[i] = 0;
		VY[i] = 0;
		VZ[i] = 0;
	}

	float const R1 = 10.0f;
	float const R2 = 2.0f;

	int xyRadialNum = NUMOFPTCLS / 128;
	int zRadialNum = NUMOFPTCLS / 256;
	int rRadialNum = NUMOFPTCLS - xyRadialNum - zRadialNum;

	float fi, theta, r;

	for (int xy = 0; xy < xyRadialNum; ++xy)
	{
		fi = ((rand() * 1.0) / (RAND_MAX * 1.0)) * 2 * 3.1415926535;
		for (int z = 0; z < zRadialNum; ++z)
		{
			theta = ((rand() * 1.0) / (RAND_MAX * 1.0)) * 2 * 3.1415926535;
			for (int r = 0; r < rRadialNum; ++r)
			{
				r = ((rand() * 1.0) / (RAND_MAX * 1.0)) * R2;
				X[xy + z + r] = R1 * cos(fi) + r * cos(theta) * cos(fi);
				Y[xy + z + r] = R1 * sin(fi) + r * cos(theta) * sin(fi);
				Z[xy + z + r] = r * sin(theta);
			}
		}
	}
}


void observe(float *X, float *Y, float *Z)
{
	for (int i = 0; i < NUMOFPTCLS; ++i)
	{
		std::cout << X[i] << '\t' << Y[i] << '\t' << Z[i] << std::endl;
	}
	std::cout << '#' << std::endl;
}


int main()
{
	float KX[4][NUMOFPTCLS]; //x from K[index of ode][index of koeff for this ode] Set of RK4 coeff-ts 
	float KY[4][NUMOFPTCLS]; //y from K[index of ode][index of koeff for this ode] Set of RK4 coeff-ts 
	float KZ[4][NUMOFPTCLS]; //z from K[index of ode][index of koeff for this ode] Set of RK4 coeff-ts 

							 //alloc arrays on host 
	float X[2][NUMOFPTCLS], Y[2][NUMOFPTCLS], Z[2][NUMOFPTCLS];
	float VX[NUMOFPTCLS], VY[NUMOFPTCLS], VZ[NUMOFPTCLS];

	//init conditions for host
	init(X[0], Y[0], Z[0], VX, VY, VZ);

	//alloc arrays on device
	float * devX, *devY, *devZ, *devVX, *devVY, *devVZ;
	unsigned int array_size = sizeof(float) * NUMOFPTCLS;
	cudaMalloc((void**)&devX, array_size); cudaMalloc((void**)&devY, array_size); cudaMalloc((void**)&devZ, array_size);
	cudaMalloc((void**)&devVX, array_size); cudaMalloc((void**)&devVY, array_size); cudaMalloc((void**)&devVZ, array_size);


	//[::RK4
	for (double t = 0.0f; t < MODELINGTIME; t += STEP)
	{
#pragma omp parallel for
		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[0][i] = 0.0f;
			KY[0][i] = 0.0f;
			KZ[0][i] = 0.0f;
			X[1][i] = X[0][i];
			Y[1][i] = Y[0][i];
			Z[1][i] = Z[0][i];
		}
		viCalc(X[1], Y[1], Z[1], KX[0], KY[0], KZ[0], devX, devY, devZ, devVX, devVY, devVZ);
#pragma omp parallel for
		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[1][i] = 0.0f;
			KY[1][i] = 0.0f;
			KZ[1][i] = 0.0f;
			X[1][i] = X[0][i] + KX[0][i] * STEP * 0.5f;
			Y[1][i] = Y[0][i] + KY[0][i] * STEP * 0.5f;
			Z[1][i] = Z[0][i] + KZ[0][i] * STEP * 0.5f;
		}
		viCalc(X[1], Y[1], Z[1], KX[1], KY[1], KZ[1], devX, devY, devZ, devVX, devVY, devVZ);
#pragma omp parallel for
		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[2][i] = 0.0f;
			KY[2][i] = 0.0f;
			KZ[2][i] = 0.0f;
			X[1][i] = X[0][i] + KX[1][i] * STEP * 0.5f;
			Y[1][i] = Y[0][i] + KY[1][i] * STEP * 0.5f;
			Z[1][i] = Z[0][i] + KZ[1][i] * STEP * 0.5f;
		}
		viCalc(X[1], Y[1], Z[1], KX[2], KY[2], KZ[2], devX, devY, devZ, devVX, devVY, devVZ);
#pragma omp parallel for
		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[3][i] = 0.0f;
			KY[3][i] = 0.0f;
			KZ[3][i] = 0.0f;
			X[1][i] = X[0][i] + KX[2][i] * STEP;
			Y[1][i] = Y[0][i] + KY[2][i] * STEP;
			Z[1][i] = Z[0][i] + KZ[2][i] * STEP;
		}
		viCalc(X[1], Y[1], Z[1], KX[3], KY[3], KZ[3], devX, devY, devZ, devVX, devVY, devVZ);
#pragma omp parallel for
		for (int i = 0; i < NUMOFPTCLS; ++i)
		{
			X[0][i] += 1.0f / 6.0f*(KX[0][i] + 2 * KX[1][i] + 2 * KX[2][i] + KX[3][i]) * STEP;
			Y[0][i] += 1.0f / 6.0f*(KY[0][i] + 2 * KY[1][i] + 2 * KY[2][i] + KY[3][i]) * STEP;
			Z[0][i] += 1.0f / 6.0f*(KZ[0][i] + 2 * KZ[1][i] + 2 * KZ[2][i] + KZ[3][i]) * STEP;
		}

		if ((int(t * 1000) % 100) == 0) observe(X[0], Y[0], Z[0]);
	}
	//::]RK4
	return 0;
}
