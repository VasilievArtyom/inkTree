#include <cmath>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <omp.h>
#include <string>

#define NUMOFPTCLS 8192 //should correspond to the block size on device
#define BLCSIZE 256 //block size on device
#define SIMTIME 1000.0f
#define MODELINGTIME 1000.0f
#define STEP 0.2f
#define MODEL_PARAM 0.00165344f // MODEL_PARAM = a_0/R * 0.75
#define MINDIST 0.000001f // minimal dist 


__global__ void onCernelCalc(float *X, float *Y, float *Z,
							 float *UX, float *UY, float *UZ,
							 int blcs_num, int ptclsNumm, float modelParam)
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
				uz_i += -1.0f / (r) - hz * (hz / (r*r*r));
			}
		}
		__syncthreads();
	}
	
	UX[id] = modelParam * ux_i;
	UY[id] = modelParam * uy_i;
	UZ[id] = modelParam * uz_i - 1.0f;
}

void viCalc(float *X, float *Y, float *Z, float *VX, float *VY, float *VZ,
			float *devX, float *devY, float *devZ, float *devVX, float *devVY, float *devVZ, 
			int ptclsNumm, float modelParam)
{
	unsigned int array_size = sizeof(float) * ptclsNumm;
	//copy state from host to device
	cudaMemcpy(devX, X, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devY, Y, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devZ, Z, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVX, VX, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVY, VY, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVZ, VZ, array_size, cudaMemcpyHostToDevice);

	int numOfThreads = BLCSIZE;
	int numOfBlocks = ptclsNumm / BLCSIZE;

	onCernelCalc <<<numOfBlocks, numOfThreads>>> (devX, devY, devZ, devVX, devVY, devVZ, numOfBlocks, ptclsNumm, modelParam);

	cudaMemcpy(VX, devVX, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(VY, devVY, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(VZ, devVZ, array_size, cudaMemcpyDeviceToHost);
}

void init(float *X, float *Y, float *Z, float *VX, float *VY, float *VZ, int num)
{
	for (int i = 0; i < num; ++i)
	{
		VX[i] = 0;
		VY[i] = 0;
		VZ[i] = 0;
	}

	float const R1 = 4.0f;
	float const R2 = 2.0f;

	int xyRadialNum = num / 128;
	int zRadialNum = num / 256;
	int rRadialNum = (num / xyRadialNum ) / zRadialNum;

	float fi, theta, r_param;
	int counter = 0;
	for (int xy = 0; xy < xyRadialNum; ++xy)
	{
		
		for (int z = 0; z < zRadialNum; ++z)
		{
			
			for (int r = 0; ((r < rRadialNum) && (counter < num)); ++r, ++counter)
			{
				fi = ((rand() * 1.0) / (RAND_MAX * 1.0)) * 2 * 3.1415926535;
				theta = ((rand() * 1.0) / (RAND_MAX * 1.0)) * 2 * 3.1415926535;				
				r_param = ((rand() * 1.0) / (RAND_MAX * 1.0)) * R2;

				X[counter] = R1 * cos(fi) + r_param * cos(theta) * cos(fi);
				Y[counter] = R1 * sin(fi) + r_param * cos(theta) * sin(fi);
				Z[counter] = r_param * sin(theta);
			}
		}
	}
}

void init_s(float *X, float *Y, float *Z, float *VX, float *VY, float *VZ, int num)
{
	for (int i = 0; i < num; ++i)
	{
		VX[i] = 0;
		VY[i] = 0;
		VZ[i] = 0;
	}

	float const R1 = 4.0f;

	int xyRadialNum = num / 128;
	int zRadialNum = num / 256;
	int rRadialNum = (num / xyRadialNum) / zRadialNum;

	float fi, theta, r_param;
	int counter = 0;
	for (int xy = 0; xy < xyRadialNum; ++xy)
	{

		for (int z = 0; z < zRadialNum; ++z)
		{

			for (int r = 0; ((r < rRadialNum) && (counter < num)); ++r, ++counter)
			{
				fi = ((rand() * 1.0) / (RAND_MAX * 1.0)) * 2 * 3.1415926535;
				theta = ((rand() * 1.0) / (RAND_MAX * 1.0)) * 2 * 3.1415926535;
				r_param = ((rand() * 1.0) / (RAND_MAX * 1.0)) * R1;

				X[counter] = r_param * cos(fi) * cos(theta);
				Y[counter] = r_param * sin(fi) * cos(theta);
				Z[counter] = r_param * sin(theta);
			}
		}
	}
}

void observe(float *X, float *Y, float *Z, int num)
{
	for (int i = 0; i < num; ++i)
	{
		std::cout << X[i] << '\t' << Y[i] << '\t' << Z[i] << std::endl;
	}
	std::cout << '#' << std::endl;
}

void RK4_step(float ** X, float ** Y, float ** Z, float * VX, float * VY, float * VZ,
	float *devX, float * devY, float * devZ, float * devVX, float * devVY, float * devVZ,
	float ** KX, float ** KY, float ** KZ, int ptclsNumm, float modelParam)
{
#pragma omp parallel for
	for (int i = 0; i < ptclsNumm; ++i) {
		KX[0][i] = 0.0f;
		KY[0][i] = 0.0f;
		KZ[0][i] = 0.0f;
		X[1][i] = X[0][i];
		Y[1][i] = Y[0][i];
		Z[1][i] = Z[0][i];
	}
	viCalc(X[1], Y[1], Z[1], KX[0], KY[0], KZ[0], devX, devY, devZ, devVX, devVY, devVZ, ptclsNumm, modelParam);
#pragma omp parallel for
	for (int i = 0; i < ptclsNumm; ++i) {
		KX[1][i] = 0.0f;
		KY[1][i] = 0.0f;
		KZ[1][i] = 0.0f;
		X[1][i] = X[0][i] + KX[0][i] * STEP * 0.5f;
		Y[1][i] = Y[0][i] + KY[0][i] * STEP * 0.5f;
		Z[1][i] = Z[0][i] + KZ[0][i] * STEP * 0.5f;
	}
	viCalc(X[1], Y[1], Z[1], KX[1], KY[1], KZ[1], devX, devY, devZ, devVX, devVY, devVZ, ptclsNumm, modelParam);
#pragma omp parallel for
	for (int i = 0; i < ptclsNumm; ++i) {
		KX[2][i] = 0.0f;
		KY[2][i] = 0.0f;
		KZ[2][i] = 0.0f;
		X[1][i] = X[0][i] + KX[1][i] * STEP * 0.5f;
		Y[1][i] = Y[0][i] + KY[1][i] * STEP * 0.5f;
		Z[1][i] = Z[0][i] + KZ[1][i] * STEP * 0.5f;
	}
	viCalc(X[1], Y[1], Z[1], KX[2], KY[2], KZ[2], devX, devY, devZ, devVX, devVY, devVZ, ptclsNumm, modelParam);
#pragma omp parallel for
	for (int i = 0; i < ptclsNumm; ++i) {
		KX[3][i] = 0.0f;
		KY[3][i] = 0.0f;
		KZ[3][i] = 0.0f;
		X[1][i] = X[0][i] + KX[2][i] * STEP;
		Y[1][i] = Y[0][i] + KY[2][i] * STEP;
		Z[1][i] = Z[0][i] + KZ[2][i] * STEP;
	}
		viCalc(X[1], Y[1], Z[1], KX[3], KY[3], KZ[3], devX, devY, devZ, devVX, devVY, devVZ, ptclsNumm, modelParam);
#pragma omp parallel for
	for (int i = 0; i < ptclsNumm; ++i)
	{
		X[0][i] += 1.0f / 6.0f*(KX[0][i] + 2 * KX[1][i] + 2 * KX[2][i] + KX[3][i]) * STEP;
		Y[0][i] += 1.0f / 6.0f*(KY[0][i] + 2 * KY[1][i] + 2 * KY[2][i] + KY[3][i]) * STEP;
		Z[0][i] += 1.0f / 6.0f*(KZ[0][i] + 2 * KZ[1][i] + 2 * KZ[2][i] + KZ[3][i]) * STEP;
	}
}

int main()
{
	float ** KX = new float*[4];
	float ** KY = new float*[4];
	float ** KZ = new float*[4];

	float ** X = new float*[2];
	float ** Y = new float*[2];
	float ** Z = new float*[2];

	for (int param = 32; param < 33; param *= 2)
	{
		
		int ptclsNum = 256 * param;
		float modelparam = MODEL_PARAM;

		//alloc arrays on host
		for (int gh = 0; gh < 4; ++gh)
		{
			KX[gh] = new float[ptclsNum];
			KY[gh] = new float[ptclsNum];
			KZ[gh] = new float[ptclsNum];
		}
		for (int gh = 0; gh < 2; ++gh)
		{
			X[gh] = new float[ptclsNum];
			Y[gh] = new float[ptclsNum];
			Z[gh] = new float[ptclsNum];
		}
		float * VX = new float[ptclsNum];
		float * VY = new float[ptclsNum];
		float * VZ = new float[ptclsNum];

		//alloc arrays on device
		float * devX, *devY, *devZ, *devVX, *devVY, *devVZ;
		unsigned int array_size = sizeof(float) * ptclsNum;
		cudaMalloc((void**)&devX, array_size); cudaMalloc((void**)&devY, array_size); cudaMalloc((void**)&devZ, array_size);
		cudaMalloc((void**)&devVX, array_size); cudaMalloc((void**)&devVY, array_size); cudaMalloc((void**)&devVZ, array_size);

		std::string path = std::to_string(param);
		
		std::freopen((path + "_out_torus.txt").c_str(), "w", stdout);

		//init conditions for host
		init(X[0], Y[0], Z[0], VX, VY, VZ, ptclsNum);

		for (double t = 0.0f; t < MODELINGTIME; t += STEP)
		{
			RK4_step(X, Y, Z, VX, VY, VZ, devX, devY, devZ, devVX, devVY, devVZ, KX, KY, KZ, ptclsNum, modelparam);
			if ((int(t * 1000) % 100) == 0) observe(X[0], Y[0], Z[0], ptclsNum);
			std::cerr << t << " of " << MODELINGTIME << std::endl;
		}

		

		std::freopen((path + "_out_sphe.txt").c_str(), "w", stdout);

		//init conditions for host
		init_s(X[0], Y[0], Z[0], VX, VY, VZ, ptclsNum);

		for (double t = 0.0f; t < MODELINGTIME; t += STEP)
		{
			RK4_step(X, Y, Z, VX, VY, VZ, devX, devY, devZ, devVX, devVY, devVZ, KX, KY, KZ, ptclsNum, modelparam);
			if ((int(t * 1000) % 100) == 0) observe(X[0], Y[0], Z[0], ptclsNum);
			std::cerr << t << " of " << MODELINGTIME << std::endl;
		}

		for (int gh = 0; gh < 4; ++gh)
		{
			delete [] KX[gh];
			delete [] KY[gh];
			delete [] KZ[gh];
		}
		for (int gh = 0; gh < 2; ++gh)
		{
			delete[] X[gh];
			delete[] Y[gh];
			delete[] Z[gh];
		}
		delete[] VX;
		delete[] VY;
		delete[] VZ;

		cudaFree(devX);
		cudaFree(devY);
		cudaFree(devZ);
		cudaFree(devVX);
		cudaFree(devVY);
		cudaFree(devVZ);
	}
	return 0;
}
