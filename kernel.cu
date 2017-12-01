#include <cmath>
#include <ostream>
#include "cuda_runtime.h"
#include "cuda.h"

#define NUMOFPTCLS 1000 //should correspond to the block size on device
#define BLCSIZE 256 //block size on device
#define SIMTIME 1000.0f
#define MODELINGTIME 1000.0f
#define STEP 0.01f
#define MODEL_PARAM 0.04f // MODEL_PARAM = a_0/R
#define MINDIST 0.0001f // minimal dist 


__global__ void onCernelCalc(float *X, float *Y, float *Z,
						  float * UX, float *UY, float *UZ,
						  float * U_ijX, float *U_ijY, float *U_ijZ)
{
	int id = threadId;
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
	
	float * devU_ijX, *devU_ijy, *devU_ijZ; //elements of summ for vectors Ui 
	cudaMalloc((void**)&devU_ijX, array_size); cudaMalloc((void**)&devU_ijy, array_size); cudaMalloc((void**)&devU_ijZ, array_size);

	int numOfThreads = BLCSIZE;
	int numOfBlocks = NUMOFPTCLS / BLCSIZE;

	onCernelCalc <<<numOfBlocks, numOfThreads >>> (devX, devY, devZ, devVX, devVY, devVZ, devU_ijX, devU_ijY, devU_ijZ);
	///!!!
	cudaMemcpy(devVX, VX, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(devVY, VY, array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(devVZ, VZ, array_size, cudaMemcpyDeviceToHost);
}

void init(float *X, float *Y, float *Z, float *VX, float *VY, float *VZ, int n)
{

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
	init(X[0], Y[0], Z[0], VX, VY, VZ, NUMOFPTCLS);

	//alloc arrays on device
	float * devX, * devY, * devZ, * devVX, * devVY, * devVZ;
	unsigned int array_size = sizeof(float) * NUMOFPTCLS;
	cudaMalloc((void**)&devX, array_size); cudaMalloc((void**)&devY, array_size); cudaMalloc((void**)&devZ, array_size);
	cudaMalloc((void**)&devVX, array_size); cudaMalloc((void**)&devVY, array_size); cudaMalloc((void**)&devVZ, array_size);
	

	
	
	//[::RK4
	for (double t = 0.0f; t < MODELINGTIME; t += STEP)
	{
		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[0][i] = 0.0f;
			KY[0][i] = 0.0f;
			KZ[0][i] = 0.0f;
			X[i][1] = X[i][0];
			Y[i][1] = Y[i][0];
			Z[i][1] = Z[i][0];
		}
		viCalc(X[1], Y[1], Z[1], KX[0], KY[0], KZ[0], devX, devY, devZ, devVX, devVY, devVZ);
		
		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[i][1] = 0.0f;
			KY[i][1] = 0.0f;
			KZ[i][1] = 0.0f;
			X[i][1] = X[i][0] + KX[i][0] * STEP * 0.5f;
			Y[i][1] = Y[i][0] + KY[i][0] * STEP * 0.5f;
			Z[i][1] = Z[i][0] + KZ[i][0] * STEP * 0.5f;
		}
		viCalc(X[1], Y[1], Z[1], KX[1], KY[1], KZ[1], devX, devY, devZ, devVX, devVY, devVZ);

		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[i][2] = 0.0f;
			KY[i][2] = 0.0f;
			KZ[i][2] = 0.0f;
			X[i][1] = X[i][0] + KX[i][1] * STEP * 0.5f;
			Y[i][1] = Y[i][0] + KY[i][1] * STEP * 0.5f;
			Z[i][1] = Z[i][0] + KZ[i][1] * STEP * 0.5f;
		}
		viCalc(X[1], Y[1], Z[1], KX[2], KY[2], KZ[2], devX, devY, devZ, devVX, devVY, devVZ);

		for (int i = 0; i < NUMOFPTCLS; ++i) {
			KX[i][2] = 0.0f;
			KY[i][2] = 0.0f;
			KZ[i][2] = 0.0f;
			X[i][1] = X[i][0] + KX[i][2] * STEP;
			Y[i][1] = Y[i][0] + KY[i][2] * STEP;
			Z[i][1] = Z[i][0] + KZ[i][2] * STEP;
		}
		viCalc(X[1], Y[1], Z[1], KX[3], KY[3], KZ[3], devX, devY, devZ, devVX, devVY, devVZ);

		for (int i = 0; i < NUMOFPTCLS; ++i)
		{
			X[i][0] += 1.0f/6.0f*(KX[i][0] + 2 * KX[i][1] + 2 * KX[i][2] + KX[i][3]) * STEP;
			Y[i][0] += 1.0f/6.0f*(KY[i][0] + 2 * KY[i][1] + 2 * KY[i][2] + KY[i][3]) * STEP;
			Z[i][0] += 1.0f/6.0f*(KZ[i][0] + 2 * KZ[i][1] + 2 * KZ[i][2] + KZ[i][3]) * STEP;
		}
	}
	//::]RK4

	return 0;
}