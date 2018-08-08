#ifndef MAXSAT_UTILITIES_H
#define MAXSAT_UTILITIES_H
#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <time.h>
#include <random>
#include <iostream>
#include <math.h>
#include <vector>

#include "MaxSatStructures.h"

__global__ void sumOrCountKernel(int* results, int* sum, int n, int log, bool count_positives)
{
	extern __shared__ int shared[];
	int* share_results = &shared[0];

	int ti = threadIdx.x;
	int bd = blockDim.x;
	int maxti = 1 << log;
	int offset = 1 << (log - 1);
	int subIndex = 0;
	int buf = 0;

	for (int i = 0; i < (n / bd) + 1; i++) {
		if (i*bd + threadIdx.x < n) {
			share_results[i*bd + threadIdx.x] = count_positives ? (int)(results[i*bd + threadIdx.x] > 0) :  results[i*bd + threadIdx.x];
		}
	}
	__syncthreads();
	for (int i = log; i > 0; i--) {
		for (int j = 0; j*bd <= offset; j++)
		{
			subIndex = offset + j*bd + ti;
			buf = 0;
			if (subIndex < maxti) {
				if (2 * subIndex < n)
					buf += share_results[2 * subIndex];
				if (2 * subIndex + 1 < n)
					buf += share_results[2 * subIndex + 1];
				share_results[subIndex] += buf;
			}
		}
		offset = offset >> 1;
		maxti = maxti >> 1;
		__syncthreads();
	}
	__syncthreads();
	if (ti == 0) {
		share_results[0] = share_results[0] + share_results[1];
		*sum = share_results[0];
	}
}

__global__ void maxKernel(int* results, int* maxindex, int n, int log)
{
	extern __shared__ int shared[];
	int* share_results = &shared[0];
	int* share_indexes = &shared[n];

	int ti = threadIdx.x;
	int bd = blockDim.x;
	int maxti = 1 << log;
	int locMax = 0;
	int locMaxIndex = 0;
	int offset = 1 << (log - 1);
	int subIndex = 0;

	for (int i = 0; i < (n / bd) + 1; i++) {
		if (i*bd + threadIdx.x < n) {
			share_results[i*bd + threadIdx.x] = results[i*bd + threadIdx.x];
			share_indexes[i*bd + threadIdx.x] = i*bd + threadIdx.x;
		}
	}
	__syncthreads();
	for (int i = log; i > 0; i--) {
		for (int j = 0; j*bd <= offset; j++)
		{
			locMax = 0;
			subIndex = offset + j*bd + ti;
			if (subIndex < maxti) {
				if (subIndex * 2 < n) {
					locMax = share_results[subIndex * 2];
					locMaxIndex = share_indexes[subIndex * 2];
				}
				if (subIndex * 2 + 1 < n && share_results[subIndex * 2 + 1] > locMax) {
					locMax = share_results[subIndex * 2 + 1];
					locMaxIndex = share_indexes[subIndex * 2 + 1];
				}
				if (share_results[subIndex] < locMax) {
					share_results[subIndex] = locMax;
					share_indexes[subIndex] = locMaxIndex;
				}
			}
		}
		offset = offset >> 1;
		maxti = maxti >> 1;
		__syncthreads();
	}
	__syncthreads();
	if (ti == 0) {
		if (share_results[1] > share_results[0]) {
			share_results[0] = share_results[1];
			share_indexes[0] = share_indexes[1];
		}
		results[0] = share_results[0];
		*maxindex = share_indexes[0];
	}
}
__global__ void oneStepKernel(SatState* state, int *results, int parent) {
	int prev = 0;
	int res = 0;
	int n = state->cnf->cu_vars_size[threadIdx.x];
	for (int i = 0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[threadIdx.x][i]].bToggleEval(state->cu_assignment, -1, parent)) prev++;
	}
	for (int i = 0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[threadIdx.x][i]].bToggleEval(state->cu_assignment, threadIdx.x, parent)) res++;
	}
	results[threadIdx.x] = res - prev;
}

__global__ void twoStepKernel(SatState* state, int **results, int* results_index) {
	int prev = 0;
	int parent = 0;
	int res = 0;
	int size = state->size;
	int sizelog = state->sizelog;
	int index = blockIdx.x;
	int n = state->cnf->cu_vars_size[index];
	for (int i = 0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[index][i]].bToggleEval(state->cu_assignment, -1, -1)) prev++;
	}
	for (int i = 0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[index][i]].bToggleEval(state->cu_assignment, index, -1)) parent++;
	}

	//oneStepKernel << < 1, size >> >(state, results[index], index);
	//cudaDeviceSynchronize();
	//if (index != 0) {
	//	maxKernel << <1, 500, (2 * size * sizeof(int)) >> > (results[index], &results_index[index], size, sizelog);
	//	auto err = cudaGetLastError();
	//	cudaDeviceSynchronize();
	//	err = cudaGetLastError();
	//	res = results[index][0];
	//}
	results[0][index] = parent + res - prev;
}


__global__ void randIntitKernel(curandState* state, int size) {
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	if (ind < size) {
		curand_init(112, ind, 0, &state[ind]);
		printf("%d\n", ind);
	}
}

__global__ void SAkernel(SatState *state, curandState* rand, int* result, bool** assignment) {
	//	extern __shared__ bool assignment[];
	int ind = blockDim.x*blockIdx.x + threadIdx.x;
	int size = state->size;
	for (int i = 0; i < size; i++)
		assignment[ind][i] = state->cu_assignment[i];
	int current = state->score;
	int best = current;
	int eval = 0;
	int maxeval = 0;
	int toggle_index = 0;
	int temperature = 5000;
	int counter = 0;
	while (temperature > 100) {
		temperature -= 1;
		counter = 0;
		while (1) {
			counter++;
			toggle_index = (curand(&rand[ind]) % (size - 1)) + 1;
			int prev = 0;
			int res = 0;
			int n = state->cnf->cu_vars_size[toggle_index];
			for (int i = 0; i<n; i++)
			{
				prev += (state->cnf->cu_clauses[state->cnf->cu_vars_cind[toggle_index][i]].iToggleEval(assignment[ind], -1, -1));
			}
			for (int i = 0; i<n; i++)
			{
				res += (state->cnf->cu_clauses[state->cnf->cu_vars_cind[toggle_index][i]].iToggleEval(assignment[ind], toggle_index, -1));
			}
			eval = res - prev;
			if (eval >= 0 || curand(&rand[ind]) % 20000 < temperature || counter > size / 2) {
				assignment[ind][toggle_index] = !assignment[ind][toggle_index];
				current += eval;
				break;
			}
		}
		if (best < current) {
			//	printf("upgrade! : process %d - score : %d \n", ind,current );
			best = current;
		}
	}
	result[ind] = best;
}

#endif