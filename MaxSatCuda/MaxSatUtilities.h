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
#include <chrono>
#include <iomanip>
#include <fstream>
#include <string>

#include "MaxSatStructures.h"
#define INF 100000

inline int randomRangeUniform(int max) {
	return (int)(((double)rand() / RAND_MAX)*(max-1));
}
				
__global__ void sumOrCountPositivesKernel(int* results, int* sum, int n, int log, bool count_positives)
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
			subIndex = offset + j*bd + ti;
			if (subIndex < maxti) {
				if (subIndex * 2 < n && share_results[subIndex * 2] > share_results[subIndex]) {
					share_results[subIndex] = share_results[subIndex * 2];
					share_indexes[subIndex] = share_indexes[subIndex * 2];
				}
				if (subIndex * 2 + 1 < n && share_results[subIndex * 2 + 1] > share_results[subIndex]) {
					share_results[subIndex] = share_results[subIndex * 2 + 1];
					share_indexes[subIndex] = share_indexes[subIndex * 2 + 1];
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

__global__ void tabuMaxKernel(int* results, int* maxindex,int* history,int step_count,int tabu_tenur, int exceptional, int n, int log)
{
	extern __shared__ int shared[];
	int* share_results = &shared[0];
	int* share_indexes = &shared[n];

	int ti = threadIdx.x;
	int bd = blockDim.x;
	int maxti = 1 << log;
	int offset = 1 << (log - 1);
	int subIndex = 0;
	int tempind = 0;
	for (int i = 0; i < (n / bd) + 1; i++) {
		 tempind = i*bd + threadIdx.x;
		if (tempind < n) {
			share_results[tempind] = results[tempind];
			if (history[tempind] + tabu_tenur >= step_count && results[tempind] < exceptional )
				share_results[tempind] = -INF;
			share_indexes[tempind] = tempind;
		}
	}
	__syncthreads();
	for (int i = log; i > 0; i--) {
		for (int j = 0; j*bd <= offset; j++)
		{
			subIndex = offset + j*bd + ti;
			if (subIndex < maxti) {
				if (subIndex * 2 < n && share_results[subIndex * 2 ] > share_results[subIndex]) {
					share_results[subIndex] = share_results[subIndex * 2];
					share_indexes[subIndex] = share_indexes[subIndex * 2];
				}
				if (subIndex * 2 + 1 < n && share_results[subIndex * 2 + 1] > share_results[subIndex]) {
					share_results[subIndex] = share_results[subIndex * 2 + 1];
					share_indexes[subIndex] = share_indexes[subIndex * 2 + 1];
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

__global__ void randomPositivePickKernel(int* results, int* picked_index, int n, int log, curandState* rand)
{
	extern __shared__ int shared[];
	int* share_results = &shared[0];
	int* share_indexes = &shared[n];

	int ti = threadIdx.x;
	int bd = blockDim.x;
	int maxti = 1 << log;
	int offset = 1 << (log - 1);
	int subIndex = 0;
	int tempind = 0;
	int p;
	for (int i = 0; i < (n / bd) + 1; i++) {
		tempind = i*bd + threadIdx.x;
		if (tempind < n) {
			share_results[tempind] = results[tempind];
			share_indexes[tempind] = tempind;
		}
	}
	__syncthreads();
	for (int i = log; i > 0; i--) {
		for (int j = 0; j*bd <= offset; j++)
		{	
			subIndex = offset + j*bd + ti;
			if (subIndex < maxti) {
				p = curand(&rand[ti]) % 2;
				if (subIndex * 2 < n && (share_results[subIndex] <= 0 || (share_results[subIndex * 2] > 0 && p==1) )) {
					share_results[subIndex] = share_results[subIndex * 2];
					share_indexes[subIndex] = share_indexes[subIndex * 2];
				}
				if (subIndex * 2 + 1 < n && (share_results[subIndex] <= 0 || (share_results[subIndex * 2 + 1] > 0 && p==1) )) {
					share_results[subIndex] = share_results[subIndex * 2 + 1];
					share_indexes[subIndex] = share_indexes[subIndex * 2 + 1];
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
		*picked_index = share_indexes[0];
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

	if (parent == threadIdx.x)
		results[threadIdx.x] = -INF;
	else
		results[threadIdx.x] = res - prev;
}



__global__ void twoStepTabuKernel(SatState* state, int **results, int* results_index, int* history, int step_count, int tabu_tenur, int exceptional)
{
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

	oneStepKernel << < 1, size >> >(state, results[index], index);
	cudaDeviceSynchronize();
	if (index != 0) {
		tabuMaxKernel << <1, 32, (2 * size * sizeof(int)) >> > (results[index], &results_index[index], history, step_count, tabu_tenur, exceptional,size, sizelog);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		res = results[index][0];
	}
	results[0][index] = parent + res - prev;
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

	oneStepKernel << < 1, size >> >(state, results[index], index);
	cudaDeviceSynchronize();
	if (index != 0) {
		maxKernel << <1, 128, (2 * size * sizeof(int)) >> > (results[index], &results_index[index], size, sizelog);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		res = results[index][0];
	}
	results[0][index] = parent + res - prev;
}

__global__ void randIntitKernel(curandState* state, int size,int seed) {
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	if (ind < size) {
		curand_init(seed, ind, 0, &state[ind]);
		//printf("%d\n", ind);
	}
}

__global__ void SAkernel(SatState *state, curandState* rand, int* result, bool** assignment,int max_temperature) {
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
	int temperature = max_temperature;
	//int counter = 0;
	while (temperature > 2) {
		temperature--;
		//counter = 0;
		while (1) {
			//counter++;
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
			if (eval >= 0 || curand(&rand[ind]) % max_temperature < temperature) {
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

__global__ void Tabukernel(SatState **states, int** results,int* results_index,int** histories,int tabu_tenur,int size,int sizelog, int max_step) {
	int ind = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("%d\n", ind);
	int best = 0;
	int maxeval = 0;
	int maxind = 0;
	int step_count = 0;
	while (step_count<max_step) {
		step_count++;

		oneStepKernel <<< 1, size >>>(states[ind], results[ind], -1);
		cudaDeviceSynchronize();
		tabuMaxKernel << <1, 32, (2 * size * sizeof(int)) >> >(results[ind], &results_index[ind], histories[ind], step_count, tabu_tenur, (best - states[ind]->score + 1), size, sizelog);
		cudaDeviceSynchronize();
		maxeval = results[ind][0];
		maxind = results_index[ind];

		states[ind]->cu_assignment[maxind] = !states[ind]->cu_assignment[maxind];
		histories[ind][maxind] = step_count;
		states[ind]->score += maxeval;
		if (best < states[ind]->score) {
			//cout << "upgrade! :" << current << endl;
			best = states[ind]->score;
		}
		
	}
	results[ind][0] = best;
}

__global__ void testKernel(int* test) {
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	int count = 0;
	for (int i = 0; i < 1000; i++)
		for (int j = 1; j < 10000; j++)
		{
			count += test[1];
		}
		printf("%d\n", ind);
}

class Recorder {
private:
	std::chrono::time_point<chrono::steady_clock> start_time;
	std::chrono::time_point<chrono::steady_clock> last_upgrade_time;
	ofstream upgrade_records;
	ofstream reset_records;
	ofstream collective_result_records;
	string solver;
	string benchmark;
	string run;
	int timeout; //seconds
public: int optimal;
public: Recorder(string solver, string benchmark, string collective_file, string run, int timeout, int optimal = -1) {
	upgrade_records.open(solver + "_" + benchmark + "_upgrades.txt", ofstream::app);
	reset_records.open(solver + "_" + benchmark + "_iterations.txt", ofstream::app);
	//upgrade_records << "run,score,time,iteration,step"<<endl;
	//reset_records << "run,score,time,step"<<endl;
	collective_result_records.open(collective_file+".txt", ofstream::app);
	this->solver = solver;
	this->benchmark = benchmark;
	this->run = run;
	this->timeout = timeout;
	this->optimal = optimal;
	last_upgrade_time = chrono::steady_clock::now();
	start_time = chrono::steady_clock::now();
}
		void start() {
			start_time = chrono::steady_clock::now();
		}

		void upgradeRec(int score, int step = 1, int iteration = 1) {
			last_upgrade_time = chrono::steady_clock::now();
			upgrade_records << run << "," << score << "," << chrono::duration <double, milli>(last_upgrade_time - start_time).count() << "," << iteration << "," << step << endl;
		}

		void resetRec(int score, int step) {
			reset_records << run << "," << score << "," << chrono::duration <double, milli>(chrono::steady_clock::now() - start_time).count() << "," << step << endl;
		}

		void finalRec(int score,int step, int iteration = 1) {
			collective_result_records << solver << "," << benchmark << "," << run << "," << score << "," << chrono::duration <double, milli>(last_upgrade_time - start_time).count() << "," <<step<<","<<iteration << endl;
		}

		void finalRecforMultistep(int score, int step,  int iteration = 1) {
			collective_result_records << solver << "," << benchmark << "," << run << "," << score << "," << chrono::duration <double, milli>(chrono::steady_clock::now() - start_time).count() << "," << step << "," << iteration << endl;
		}

		~Recorder() {
			upgrade_records.close();
			reset_records.close();
			collective_result_records.close();
		}

		bool timeOut()
		{
			return chrono::duration <double, milli>(chrono::steady_clock::now() - start_time).count() > timeout * 1000;
		}

};


#endif