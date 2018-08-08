#ifndef MAXSAT_TESTS_H
#define MAXSAT_TESTS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "sortingNetworksNvidia\helper_cuda.h"
#include <chrono>
//#include "sortingNetworksNvidia\helper_timer.h"

#include <stdio.h>
#include <time.h>
#include <random>
#include <iostream>
#include <math.h>
#include <vector>

#include "MaxSatUtilities.h"
#include "sortingNetworksNvidia\sortingNetworks_common.h"
#include "sortingNetworksNvidia\bitonicSort.h"

using namespace std;

class Test {
	int a;
public:
	Test(int in) {
		a = in;
	}

	void init() {
		a = rand() % 20;
	}

	__device__ bool testFunc() {
		return (a < 5 || a > 95);
	}
};


__global__ void classTest(Test *t, int size) {
	int ti = threadIdx.x;
	if (ti<size)
		printf("%u \n", t[ti].testFunc());
}


void maxFanTest() {
	srand(time(NULL));
	const int size = 1000;
	int* a = NULL;
	int max = 0;
	int slog = (int)(log2(size));
	auto err = cudaMallocManaged(&a, (size_t)(size * sizeof(int)));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}

	// Add vectors in parallel.
	/*for (int i = 0; i < size; i++)
	if (a[i] > max)
	max = a[i];
	*/
	cout << "start";

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			a[j] = 0;
		a[i] = 10;
		//	maxKernel << <1, 200,(2*size*sizeof(int)) >> >(a, size, slog);
		auto err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
			//exit(EXIT_FAILURE);
		}
		cudaDeviceSynchronize();
		if (a[0] != 10)
			cout << "fail at " << i << endl;
		a[i] = 0;
		cout << i << endl;
	}


	cudaDeviceSynchronize();
	cudaFree(a);
	cout << "end";
	/*if (a[0] == max)
	cout << "Pass!";*/
	cudaDeviceReset();
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed with cuda!");
	}

}

void sumOrCountFanTest() {
	srand(time(NULL));
	const int size = 100;
	int* a = NULL;
	int* sum;
	int slog = (int)(log2(size));
	auto err = cudaMallocManaged(&a, (size_t)(size * sizeof(int)));
	err = cudaMallocManaged(&sum, (size_t)(sizeof(int)));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}

	cout << "start";
	for (int i = 0; i < size; i++) {
		a[i] = i%2 ? i : -i;
	}
	sumOrCountKernel << <1, 200, (size * sizeof(int)) >> >(a, sum, size, slog,true);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
	cudaDeviceSynchronize();
	//if (*sum != size*(size - 1) / 2)
	//	cout << "failed" << endl;
	if (*sum != size / 2)
		cout << "failed" << endl;

	cudaDeviceSynchronize();
	cudaFree(a);
	cout << " end";
	cudaDeviceReset();
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed with cuda!");
	}

}

void classInCudaTest() {
	const int size = 100;
	Test * t;
	cudaMallocManaged(&t, sizeof(Test)*size);
	for (int i = 0; i < size; i++) {
		//t[i].init();
		new(&t[i]) Test(i);
	}

	classTest << <1, 1000 >> >(t, size);
	cudaDeviceSynchronize();
	cudaFree(t);
}

void sortTest() {
	const int size = 1024;
	int* data;
	int* indexes;
	cudaMallocManaged(&data,size*sizeof(int));
	cudaMallocManaged(&indexes, size * sizeof(int));

	int* sdata;
	int* sindexes;
	cudaMallocManaged(&sdata, size * sizeof(int));
	cudaMallocManaged(&sindexes, size * sizeof(int));

	for (int i = 0; i < size; i++) {
		data[i] = i % 100;
		indexes[i] = i;
	}



	uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
	uint *d_InputKey, *d_InputVal, *d_OutputKey, *d_OutputVal;


	const uint             N = 1048576;
	const uint           DIR = 0;
	const uint     numValues = 65536;
	const uint numIterations = 1;



	int flag = 1;
	printf("Running GPU bitonic sort (%u identical iterations)...\n\n", numIterations);

	
	auto start = chrono::steady_clock::now();

	//  Insert the code that will be timed
		
		uint threadCount = 0;

			threadCount = bitonicSort(
				sdata,
				sindexes,
				data,
				indexes,
				1,
				size,
				DIR
			);

		auto error = cudaDeviceSynchronize();
		checkCudaErrors(error);
	
		auto end = chrono::steady_clock::now();

		// Store the time difference between start and end
		auto diff = end - start;
		cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;

	printf("Shutting down...\n");
	cudaFree(data);
	cudaFree(sdata);
	cudaFree(indexes);
	cudaFree(sindexes);
}

#endif