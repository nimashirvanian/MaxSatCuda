

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <time.h>
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>
#include <fstream>

#include "MaxSatStructures.h"
#include "MaxSatSolvers.h"
#include "MaxSatTests.h"
#include "sortingNetworksNvidia\sortingNetworks_common.h"

using namespace std;

typedef vector< pair<int, bool> > vib;
typedef vector<bool> vb;
typedef vector<int> vi;

#define L(a) (int)((a).size())
#define all(a) (a).begin(), (a).end()
#define mp make_pair

#define Trace(X) cerr << #X << " = " << X << endl
#define _ << " _ " << 


int main()
{
	//sumOrCountFanTest();
	//classInCudaTest();
	//sortTest();
	srand(time(NULL));
	cudaDeviceReset();
	int nbvars, nbclauses;
	ifstream inFile;
	inFile.open("in.txt");
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}
	inFile >> nbvars >> nbclauses;
	nbvars++;
	Cnf* cnf;
	cudaMallocManaged(&cnf, sizeof(Cnf));
	new(cnf) Cnf(nbvars);
	while (nbclauses--) {
		int tempvar;
		Clause *tmpcls = new Clause();
		inFile >> tempvar;
		while (tempvar) {
			int id = abs(tempvar);
			bool sgn = tempvar > 0;
			tmpcls->addLiteral(id, sgn);
			inFile >> tempvar;
		}
		cnf->addClause(*tmpcls);
	}

	cnf->cudable();
	SatSolver *solver = new CudaMultiStepTabuSatSolver(nbvars, cnf);
	//SatSolver *solver = new GreedySatSolver(nbvars, cnf);
	
	auto start = chrono::steady_clock::now();
	//  Insert the code that will be timed

	solver->solve();

	auto end = chrono::steady_clock::now();
	// Store the time difference between start and end
	auto diff = end - start;
	cout << endl<<chrono::duration <double, milli>(diff).count() << " ms" << endl;

	cudaDeviceReset();
    return 0;
}


