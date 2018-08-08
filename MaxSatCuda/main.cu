

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
	cudaDeviceReset();
	int nbvars, nbclauses;
	cin >> nbvars >> nbclauses;
	nbvars++;
	Cnf* cnf;
	cudaMallocManaged(&cnf, sizeof(Cnf));
	new(cnf) Cnf(nbvars);
	while (nbclauses--) {
		int tempvar;
		Clause *tmpcls = new Clause();
		cin >> tempvar;
		while (tempvar) {
			int id = abs(tempvar);
			bool sgn = tempvar > 0;
			tmpcls->addLiteral(id, sgn);
			cin >> tempvar;
		}
		cnf->addClause(*tmpcls);
	}

	//SatSolver *solver = new CudaSingleStepSASatSolver(nbvars, cnf);
	SatSolver *solver = new GreedySatSolver(nbvars, cnf);
	
	solver->solve();
	cudaDeviceReset();
    return 0;
}


