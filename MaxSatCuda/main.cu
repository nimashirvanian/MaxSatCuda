

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

const vector<string> SatSolver::solvers_list = { "GreedySatSolver", "GreedyDeepSatSolver", "CudaGreedyDeepSatSolver" };

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
	inFile.close();

	cnf->cudable();

//for(auto &type : SatSolver::solvers_list)
		//for (int i = 11; i <=20 ; i++) {
			//string type = "CudaGreedyDeepSatSolver";
		//	cout << endl << "------------ " << type << "---" << i << endl;
			//SatSolver *solver = SatSolver::factory(type,nbvars, cnf);
			SatSolver *solver = new CudaDeepSingleStepTabuSatSolver(nbvars, cnf);
		//	Recorder *recorder = new Recorder(solver->getName(), "1", "collective1", to_string(i), 150);
		//	solver->setRecorder(recorder);
			//auto start = chrono::steady_clock::now();
			//  Insert the code that will be timed
		//	recorder->start();
			solver->solve();
		//}

	//auto end = chrono::steady_clock::now();
	// Store the time difference between start and end
	//auto diff = end - start;
	//cout << endl<<chrono::duration <double, milli>(diff).count() << " ms" << endl;

	cudaDeviceReset();
    return 0;
}


