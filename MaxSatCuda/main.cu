

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

const vector<string> SatSolver::solvers_list = { "GreedySatSolver", "GreedyDeepSatSolver", "CudaGreedyDeepSatSolver", "TabuSatSolver", "CudaSingleStepTabuSatSolver", "CudaDeepSingleStepTabuSatSolver","CudaMultiStepTabuSatSolver", "SASatSolver","CudaSingleStepSSASatSolver", "CudaMultiStepSASatSolver" };

int main()
{
	//sumOrCountFanTest();
	//classInCudaTest();
	//sortTest();
	srand(time(NULL));
	cudaDeviceReset();
	int nbvars, nbclauses;
	ifstream inFile;
	inFile.open("4-s3v110c1000-2-random-973.cnf");
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

	for(int s=7; s <9; /*SatSolver::solvers_list.size();*/ s++)
		for (int i = 1; i <=20; i++) {
			//string type = "CudaGreedyDeepSatSolver";
			cout << endl << "------------ " << SatSolver::solvers_list[s] << "---" << i << endl;
			SatSolver *solver = SatSolver::factory(SatSolver::solvers_list[s],nbvars, cnf);
			//SatSolver *solver = new CudaMultiStepTabuSatSolver(nbvars, cnf);
			Recorder *recorder = new Recorder(solver->getName(), "3", "SAcollective1-4", to_string(i), 150, 973);
			solver->setRecorder(recorder);
			recorder->start();
			solver->solve();
		}


	cudaDeviceReset();
    return 0;
}


