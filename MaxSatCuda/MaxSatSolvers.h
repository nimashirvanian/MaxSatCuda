#ifndef MAXSAT_SOLVERS_H
#define MAXSAT_SOLVERS_H


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
#include "MaxSatUtilities.h"

using namespace std;

// abstracts
class SatSolver {

private: 
	Recorder* recorder;
	vector<bool> solution;
	int max_step;
protected: 
	SatState *state;
	int best;
	int step;
	int iteration;
	 
public:
	const static vector<string> solvers_list; 
	SatSolver() {
		best = 0;
		step = 0;
		iteration = 1;
		max_step = 500000;
		recorder = NULL;
	}
	SatSolver(int nbvars, Cnf* c):SatSolver() {
		state = new SatState(c, nbvars);
	}

	static SatSolver* factory(string type, int nbvars, Cnf* cnf);

	virtual int solve() { return 0; }

	virtual string getName() { return "unidetified"; }

	void setRecorder(Recorder* recorder) {
		this->recorder = recorder;

	}

	void checkUpgrade() {
		if (best < state->score)
		{
			best = state->score;
			if (recorder != NULL)
				recorder->upgradeRec(best, step, iteration);
			//else
				cout << "upgrade! " << best << endl;
			//fill solution
		}
	}

	void resetSearch() {
		cout << "reset on: " << state->score << endl;
		if(recorder !=NULL)
			recorder->resetRec(state->score, step);
		state->randAssign();
		iteration++;
	}

	void setMaxStep(int max_step) {
		this->max_step = max_step;
	}

	bool checkStop() {
		bool ret = false;
		if (recorder != NULL)
			ret = recorder->timeOut() || best==recorder->optimal;
		ret = ret || step == max_step;
		if (ret && recorder != NULL)
			recorder->finalRec(best,step,iteration);
		return ret;
	}

	void multiStepStop() {
		if (recorder != NULL)
			recorder->finalRecforMultistep(best, step);
	}

};

class CudaSatSolver: public SatSolver {

public:
	CudaSatSolver(int nbvars, Cnf* cnf) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(cnf, nbvars);
		cout << "solver constructed" << endl;
	}

	virtual int solve() { return 0; }

};

class TabuSolver {
protected:
	int* history;
	int** histories;
	int tabu_tenur;
public:
	TabuSolver(int nbvars, int tabu_tenur, int parallel) {
		if (parallel == 1) {
			cudaMallocManaged(&history, nbvars * sizeof(int));
			for (int i = 0; i < nbvars; i++)
				history[i] = -tabu_tenur;
		}
		else {
			cudaMallocManaged(&histories, parallel * sizeof(int*));
			for (int i = 0; i < parallel; i++) {
				cudaMallocManaged(&histories[i], nbvars * sizeof(int));
				for (int j = 0; j < nbvars; j++)
					histories[i][j] = -tabu_tenur;
			}

		}
		this->tabu_tenur = tabu_tenur;
	}
	TabuSolver(int nbvars, int parallel) :TabuSolver(nbvars, 4*(log2(nbvars)),parallel){
	}
};

class SASolver {
protected:
	int temperature;
	int max_temperature;
public:
	SASolver(int nbvars) {
		this->max_temperature = log2(nbvars)*log2(nbvars)*2800;
		this->temperature = max_temperature/2;
	}

	void coolDown() {
		temperature--;
	}

	double getP(int delta =1) {
		return (double)temperature / (double)max_temperature;
	}

	bool booleanByProbability (double p){
		return randomRangeUniform(1000000) < (int)(p * 1000000);
	}

};



//concretes

class TabuSatSolver : public SatSolver , public TabuSolver{

public:

	TabuSatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) , TabuSolver(nbvars,1){
		//setMaxStep(log2(nbvars) * 2000);
		setMaxStep(-1);
	}

	int solve() {
		
		int nbvars = state->size;
		int nbvarlog = (int)(log2(nbvars));
		int * results = new int [nbvars]; 

		while (!checkStop()) {
			int exceptional = best - state->score + 1;
			int maxeval = -INF;
			int maxind = 0;
			step++;

			for (int i = 0; i < nbvars; i++)
			{
				results[i] = state->toggleAndScore(i);
				state->toggle(i);
				if (history[i] + tabu_tenur >= step && results[i] < exceptional)
					continue;
				else
				{
					if (results[i] > maxeval) {
						maxeval = results[i];
						maxind = i;
					}
				}
			}

			state->toggle(maxind);
			history[maxind] = step;
			state->score += maxeval;
			checkUpgrade();
		}
		
		return best;
	}

	string getName() {
		return "TabuSatSolver";
	}
	
};

class SASatSolver : public SatSolver,public SASolver {

public:

	SASatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c), SASolver(nbvars) {
		setMaxStep(temperature-2);
	}

	int solve() {
		int nbvar = L(state->assignment);
		while (!checkStop()) {
			step++;
			int eval = 0;
			int ind = 0;
			bool worsen;
			while (1) {
				ind = randomRangeUniform(nbvar);
				eval = state->toggleAndScore(ind);
				worsen = booleanByProbability(getP());
				if (eval > 0 || worsen) {
					state->score += eval;
					break;
				}
				state->toggle(ind);
			}
			coolDown();
			checkUpgrade();
		}
		return best;
	}

	string getName() {
		return "SASatSolver";
	}

};

class GreedyDeepSatSolver : public SatSolver {


public:
	GreedyDeepSatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) {

	}

	int solve() {

		int nbvar = L(state->assignment);
		bool deep = false;
		//int go = 100;
		while (!checkStop()) {
			step++;
			int maxeval = 0;
			int maxind = 0;
			int maxindd = 0;
			int parentval = 0;
			for (int i = 1; i < nbvar; i++) {
				if (deep) {
					parentval = state->toggleAndScore(i);
					for (int j = 1; j < nbvar; j++) {
						if (i == j) continue;
						int eval = state->toggleAndScore(j);
						if (eval + parentval > maxeval) {
							maxindd = j;
							maxind = i;
							maxeval = eval + parentval;
						}
						state->toggle(j);
					}
					state->toggle(i);
				}
				else {
					int eval = state->toggleAndScore(i);
					if (eval > maxeval) {
						maxind = i;
						maxeval = eval;
					}
					state->toggle(i);
				}

			}

			if (maxeval > 0) {
				if (deep) {
					deep = false;
					state->toggle(maxindd);
				}
				state->toggle(maxind);
				state->score += maxeval;
				checkUpgrade();
			}
			else {
				if (!deep) {
					deep = true;
				}
				else {
					resetSearch();
					deep = false;
				}
			}
		}
		return best;
	}

	string getName() {
		return "GreedyDeepSatSolver";
	}

};

class GreedySatSolver : public SatSolver {


public:

	GreedySatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) {

	}

	int solve() {
		int nbvar = L(state->assignment);
		while (!checkStop()) {
			step++;
			int maxeval = 0;
			int maxind = 0;
			int maxindd = 0;
			for (int i = 1; i < nbvar; i++) {
					int eval = state->toggleAndScore(i);
					if (eval > maxeval) {
						maxind = i;
						maxeval = eval;
					}
					state->toggle(i);
			}

			if (maxeval > 0) {
				state->toggle(maxind);
				state->score += maxeval;
				checkUpgrade();
			}
			else{
				resetSearch();
				state->score;
			}
		}
		return best;
	}

	string getName() {
		return "GreedySatSolver";
	}

};


class CudaMultiStepSASatSolver : public SatSolver, public SASolver {
public:
	int* results;
	int* results_index;
	bool** assignments;
	curandState *d_state;
	int parallel;

	CudaMultiStepSASatSolver(int nbvars, Cnf* cnf, int parallel=96) : SatSolver(), SASolver(nbvars) {
		this->parallel = parallel;
		cudaMalloc(&d_state, parallel * sizeof(curandState));
		randIntitKernel << <(parallel / 32), 32 >> > (d_state, parallel, rand()%100);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(cnf, nbvars);
		cudaMallocManaged(&results, parallel * sizeof(int));
		cudaMallocManaged(&assignments, parallel * sizeof(bool*));
		cudaMallocManaged(&results_index, parallel * sizeof(int));
		for (int i = 0; i < parallel; i++) {
			//cudaMallocManaged(&results[i], nbvars * sizeof(int));
			cudaMallocManaged(&assignments[i], nbvars * sizeof(bool));
		}
	}

	int solve() {
		state->randAssign();
		int nbvar = state->size;
		max_temperature = 40000;
		int nbvarlog = (int)(log2(nbvar));
		SAkernel <<<parallel, 1 >>> (state, d_state, results, assignments,max_temperature);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		maxKernel << <1, 128, (2 * parallel * sizeof(int)) >> > (results, &results_index[0], parallel, (int)(log2(parallel)));
		err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		cout << endl << endl << "Done ! best was : " << results[0];
		step = max_temperature/2;
		best = results[0];
		cout << "best was :" << best << endl;
		multiStepStop();
		return best;
	}

	string getName() {
		return "CudaMultiStepSASatSolver";
	}

};

class CudaSingleStepSSASatSolver : public SatSolver, public SASolver {
public:
	int* results;
	int* results_index;
	curandState *d_state;
	CudaSingleStepSSASatSolver(int nbvars, Cnf* c) : SatSolver(), SASolver(nbvars) {
		cudaMalloc(&d_state, nbvars * sizeof(curandState));
		randIntitKernel << <1 , nbvars >> > (d_state, nbvars,rand()%100);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
		setMaxStep(temperature-1);
	}

	int solve() {
		int nbvar = state->size;
		int nbvarlog = (int)(log2(nbvar));
		int eval = -INF;
		int ind = -1;
		int *number_of_positives;
		cudaMallocManaged(&number_of_positives, sizeof(int));
		*number_of_positives = nbvar;
		double SA_randomwalk_probability;
		double SA_current_probability;
		while (!checkStop()) {
			step++;
			if (temperature < max_temperature/100) {
				//cout << temperature << " ";
				oneStepKernel << < 1, nbvar >> > (state, results, -1);
				cudaDeviceSynchronize();
				randomPositivePickKernel << <1, 128, (2 * nbvar * sizeof(int)) >> > (results, results_index, nbvar, nbvarlog,d_state);
				auto err = cudaGetLastError();
				cudaDeviceSynchronize();
				eval = results[0];
				ind = results_index[0];
				err = cudaGetLastError();
				sumOrCountPositivesKernel << <1, 128, (nbvar * sizeof(int)) >> > (results, number_of_positives, nbvar, nbvarlog, true);
				cudaDeviceSynchronize();
				SA_current_probability = getP();
				double non_positive_ratio = (1 - ((double)*number_of_positives / nbvar));
				SA_randomwalk_probability = (non_positive_ratio * SA_current_probability) / (1 - non_positive_ratio * (1 - SA_current_probability));
				if (number_of_positives == 0)
					SA_randomwalk_probability = 1;
				//if (temperature < 5000)
				//	cout << temperature<<" ";
				if (booleanByProbability(SA_randomwalk_probability)) {
					while (1) {
						ind = randomRangeUniform(nbvar);
						eval = state->toggleAndScore(ind);
						if (eval <= 0) {
							state->score += eval;
							break;
						}
						state->toggle(ind);
					}
				}
				else {
					state->score += eval;
					state->toggle(ind);
				}
			}
			else {
				while (1) {
					int i = randomRangeUniform(nbvar);
					int eval = state->toggleAndScore(i);
					if (eval > 0 || booleanByProbability(getP())) {
						state->score += eval;
						break;
					}else
						state->toggle(i);
				}

			}

			checkUpgrade();
			coolDown();
		}
		
		return best;
	}

	string getName() {
		return "CudaSingleStepSSASatSolver";
	}

};

class CudaGreedyDeepSatSolver : public SatSolver {
public:
	int** results;
	int* results_index;
	CudaGreedyDeepSatSolver(int nbvars, Cnf* c) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
		for (int i = 0; i < nbvars; i++) {
			cudaMallocManaged(&results[i], nbvars * sizeof(int));
		}
	}

	int solve() {
		int nbvar = state->size;
		int nbvarlog = (int)(log2(nbvar));
		bool deep = false;
		while (!checkStop()) {
			step++;
			int maxeval = 0;
			int maxind = 0;
			int maxindd = 0;
			int parentval = 0;

			if (deep) {
				twoStepKernel << < this->state->size, 1 >> >(state, results, results_index);
				cudaDeviceSynchronize();
				auto err = cudaGetLastError();
				maxKernel << <1, 128, (2 * nbvar * sizeof(int)) >> >(results[0], &results_index[0], nbvar, nbvarlog);
				err = cudaGetLastError();
				cudaDeviceSynchronize();
				err = cudaGetLastError();
				maxeval = results[0][0];
				maxind = results_index[0];
				maxindd = results_index[maxind];
			}
			else {
				oneStepKernel << < 1, this->state->size >> >(state, results[0], -1);
				cudaDeviceSynchronize();
				maxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> >(results[0], &results_index[0], nbvar, nbvarlog);
				auto err = cudaGetLastError();
				cudaDeviceSynchronize();
				err = cudaGetLastError();
				maxeval = results[0][0];
				maxind = results_index[0];
			}


			if (maxeval > 0) {
				if (deep) {
					deep = false;
					state->toggle(maxindd);
				}
				state->toggle(maxind);
				state->score += maxeval;
				checkUpgrade();
			}
			else {
				if (!deep) {
					deep = true;
				}
				else {
					resetSearch();
					deep = false;
				}
			}
		}
		return best;
	}

	string getName() {
		return "CudaGreedyDeepSatSolver";
	}

};

class CudaMultiStepTabuSatSolver : public CudaSatSolver, public TabuSolver {
public:
	int** results;
	int* results_index;
	bool ** assignments;
	SatState** states;
	int parallel;

	CudaMultiStepTabuSatSolver(int nbvars, Cnf* cnf,int parallel = 96) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars,parallel) {
		cudaMallocManaged(&results, parallel * sizeof(int*));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
		for (int i = 0; i < parallel; i++) {
			cudaMallocManaged(&results[i], nbvars * sizeof(int));
		}
		cudaMallocManaged(&assignments, parallel * sizeof(bool *));
		cudaMallocManaged(&states, parallel * sizeof(SatState*));
		for (int i = 0; i < parallel; i++) {
			cudaMallocManaged(&states[i],sizeof(SatState));
			cudaMallocManaged(&assignments[i], nbvars * sizeof(bool));
			new(states[i]) SatState(cnf , nbvars);
		}
		this->parallel = parallel;
	}

	int solve() {
		int nbvar = states[0]->size;
		int nbvarlog = (int)(log2(nbvar));
		int maxind = -1;
		int max_step = 3*nbvar/4;//(log2(nbvar)) * nbvar/2;
		Tabukernel <<< parallel / 16, 16 >>> (states, results, results_index,  histories,  tabu_tenur, nbvar,  nbvarlog, max_step);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		for (int i = 0; i < parallel ; i++)
		{
			if (results[i][0] > best)
			{	
				best = results[i][0];
				maxind = i;
			}
		}
		step = max_step;
		cout << "best was :" << best << endl;
		multiStepStop();
		return best;
	}
	
	string getName() {
		return "CudaMultiStepTabuSatSolver";
	}
};

class CudaDeepSingleStepTabuSatSolver : public CudaSatSolver, public TabuSolver {
public:
	int** results;
	int* results_index;
	CudaDeepSingleStepTabuSatSolver(int nbvars, Cnf* cnf) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars, 2 * (log2(nbvars)),1) {
		cudaMallocManaged(&results, nbvars * sizeof(int*));
		cudaMallocManaged(&results_index, nbvars* sizeof(int));
		for (int i = 0; i < nbvars; i++)
			cudaMallocManaged(&results[i], nbvars * sizeof(int));
		//setMaxStep(sqrt(log2(nbvars)) * 5000);
		setMaxStep(-1);
	}

//	CudaDeepSingleStepTabuSatSolver(int nbvars, Cnf* cnf, int tabu_tenur) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars, tabu_tenur, 1), CudaDeepSingleStepTabuSatSolver(nbvars, cnf){
//	}

	int solve() {
		int nbvars = state->size;
		int nbvarlog = (int)(log2(nbvars));
		int maxeval = 0;
		int maxind = 0;
		int maxindd = 0;
		int exceptional = 1;
		while (!checkStop()) {
			exceptional = best - state->score + 1;
			step++;
			if (step % 100 == 0)
				cout << step << endl;

			oneStepKernel << < 1, nbvars >> >(state, results[0], -1);
			cudaDeviceSynchronize();
			tabuMaxKernel << <1, 256, (2 * nbvars * sizeof(int)) >> >(results[0], &results_index[0], history, step, tabu_tenur, exceptional, nbvars, nbvarlog);
			auto err = cudaGetLastError();
			cudaDeviceSynchronize();
			err = cudaGetLastError();
			maxeval = results[0][0];
			maxind = results_index[0];

			if (maxeval > 0) {
				state->toggle(maxind);
				history[maxind] = step;
				state->score += maxeval;
			}
			else {
				twoStepTabuKernel << < nbvars, 1 >> > (state, results, results_index, history, step, tabu_tenur, exceptional);
				cudaDeviceSynchronize();
				tabuMaxKernel << <1, 256, (2 * nbvars * sizeof(int)) >> > (results[0], results_index, history, step, tabu_tenur, exceptional, nbvars, nbvarlog);
				auto err = cudaGetLastError();
				cudaDeviceSynchronize();
				err = cudaGetLastError();
				maxeval = results[0][0];
				maxind = results_index[0];
				maxindd = results_index[maxind];
				state->toggle(maxind);
				state->toggle(maxindd);
				history[maxind] = step;
				history[maxindd] = step;
				state->score += maxeval;
			//	cout << "deep";
			}
			checkUpgrade();
		}
		return best;
	}

	string getName() {
		return "CudaDeepSingleStepTabuSatSolver";
	}

};

class CudaSingleStepTabuSatSolver : public CudaSatSolver, public TabuSolver {
	public:
		int* results;
		int* results_index;
		CudaSingleStepTabuSatSolver(int nbvars, Cnf* cnf) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars, 1) {
			cudaMallocManaged(&results, nbvars * sizeof(int));
			cudaMallocManaged(&results_index, sizeof(int));
			//setMaxStep((int)(log2(nbvars))*5000);
			setMaxStep(-1);
		}

	//	CudaSingleStepTabuSatSolver(int nbvars, Cnf* cnf, int tabu_tenur) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars, tabu_tenur, 1) ,CudaSingleStepTabuSatSolver(nbvars, cnf){
	//	}

		int solve() {
			int nbvars = state->size;
			int nbvarlog = (int)(log2(nbvars));

			while (!checkStop()) {
				int maxeval = 0;
				int maxind = 0;
				step++;
				if (step % 100 == 0)
					cout << step << endl;

				oneStepKernel << < 1, nbvars >> >(state, results, -1);
				cudaDeviceSynchronize();
				tabuMaxKernel << <1, 128, (2 * nbvars * sizeof(int)) >> >(results, &results_index[0], history, step, tabu_tenur, (best - state->score + 1), nbvars, nbvarlog);
				auto err = cudaGetLastError();
				cudaDeviceSynchronize();
				err = cudaGetLastError();
				maxeval = results[0];
				maxind = results_index[0];

				state->toggle(maxind);
				history[maxind] = step;
				state->score += maxeval;
				checkUpgrade();
			}
			return best;
		}
		string getName() {
			return "CudaSingleStepTabuSatSolver";
		}

};

SatSolver* SatSolver::factory(string type, int nbvars, Cnf* cnf) {
	if (type == solvers_list[0])
		return new GreedySatSolver(nbvars, cnf);
	else if (type == solvers_list[1])
		return new GreedyDeepSatSolver(nbvars, cnf);
	else if (type == solvers_list[2])
		return new CudaGreedyDeepSatSolver(nbvars, cnf);
	else if (type == solvers_list[3])
		return new TabuSatSolver(nbvars, cnf);
	else if (type == solvers_list[4])
		return new CudaSingleStepTabuSatSolver(nbvars, cnf);
	else if (type == solvers_list[5])
		return new CudaDeepSingleStepTabuSatSolver(nbvars, cnf);
	else if (type == solvers_list[6])
		return new CudaMultiStepTabuSatSolver(nbvars, cnf);
	else if (type == solvers_list[7])
		return new SASatSolver(nbvars, cnf);
	else if (type == solvers_list[8])
		return new CudaSingleStepSSASatSolver(nbvars, cnf);
	else if (type == solvers_list[9])
		return new CudaMultiStepSASatSolver(nbvars, cnf);
	else
		return NULL;
}


#endif
