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
		max_step = 50000;
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
		ret = ret || step > max_step;
		if (ret && recorder != NULL)
			recorder->finalRec(best,step,iteration);
		return ret;
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



//concretes

class TabuSatSolver : public SatSolver , public TabuSolver{

public:

	TabuSatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) , TabuSolver(nbvars,1){

	}

	int solve() {
		
		//state->randAssign();
		int current = state->score;
		int best = current;
		int nbvars = state->size;
		int nbvarlog = (int)(log2(nbvars));
		int step_count = 0;
		int * results = new int [nbvars]; 

		while (best != 2141) {
			int exceptional = best - current + 1;
			int maxeval = -INF;
			int maxind = 0;
			step_count++;

			for (int i = 0; i < nbvars; i++)
			{
				results[i] = state->toggleAndScore(i);
				state->toggle(i);
				if (history[i] + tabu_tenur >= step_count && results[i] < exceptional)
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
			history[maxind] = step_count;
			state->score += maxeval;
			current = state->score;
			if (best < current) {
				cout << "upgrade! :" << current << endl;
				best = current;
			}
		}
		
		return best;
	}
	
};

class SASatSolver : public SatSolver {

public:

	SASatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) {

	}

	int solve() {
		state->randAssign();
		int current = state->score;
		int best = 0;
		cout << "start: " << current << endl;
		int nbvar = L(state->assignment);
		int temp = 10000;
		int counter = 0;
		srand(time(NULL));
		while (1) {
			counter++;
			if (temp < 19950)
				temp += 1;
			cout << temp;
			if (counter > 6000 && temp == 19950)
			{
				counter = 0;
				temp = 15000;
			}
			int maxeval = 0;
			int maxind = 0;
			int i;
			int counter = 0;
			while (1) {
				counter++;
				i = (rand() % nbvar);
				int eval = state->toggleAndScore(i);
				if (eval >= maxeval || rand() % 20000 > temp) {
					maxind = i;
					maxeval = eval;
					break;
				}
				state->toggle(i);
			}

			state->score += maxeval;
			current = state->score;
			if (best < current) {
				cout << "upgrade! :" << current << endl;
				best = current;
			}
		}
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
		return "DeepSatSolver";
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

class CudaGreedySatSolver : public SatSolver {
public:
	int** results;
	int* results_index;
	CudaGreedySatSolver(int nbvars, Cnf* c) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int*));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
		for (int i = 0; i < nbvars; i++) {
			cudaMallocManaged(&results[i], nbvars * sizeof(int));
		}
	}

	int solve() {
		state->randAssign();
		int current = state->score;
		int best = 0;
		int deep_temperature = 20000;
		int main_temperature = 5000;

		cout << "start: " << current << endl;
		int nbvar = state->size;
		int nbvarlog = (int)(log2(nbvar));
		bool deep = false;
		int lastdeep = 0;
		while (1) {
			int maxeval = 0;
			int maxind = 0;
			int maxindd = 0;
			int parentval = 0;

			if (rand() % 10000 < main_temperature) {
				state->score += state->toggleAndScore(rand() % nbvar);
				current = state->score;
				main_temperature -= 2;
			}
			else {
				if (deep) {
					twoStepKernel << < this->state->size, 1 >> > (state, results, results_index);
					cudaDeviceSynchronize();
					maxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> > (results[0], &results_index[0], nbvar, nbvarlog);
					auto err = cudaGetLastError();
					cudaDeviceSynchronize();
					err = cudaGetLastError();
					maxeval = results[0][0];
					maxind = results_index[0];
					maxindd = results_index[maxind];
				}
				else {
					oneStepKernel << < 1, this->state->size >> > (state, results[0], -1);
					cudaDeviceSynchronize();
					maxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> > (results[0], &results_index[0], nbvar, nbvarlog);
					auto err = cudaGetLastError();
					cudaDeviceSynchronize();
					err = cudaGetLastError();
					maxeval = results[0][0];
					maxind = results_index[0];
				}


				//non parallel test
				/*int tmaxeval = 0;
				int tmaxind = 0;
				int tmaxindd = 0;
				int tparentval = 0;
				int teval = 0;
				for (int i = 1; i < nbvar; i++) {
				if (deep) {
				tparentval = state->toggleAndScore(i);
				for (int j = 1; j < nbvar; j++) {
				if (i == j) continue;
				int teval = state->toggleAndScore(j);
				if (teval + tparentval > tmaxeval) {
				tmaxindd = j;
				tmaxind = i;
				tmaxeval = teval + tparentval;
				}
				state->toggle(j);
				}
				state->toggle(i);
				}
				else {
				teval = state->toggleAndScore(i);
				if (teval > tmaxeval) {
				tmaxind = i;
				tmaxeval = teval;
				}
				state->toggle(i);
				}

				}

				if (tmaxeval != maxeval)
				cout << "failed!" << endl;*/
				//end of test

				if (maxeval > 0) {
					if (deep) {
						cout << "found by deep";
						deep = false;
						state->toggle(maxindd);
					}
					state->toggle(maxind);
					state->score += maxeval;
					int eval_test = state->eval();
					current = state->score;
					//if (current != eval_test)
					//cout << "failed score!" << endl;
					cout << current << endl;
				}
				else {
					if (!deep) {
						if (rand() % 20000 < deep_temperature) {
							state->score += state->toggleAndScore(rand() % nbvar);
							current = state->score;
							deep_temperature -= 1;
						}
						else
							deep = true;
						//lastdeep = current;
					}
					else {
						if (rand() % 20000 < deep_temperature) {
							deep = false;
							state->toggle(maxindd);
							state->toggle(maxind);
							state->score += maxeval;
							deep_temperature -= 1;
						}
						else {
							cout << "reset on: " << best << " temps: " << main_temperature << " - " << deep_temperature << endl;
							if (best < current) {
								cout << "upgrade! :" << current << endl;
								best = current;
							}
							state->randAssign();
							current = state->score;
							deep = false;
							main_temperature = 5000;
							deep_temperature = 20000;
							cin.get();
						}
					}
				}
			}//main rand
			if (best < current)
				best = current;
		}
	}

};

class CudaMultiStepSASatSolver : public SatSolver {
public:
	int* results;
	int* results_index;
	bool** assignments;
	curandState *d_state;
	int parallel;

	CudaMultiStepSASatSolver(int nbvars, Cnf* cnf, int parallel) : SatSolver() {
		this->parallel = parallel;
		cudaMalloc(&d_state, parallel * sizeof(curandState));
		randIntitKernel << <(parallel / 32) + 1, 32 >> > (d_state, parallel);
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
		int nbvarlog = (int)(log2(nbvar));
		SAkernel <<<parallel, 1 >>> (state, d_state, results, assignments);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		maxKernel << <1, 500, (2 * parallel * sizeof(int)) >> > (results, &results_index[0], parallel, (int)(log2(parallel)));
		err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		cout << endl << endl << "Done ! best was : " << results[0];
		return results[0];
	}

};

class CudaSingleStepSASatSolver : public SatSolver {
public:
	int* results;
	int* results_index;
	int* tabu_history;
	int tabu_step;
	bool tabu;
	CudaSingleStepSASatSolver(int nbvars, Cnf* c, bool tabu) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
		this->tabu = tabu;
		if (tabu) {
			tabu_step = (int)(log2(nbvars));
			cudaMallocManaged(&tabu_history, nbvars * sizeof(int));
			for (int i = 0; i < nbvars; i++) {
				tabu_history[i] = -tabu_step;
			}
		}
	}

	int solve() {
		srand(time(NULL));
		state->randAssign();
		bool cuda = false;
		int nbvar = state->size;
		int nbvarlog = (int)(log2(nbvar));
		int maxeval = -100000;
		int maxind = -1;
		int best = 0;
		int counter = 0;
		int temperature = 4000;
		const int max_temperature = 8000;
		int *number_of_positives;
		cudaMallocManaged(&number_of_positives, sizeof(int));
		float SA_randomwalk_probability;
		float SA_current_probability;
		while (temperature > 0) {
			counter++;
			if (temperature < 2000) {
				//cout << temperature << " ";
				if (!cuda) cout << "------------"<< cuda++ << endl;
				oneStepKernel << < 1, nbvar >> > (state, results, -1);
				cudaDeviceSynchronize();
				/*if (tabu) {
					tabuMaxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> > (results, results_index,tabu_history, counter, tabu_step,(best-state->score), nbvar, nbvarlog);
				}
				else {
					maxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> > (results, results_index, nbvar, nbvarlog);
				}*/
				auto err = cudaGetLastError();
				cudaDeviceSynchronize();
				maxeval = results[0];
				maxind = results_index[0];
				err = cudaGetLastError();
				sumOrCountKernel << <1, 200, (nbvar * sizeof(int)) >> > (results, number_of_positives, nbvar, nbvarlog, true);
				cudaDeviceSynchronize();
				SA_current_probability = temperature < max_temperature/10000 ? 0.0001 : (temperature / (float)max_temperature);
				float non_positive_ratio = (1 - ((float)*number_of_positives / nbvar));
				SA_randomwalk_probability = (non_positive_ratio * SA_current_probability) / (1 - non_positive_ratio * (1 - SA_current_probability));
				if (number_of_positives == 0)
					SA_randomwalk_probability = 0;
				if (randomRangeUniform(100000) < (int)(SA_randomwalk_probability * 100000)) {
					maxind = randomRangeUniform(nbvar);
					state->score += state->toggleAndScore(maxind);
				}
				else {
					state->score += maxeval;
					state->toggle(maxind);
				}
				/*if(tabu)
					tabu_history[maxind] = counter;*/
			}
			else {
				//cout << temperature;
				while (1) {
					int i = randomRangeUniform(nbvar);
					int eval = state->toggleAndScore(i);
					int rand_temp = randomRangeUniform(max_temperature);
					if (eval > 0 || rand_temp < temperature + 1) {
						state->score += eval;
						break;
					}else
						state->toggle(i);
				}

			}

			if (best < state->score) {
				cout << "upgrade! " << state->score << endl;
				best = state->score;
			}
			temperature--;
		}
		
		return best;
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
				maxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> >(results[0], &results_index[0], nbvar, nbvarlog);
				auto err = cudaGetLastError();
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

	CudaMultiStepTabuSatSolver(int nbvars, Cnf* cnf,int parallel = 128) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars,parallel) {
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
		//state->randAssign();
		int nbvar = states[0]->size;
		int nbvarlog = (int)(log2(nbvar));
		int maxscore = 0;
		int maxind = -1;

		Tabukernel <<< parallel / 32, 32 >>> (states, results, results_index,  histories,  tabu_tenur, nbvar,  nbvarlog);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		for (int i = 0; i < parallel ; i++)
		{
			if (results[i][0] > maxscore)
			{	
				maxscore = results[i][0];
				maxind = i;
			}
		}

		cout << "best was :" << maxscore << endl;

		return maxscore;
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
		setMaxStep(sqrt(log2(nbvars)) * 1000);
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



};


class CudaSingleStepTabuSatSolver : public CudaSatSolver, public TabuSolver {
	public:
		int* results;
		int* results_index;
		CudaSingleStepTabuSatSolver(int nbvars, Cnf* cnf) : CudaSatSolver(nbvars, cnf), TabuSolver(nbvars, 1) {
			cudaMallocManaged(&results, nbvars * sizeof(int));
			cudaMallocManaged(&results_index, sizeof(int));
			setMaxStep((int)(log2(nbvars))*1000);
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


};

SatSolver* SatSolver::factory(string type, int nbvars, Cnf* cnf) {
	if (type == solvers_list[0])
		return new GreedySatSolver(nbvars, cnf);
	else if (type == solvers_list[1])
		return new GreedyDeepSatSolver(nbvars, cnf);
	else if (type == solvers_list[2])
		return new CudaGreedyDeepSatSolver(nbvars, cnf);
	else
		return NULL;
}


#endif
