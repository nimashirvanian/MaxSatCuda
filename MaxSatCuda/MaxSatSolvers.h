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

class SatSolver {

protected: SatState *state;

public:
	SatSolver() {}
	SatSolver(int nbvars, Cnf* c) {
		state = new SatState(c, nbvars);
	}

	virtual int solve() { return 0; }

};


class GreedySatSolver : public SatSolver {

public:

	GreedySatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) {

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
			int maxindd = 0;
			int i;
			int counter = 0;
			// bool reset = false;
			while (1) {
				counter++;
				i = (rand() % nbvar);
				int eval = state->toggleAndScore(i);
				if (eval >= maxeval || rand() % 20000 > temp || counter > nbvar) {
					maxind = i;
					maxeval = eval;
					break;
				}
				state->toggle(i);

				// if(maxeval > 0)
				//    break;
			}

			state->score += maxeval;
			//current = state->eval();
			current = state->score;
			//	cout << current << "-" << state->score << endl;
			if (best < current) {
				cout << "upgrade! :" << current << endl;
				best = current;
			}
		}
	}

};


class DeepSatSolver : public SatSolver {


public:

	DeepSatSolver(int nbvars, Cnf* c) :SatSolver(nbvars, c) {

	}

	int solve() {
		state->randAssign();
		int current = state->score;
		int best = 0;
		cout << "start: " << current << endl;
		int nbvar = L(state->assignment);
		vb cstate = state->assignment;
		bool deep = false;
		int lastdeep = 0;
		//int go = 100;
		while (1) {
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

				if (maxeval > 0)
					break;
			}
			if (maxeval > 0) {
				if (deep) {
					cout << "found by deep";
					deep = false;
					state->toggle(maxindd);
				}
				state->toggle(maxind);

				state->score += maxeval;
				//current = state->eval();
				current = state->score;
				cout << current << "-" << state->score << endl;
			}
			else {
				if (!deep && current > lastdeep - 10) {
					deep = true;
					lastdeep = current;
				}
				else {
					cout << "reset on: " << current << endl;
					if (best < current) {
						cout << "upgrade! :" << current << endl;
						best = current;
					}
					state->randAssign();
					current = state->score;
					deep = false;
				}
			}
		}
	}

};

class CudaGreedySatSolver : public SatSolver {
public:
	int** results;
	int* results_index;
	CudaGreedySatSolver(int nbvars, Cnf* c) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int));
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
	CudaMultiStepSASatSolver(int nbvars, Cnf* c) : SatSolver() {
		parallel = 96;
		cudaMalloc(&d_state, parallel * sizeof(curandState));
		randIntitKernel << <(parallel / 32) + 1, 32 >> > (d_state, parallel);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
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
		SAkernel << <parallel, 1 >> > (state, d_state, results, assignments);
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
	CudaSingleStepSASatSolver(int nbvars, Cnf* c) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
	}

	int solve() {
		srand(time(NULL));
		state->randAssign();
		int nbvar = state->size;
		int nbvarlog = (int)(log2(nbvar));
		int maxeval = -100000;
		int maxind = -1;
		int best = 0;
		int temperature = 25000;
		const int max_temperature = 50000;
		int *number_of_positives;
		cudaMallocManaged(&number_of_positives, sizeof(int));
		float SA_randomwalk_probability;
		float SA_current_probability;
		while (temperature > -5000) {
			if (temperature < max_temperature / 20) {
				cout << temperature << " ";
				oneStepKernel << < 1, nbvar >> > (state, results, -1);
				cudaDeviceSynchronize();
				maxKernel << <1, 500, (2 * nbvar * sizeof(int)) >> > (results, results_index, nbvar, nbvarlog);
				auto err = cudaGetLastError();
				cudaDeviceSynchronize();
				maxeval = results[0];
				maxind = results_index[0];
				err = cudaGetLastError();
				sumOrCountKernel << <1, 200, (nbvar * sizeof(int)) >> > (results, number_of_positives, nbvar, nbvarlog, true);
				cudaDeviceSynchronize();
				SA_current_probability = temperature < max_temperature/100 ? 0.01 : (temperature / (float)max_temperature);
				float non_positive_ratio = (1 - ((float)*number_of_positives / nbvar));
				SA_randomwalk_probability = (non_positive_ratio * SA_current_probability) / (1 - non_positive_ratio * (1 - SA_current_probability));
				if (number_of_positives == 0)
					SA_randomwalk_probability = 0;
				if (rand() % 100000 < SA_randomwalk_probability * 100000) {
					state->score += state->toggleAndScore(rand() % nbvar);
				}
				else {
					state->score += maxeval;
					state->toggle(maxind);
				}
			}
			else {
				cout << ".";
				while (1) {
					int i = (rand() % nbvar);
					int eval = state->toggleAndScore(i);
					if (eval > 0 || rand() % max_temperature < temperature) {
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

class CudaDeepSatSolver : public SatSolver {
public:
	int** results;
	int* results_index;
	CudaDeepSatSolver(int nbvars, Cnf* c) : SatSolver() {
		cudaMallocManaged(&state, sizeof(SatState));
		new(state) SatState(c, nbvars);
		cudaMallocManaged(&results, nbvars * sizeof(int));
		cudaMallocManaged(&results_index, nbvars * sizeof(int));
		for (int i = 0; i < nbvars; i++) {
			cudaMallocManaged(&results[i], nbvars * sizeof(int));
		}
	}

	int solve() {
		state->randAssign();
		int current = state->score;
		int best = 0;

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
					//cout << "found by deep";
					deep = false;
					state->toggle(maxindd);
				}
				state->toggle(maxind);
				state->score += maxeval;
				current = state->score;
			}
			else {
				if (!deep) {
					deep = true;
				}
				else {
					cout << "reset on: " << current << endl;
					if (best < current) {
						cout << "upgrade! :" << current << endl;
						best = current;
					}
					state->randAssign();
					current = state->score;
					deep = false;
				}
			}
		}
	}

};

#endif
