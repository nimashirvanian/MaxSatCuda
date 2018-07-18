

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

using namespace std;

typedef vector< pair<int, bool> > vib;
typedef vector<bool> vb;
typedef vector<int> vi;

#define L(a) (int)((a).size())
#define all(a) (a).begin(), (a).end()
#define mp make_pair

#define Trace(X) cerr << #X << " = " << X << endl
#define _ << " _ " << 




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

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void classTest(Test *t,int size) {
	int ti = threadIdx.x;
	if(ti<size)
		printf("%u \n",t[ti].testFunc());
}



class Clause {
public:
	vib literals;
	int* cu_vars;
	bool* cu_signs;
	int size;

	void cudable() {
		size = literals.size();
		cudaMallocManaged(&cu_vars,size * sizeof(int));
		cudaMallocManaged(&cu_signs, size * sizeof(bool));
		for (int i = 0; i < size; i++) {
			cu_vars[i] = literals[i].first;
			cu_signs[i] = literals[i].second;

		}
	}

	bool bEval(vb &assignment) {
		for (int i = 0; i < L(literals); i++) {
			if (!(assignment[literals[i].first] ^ literals[i].second))
				return true;
		}
		return false;
	}

	__device__ bool bToggleEval(bool* assignment, int toggle_index,int parent_index) {
		for (int i = 0; i < size; i++)
			if (!(assignment[cu_vars[i]] ^ (cu_signs[i] ^ (cu_vars[i] == toggle_index || cu_vars[i] == parent_index))))
				return true;
		return false;
	}

	int iEval(vb &assignment) {
		int res = 0;
		for (int i = 0; i < L(literals); i++) {
			if (!(assignment[literals[i].first] ^ literals[i].second))
				res++;
		}
		return res;
	}

	void addLiteral(int var, bool sign) {
		literals.push_back({ var,sign });
	}

	void print() {
		int n = L(literals);
		cout << "( ";
		for (int i = 0; i < n; i++) {
			if (!literals[i].second)
				cout << "-";
			cout << literals[i].first;
			if (i<n - 1) cout << " | ";
		}
		cout << ")";
	}

	__device__ bool cuBEval(bool* assignment) {
		for (int i = 0; i < size; i++) {
			if (!(assignment[cu_vars[i]] ^ cu_signs[i]))
				return true;
		}
		return false;
	}

};

class Cnf {
public:
	vector<Clause> clauses;
	vector<vib> vars;

	Clause* cu_clauses;
	int** cu_vars_cind;
	bool** cu_vars_csgn;
	int * cu_vars_size;


	Cnf() {}

	Cnf(int nbvars) {
		vars.resize(nbvars);
	}

	void cudable() {
		cudaMallocManaged(&cu_clauses, L(clauses) * sizeof(Clause));
		for (int i = 0; i < L(clauses); i++) {
			clauses[i].cudable();
			cu_clauses[i] = clauses[i];
		}
		cudaMallocManaged(&cu_vars_cind, L(vars) * sizeof(int*));
		cudaMallocManaged(&cu_vars_csgn, L(vars) * sizeof(bool*));
		cudaMallocManaged(&cu_vars_size, L(vars) * sizeof(int));
		for (int i = 0; i < L(vars); i++) {
			cu_vars_size[i] = L(vars[i]);
			cudaMallocManaged(&cu_vars_cind[i], L(vars[i]) * sizeof(int));
			cudaMallocManaged(&cu_vars_csgn[i], L(vars[i]) * sizeof(bool));
			for (int j = 0; j < L(vars[i]); j++) {
				cu_vars_cind[i][j] = vars[i][j].first;
				cu_vars_csgn[i][j] = vars[i][j].second;
			}
		}
	}

	int eval(vb &assignment) {
		int result = 0;
		for (int i = 0; i< L(clauses); i++) {
			if (clauses[i].bEval(assignment))
				result++;
		}
		return result;
	}

	int eval(vb &assignment, vi& status) {
		int result = 0;
		for (int i = 0; i< L(clauses); i++) {
			status[i] = clauses[i].iEval(assignment);
			if (status[i])
				result++;
		}
		return result;
	}

	void addClause(Clause &c) {
		clauses.push_back(c);
		for (int i = 0; i< L(c.literals); i++) {
			vars[c.literals[i].first].push_back({ L(clauses) - 1,c.literals[i].second });
		}
	}

	void print() {
		int n = L(clauses);
		for (int i = 0; i<n; i++) {
			clauses[i].print();
			if (i<n - 1)cout << " & ";
		}
		cout << endl;
	}

	int getSize() {
		return L(clauses);
	};
};

class SatState {
public:
	Cnf* cnf;
	vb assignment;
	/*vi clause_stats;*/
	bool* cu_assignment;
	int score;
	int size;
	int sizelog;

	SatState(Cnf* c, int nbvars) {
		assignment.resize(nbvars);
		//clause_stats.resize(c.getSize());
		cnf = c;
		size = nbvars;
		sizelog = (int)(log2(size));
		cudable();
		randAssign();
	}

	void cudable() {
		cudaMallocManaged(&cu_assignment, L(assignment)*sizeof(bool));
		cnf->cudable();
	}

	void randAssign() {
		srand(time(NULL));
		for (int i = 0; i < L(assignment); i++) {
			assignment[i] = (rand() % 2 == 0);
		}
		//also may set clause_stats
		//score = cnf.eval(assignment, clause_stats);
		score = cnf->eval(assignment);
		for (int i = 0; i < assignment.size(); i++) {
			cu_assignment[i] = assignment[i];
		}
	}

	void printAssignment() {
		cout << "Assignment: ";
		for (int i = 1; i < L(assignment); i++) {
			cout << i << ":" << assignment[i] << ", ";
		}
		cout << endl;
	}

	void printCnf() {
		cnf->print();
	}

	int eval() {
		return cnf->eval(assignment);
	}

	void toggle(int i) {
		assignment[i] = !assignment[i];
		cu_assignment[i] = !cu_assignment[i];
	}

	int toggleAndScore(int i) {
		int prev = 0;
		int res = 0;
		/*		for(auto& var : cnf.vars[i])
		{
		num = cnf.clauses[var.first].iEval(assignment);
		if (num) {
		if (num == 1 && !(l ^ var.second))
		res--;
		}else {
		res++;
		}
		}*/
		//should be vector of toggles?
		for (auto&& var : cnf->vars[i])
		{
			if (cnf->clauses[var.first].bEval(assignment)) prev++;
		}
		toggle(i);
		for (auto&& var : cnf->vars[i])
		{
			if (cnf->clauses[var.first].bEval(assignment)) res++;
		}
		return res - prev;
	}
};

class SatSolver {

protected: SatState *state;

public:
	SatSolver(){}
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
		int p = 70;
		int temp = 10000;
		int counter = 0;
		srand(time(NULL));
		//int go = 100;
		while (1) {
			counter++;
			if (temp < 19900)
				temp += 2;
			//cout << temp;
			if (counter > 6000 && temp == 19900)
			{
				counter = 0;
				temp = 19400;
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

			/*	else {
			cout << "reset on: " << current << endl;
			if (best < current) {
			cout << "upgrade! :" << current << endl;
			best = current;
			}
			state->randAssign();
			current = state->score;
			} */
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

__global__ void maxKernel(int* results,int* maxindex, int n, int log)
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

	for (int i = 0; i < (n / bd)+1; i++) {
		if (i*bd + threadIdx.x < n) {
			share_results[i*bd + threadIdx.x] = results[i*bd + threadIdx.x];
			share_indexes[i*bd + threadIdx.x] = i*bd + threadIdx.x;
		}
	}
	__syncthreads();
	for (int i = log ; i > 0; i--) {
		for (int j = 0; j*bd <= offset ; j++)
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
	/*	if (share_results[0] >= 0) {
			state->cu_assignment[share_indexes[0]] = !(state->cu_assignment[share_indexes[0]]);
			state->score = state->score + share_results[0];
			printf("Found %d \n", state->score);
		}else
		printf("Local minimum");*/
	}
}
__global__ void oneStepKernel(SatState* state,int *results,int parent) {
	int prev = 0;
	int res = 0;
	int n = state->cnf->cu_vars_size[threadIdx.x];
	for (int i=0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[threadIdx.x][i]].bToggleEval(state->cu_assignment,-1,parent)) prev++;
	}
	for (int i = 0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[threadIdx.x][i]].bToggleEval(state->cu_assignment, threadIdx.x,parent)) res++;
	}
	results[threadIdx.x] = res -prev;
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
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[index][i]].bToggleEval(state->cu_assignment, -1,-1)) prev++;
	}
	for (int i = 0; i<n; i++)
	{
		if (state->cnf->cu_clauses[state->cnf->cu_vars_cind[index][i]].bToggleEval(state->cu_assignment, index,-1)) parent++;
	}

	oneStepKernel << < 1, size >> >(state, results[index], index);
	cudaDeviceSynchronize();
	if (index != 0) {
		maxKernel << <1, 500, (2 * size * sizeof(int)) >> > (results[index], &results_index[index], size, sizelog);
		auto err = cudaGetLastError();
		cudaDeviceSynchronize();
		err = cudaGetLastError();
		res = results[index][0];
	}
	results[0][index] = parent + res - prev;
}

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
							cout << "reset on: " << best << " temps: " <<main_temperature<< " - "<< deep_temperature << endl;
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


__global__ void randIntitKernel(curandState* state, int size) {
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	if (ind < size)
		curand_init(123, ind, 0, &state[ind]);
}

__global__ void 

class CudaSASatSolver : public SatSolver {
public:
	int** results;
	int* results_index;
	curandState *d_state;
	int parallel;
	CudaSASatSolver(int nbvars, Cnf* c, int pl) : SatSolver() {
		parallel = pl;
		cudaMallocManaged(&d_state,parallel*sizeof(curandState));
		randIntitKernel << <(parallel/32)+1,32 >> > (d_state,parallel);
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
		//int temperature = 10000;

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
				int eval_test = state->eval();
				current = state->score;
				//if (current != eval_test)
				//cout << "failed score!" << endl;
				//cout << current << "-" << state->score << endl;
			}
			else {
				if (!deep) {
					/*if (rand() % 10000 < temperature) {
						state->score += state->toggleAndScore(rand() % nbvar);
						current = state->score;
						temperature -= 1;
					}
					else*/
						deep = true;
					//lastdeep = current;
				}
				else {
					/*if (rand() % 10000 < temperature) {
						deep = false;
						state->toggle(maxindd);
						state->toggle(maxind);
						state->score += maxeval;
						temperature -= 1;
					}
					else {*/
						//if (prob > prob_threshold)
						cout << "reset on: " << current << endl;
						if (best < current) {
							cout << "upgrade! :" << current << endl;
							best = current;
						}
						state->randAssign();
						current = state->score;
						deep = false;
						//temperature = 10000;
					//}
				}
			}
		}
	}

};

void maxFan() {
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
		if (a[0] != 10 )
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

int main()
{
	//maxFan();
	//classInCudaTest();
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

	SatSolver *solver = new CudaDeepSatSolver(nbvars, cnf);
	
	solver->solve();
	cin.get();
	cudaDeviceReset();
    return 0;
}


