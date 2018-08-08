
#ifndef MAXSAT_STRUCTURES_H
#define MAXSAT_STRUCTURES_H


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

class Clause {
public:
	vib literals;
	int* cu_vars;
	bool* cu_signs;
	int size;

	void cudable() {
		size = literals.size();
		cudaMallocManaged(&cu_vars, size * sizeof(int));
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

	__device__ bool bToggleEval(bool* assignment, int toggle_index, int parent_index) {
		for (int i = 0; i < size; i++)
			if (!(assignment[cu_vars[i]] ^ (cu_signs[i] ^ (cu_vars[i] == toggle_index || cu_vars[i] == parent_index))))
				return true;
		return false;
	}

	__device__ int iToggleEval(bool* assignment, int toggle_index, int parent_index) {
		for (int i = 0; i < size; i++)
			if (!(assignment[cu_vars[i]] ^ (cu_signs[i] ^ (cu_vars[i] == toggle_index || cu_vars[i] == parent_index))))
				return 1;
		return 0;
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
		cudaMallocManaged(&cu_assignment, L(assignment) * sizeof(bool));
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

#endif