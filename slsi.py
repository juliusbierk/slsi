# Julius Bier Kirkegaard 2017
# juliusbierk@gmail.com

import numpy as np
from scipy import linalg, sparse
import scipy.sparse.linalg as sparse_linalg

def preprocess(system, rhs):
	neqs = len(system)
	nvars = len(system[0])

	var_sizes = [-1]*nvars
	for i in range(neqs):
		for j in range(nvars):
			var_sizes[j] = var_sizes[j] if system[i][j] is None else system[i][j].shape[1]
	rhs_sizes = map(len, rhs)

	### Test sizes of matrices
	dtype = np.float64
	assert len(system) == len(rhs), 'Unequal number equations and right-hand-side vectors'
	for i in range(neqs):
		assert nvars == len(system[i]), 'All equations must be the same length'
		for j in range(nvars):
			if system[i][j] is not None:
				if system[i][j].dtype == np.complex64 or system[i][j].dtype == np.complex128:
					dtype = np.complex128
				assert system[i][j].shape[0] == rhs_sizes[i], 'Matrices must be consistent'
				assert system[i][j].shape[1] == var_sizes[j], 'Matrices must be consistent'
	for j in range(nvars):
		if rhs[j].dtype == np.complex64 or rhs[j].dtype == np.complex128:
			dtype = np.complex128

	return neqs, nvars, var_sizes, rhs_sizes, dtype

def postprocess(res, var_sizes):
	extra_output = False
	if isinstance(res, tuple):
		output = list(res)
		res = res[0]
		extra_output = True

	res = np.split(res, np.cumsum(var_sizes))[:-1]

	if extra_output:
		output[0] = res
		return tuple(output)
	else:
		return res

def solve_system(system, rhs, method=linalg.solve, **kwargs):
	neqs, nvars, var_sizes, rhs_sizes, dtype = preprocess(system, rhs)

	### Create full matrix
	A = np.zeros((sum(rhs_sizes),sum(var_sizes)), dtype=dtype)
	assert A.shape[0] == A.shape[1], 'Number of equations and variables must equal'

	r = 0
	for i in range(neqs):
		c = 0
		for j in range(nvars):
			if system[i][j] is not None:
				A[r:r+rhs_sizes[i], c:c+var_sizes[j]] = system[i][j]
			c += var_sizes[j]
		r += rhs_sizes[i]
	b = np.hstack(rhs)

	### Solve
	res = method(A, b, **kwargs)
	return postprocess(res, var_sizes)


def solve_sparse_system(system, rhs, method=sparse_linalg.spsolve, **kwargs):
	neqs, nvars, var_sizes, rhs_sizes, dtype = preprocess(system, rhs)

	### Create full matrix
	A = sparse.lil_matrix((sum(rhs_sizes),sum(var_sizes)), dtype=dtype)
	assert A.shape[0] == A.shape[1], 'Number of equations and variables must equal'
	r = 0
	for i in range(neqs):
		c = 0
		for j in range(nvars):
			if system[i][j] is not None:
				A[r:r+rhs_sizes[i], c:c+var_sizes[j]] = system[i][j]
			c += var_sizes[j]
		r += rhs_sizes[i]
	b = np.hstack(rhs)

	### Solve
	A = sparse.csr_matrix(A)
	res = method(A, b, **kwargs)
	return postprocess(res, var_sizes)

if __name__ == '__main__':
	### Solve system
	# A1 x + B1 y = a1 
	# A2 x + B2 y = a2

	a1 = np.array([1, 2, 3])
	a2 = np.array([7, 8, 9, 10])
	A1 = np.array([[2, 0],
				   [0, 1],
				   [2, 1]])
	B1 = np.array([[2, 5, 0, 1, 2],
				   [0, 2, 0, 0, 1],
				   [0, 1, 0, 2, 0]])
	A2 = np.array([[3, 0],
				   [1, 2],
				   [0, 2+1j],
				   [3, 3]])
	B2 = np.array([[5, 0, 2, 2, 5],
				   [0, 1, 2, 0, 0],
				   [0, 0, 2, 1, 0],
				   [8, 0, 0, 0, 8]])

	print 'Test dense solver'
	x, y = solve_system(((A1, B1), (A2, B2)), (a1, a2))
	print np.allclose(np.dot(A1, x) + np.dot(B1, y), a1) and \
		  np.allclose(np.dot(A2, x) + np.dot(B2, y), a2)

	print 'Test sparse solver'
	A2 = sparse.lil_matrix(A2)
	B1 = sparse.csc_matrix(B1)
	x, y = solve_sparse_system(((A1, B1), (A2, B2)), (a1, a2))
	A2 = A2.todense()
	B1 = B1.todense()
	print np.allclose(np.dot(A1, x) + np.dot(B1, y), a1) and \
		  np.allclose(np.dot(A2, x) + np.dot(B2, y), a2)

	print 'Test missing array'
	x, y = solve_system(((A1, B1), (None, B2)), (a1, a2))
	A2 = np.array([[0, 0],
				   [0, 0],
				   [0, 0],
				   [0, 0]])
	print np.allclose(np.dot(A1, x) + np.dot(B1, y), a1) and \
		  np.allclose(np.dot(A2, x) + np.dot(B2, y), a2)
