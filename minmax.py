# Min-max step of the algorithm

import numpy as np
arms  = np.random.randint(5, size=(3, 3))
print('arms:\n', arms)
A  = np.random.randint(5, size=(3, 3))
print('A:\n', A)
print('f = x\' A x')

def minmax(x, A):

	n_arms, dim = x.shape
	min_temp = np.zeros(n_arms)
	
	for i in range(n_arms):
		max_temp = np.zeros(n_arms-1)
		for j in range(1,n_arms):
			if(i+j>= n_arms):
				k = (i+j) % n_arms
			else :
				k = i+j
			max_temp[j-1] = f(x[:,i], x[:,k], A)
		
		min_temp[i] = np.max(max_temp)
	
	print('optimal arm :', np.argmin(min_temp))

def f(x1, x2, A):
	d, = x1.shape
	x1 = x1.reshape(d,1)
	x2 = x2.reshape(d,1)
	t = np.matmul( np.matmul(np.transpose(x2), (A + np.matmul(x1, np.transpose(x1)))), x2)
	return t[0,0]

minmax(arms, A)
