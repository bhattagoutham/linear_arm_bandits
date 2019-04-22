import numpy as np
import time
import math
import sys

dominant_arm = {0:np.array([0,1]), 1:np.array([1,1]), 2:np.array([1,0])}
directions = {}
real_theta = np.array([[0.5], [0.5]])

def compute_directions():
	
	global directions
	n = len(dominant_arm)

	for arm_1 in range(n):
		for arm_2 in range(arm_1+1, n):

			# print(arm_1, arm_2)
			temp = dominant_arm[arm_1] - dominant_arm[arm_2]
			
			if arm_1 in directions:
				directions[arm_1] = np.vstack((directions[arm_1], temp))
			else:
				directions[arm_1] = np.array([temp])

			if arm_2 in directions:
				directions[arm_2] = np.vstack((directions[arm_2], -temp))
			else:
				directions[arm_2] = np.array([-temp])

def pull_arm(arm):
	sigma = 1
	mu = 0
	eps = np.random.normal(mu, sigma, 1)
	r = np.dot(dominant_arm[arm], real_theta)
	return (r+eps)

def minmax(A):

	min_dict = {}

	for idx1, arm1 in dominant_arm.items():
		max_arr = np.array([])
		for idx2, arm2 in dominant_arm.items():

				arm1 = arm1.reshape(arm1.shape[0], 1)
				arm2 = arm2.reshape(arm2.shape[0], 1)

				A_temp = (A + np.multiply(arm1, np.transpose(arm1)))
				val = np.sqrt(np.dot(np.dot(np.transpose(arm2), np.linalg.inv(A_temp)), arm2))
				max_arr = np.append(max_arr, val[0][0])
		
		# print(max_arr, max(max_arr))
		min_dict[idx1] = max(max_arr)

	min_arm = min(min_dict, key= min_dict.get)
	return min_arm


def check_fr_best_arm(A, theta, k, n, delta_conf):
	
	global dominant_arm
	c = pow(k,2)/delta_conf

	temp1 = 0.5*np.sqrt(math.log(c, n)) 
	
	for x1 in dominant_arm :
		p = 0;
		for x2 in dominant_arm:
			if x1 != x2 :
				diff = dominant_arm[x1] - dominant_arm[x2]
				temp2 = np.sqrt(np.matmul(np.matmul(np.transpose(diff), np.linalg.inv(A)), diff))
				temp3 = np.sum(np.multiply(diff, theta))
				if temp1*temp2 <= temp3:
					p = p+1

		if p == len(dominant_arm)-1:
			return True

	return False

def g_static(n_dim):

	A = np.identity(n_dim)
	b = np.zeros((n_dim,))
	delta_conf = 0.05
	k = len(dominant_arm)
	it = 1
	while True:
		x_t = minmax(A)
		arm = dominant_arm[x_t]
		arm_t = arm.reshape(n_dim,1)
		A = A + np.multiply(arm_t, np.transpose(arm_t))
		r = pull_arm(x_t)
		b = b + (r*arm)
		theta_hat = np.dot(np.linalg.inv(A), b)
		it = it + 1
		if check_fr_best_arm(A,theta_hat,k,it,delta_conf):
			break
		elif it%100 == 0:
			print('Iterations:',it)
			print('theta:',theta_hat)

	print("Arms:", dominant_arm)
	print('predicted_theta:',theta_hat)
	print('original_theta:',real_theta.reshape(real_theta.shape[0], ))
	max_rew = 0; b_arm = 0
	for idx, arm in dominant_arm.items():
		temp = np.sum(np.multiply(arm.reshape(arm.shape[0],1),real_theta))
		if temp > max_rew:
			max_rew = temp
			b_arm = idx
	print('best_arm:', b_arm)
	print('Iterations required: ', it)


def initialize(n_arms, n_dim):
	global dominant_arm, real_theta
	print("Initializing...")
	for i in range(n_arms):
		dominant_arm[i] = np.random.randint(n_dim, size=(n_dim,))

	real_theta = 0.5*np.ones((n_dim,1))

n_arms = int(sys.argv[1])
n_dim = int(sys.argv[2])

if len(sys.argv) == 3:
	initialize(n_arms, n_dim)

compute_directions()
g_static(n_dim)