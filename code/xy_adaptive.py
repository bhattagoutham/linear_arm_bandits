from termcolor import colored
import numpy as np
import time
import sys

dominant_arms = {0:np.array([[1],[0]]), 1:np.array([[0],[1]]), 2:np.array([[1],[1]])}
dominant_arm = {0:np.array([1,0]), 1:np.array([0,1]), 2:np.array([1,1])}
directions = {}
real_theta = np.array([[0.5], [0.5]])

def pull_arm(arm):
	sigma = 0.01
	mu = 0
	eps = np.random.normal(mu, sigma, 1)
	r = np.sum(np.multiply(dominant_arms[arm], real_theta))
	return (r+eps)


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

def update_directions_set(arms):
	global directions
	
	for arm in arms:
		del dominant_arms[arm]
		if arm in directions:
			del directions[arm]
	

def rho_prev(prev_phase_design_mat):
	iter = 0; max_val = 0
	for arm in directions:
		for y in directions[arm]:
			y = y.reshape(y.shape[0], 1)
			temp = np.matmul(np.matmul(np.transpose(y), np.linalg.inv(prev_phase_design_mat)), y)
			if iter == 0:
				max_val = temp
			elif max_val < temp:
				max_val = temp
			iter = iter + 1
	return max_val

def rho_curr(A):	
	iter = 0; max_val = 0
	for arm in directions:
		for y in directions[arm]:
			y = y.reshape(y.shape[0], 1)
			temp = np.sqrt(np.matmul(np.matmul(np.transpose(y), np.linalg.inv(A)), y))
			if iter == 0:
				max_val = temp
			elif max_val < temp:
				max_val = temp
			iter = iter + 1
	return max_val[0][0]

def compute_design_matrix(nt_arm_pull_vec, n, d):
	design_mat = np.identity(d)
	for arm in dominant_arms:
		if arm in nt_arm_pull_vec:
			design_mat = design_mat + (nt_arm_pull_vec[arm]/n)*(np.multiply(dominant_arms[arm], np.transpose(dominant_arms[arm])))
	return design_mat

def minmax(A):

	min_dict = {}
	for arm in dominant_arms:
		max_arr = np.array([])
		for dir_arm in directions:
			for y in directions[dir_arm]:
				y = y.reshape(y.shape[0], 1)
				A_t = (A.astype(int) + np.multiply(dominant_arms[arm], np.transpose(dominant_arms[arm])))
				val = np.dot(np.dot(np.transpose(y), np.linalg.inv(A_t)), y)
				# print('val..',val, y, A_t)
				max_arr = np.append(max_arr, val[0][0])
		min_dict[arm] = max(max_arr)
		# print(max_arr)

	# print(min_dict)
	min_arm = min(min_dict, key= min_dict.get)
	# print('min_arm:', min_arm)
	return min_arm



def xy_adaptive():

	phase_num = 1
	prev_rho = 1
	delta_conf = 0.05
	alpha = 0.1
	d = dominant_arms[0].shape[0]
	prev_phase_length = (d*(d+1)) + 1
	design_mat = np.identity(d)
	k = len(dominant_arms)

	while( len(dominant_arms) > 1 ):

		curr_rho = prev_rho
		A = np.identity(d); nt_arm_pull_vec = {}
		curr_phase_length = 1;
		b = np.zeros((d,1))

		if phase_num == 1:
			temp = (alpha * prev_rho)/prev_phase_length
		else:
			temp = (alpha * rho_prev(design_mat))/prev_phase_length

		while(curr_rho/curr_phase_length >= temp):

			x_t = minmax(A)
			# print('arm_pulled:',x_t); #time.sleep(0.5)
			if x_t in nt_arm_pull_vec:
				nt_arm_pull_vec[x_t] = nt_arm_pull_vec[x_t] + 1
			else:
				nt_arm_pull_vec[x_t] = 1

			r = pull_arm(x_t)
			b = b + (r*dominant_arms[x_t])
			A = A + np.matmul(dominant_arms[x_t], np.transpose(dominant_arms[x_t]))
			curr_phase_length = curr_phase_length + 1
			curr_rho = rho_curr(A)
			# if curr_phase_length % 100 == 0:
				# print('curr_phase_length: ',curr_phase_length)

		prev_phase_length = curr_phase_length
		design_mat = compute_design_matrix(nt_arm_pull_vec, curr_phase_length, d)
		theta_hat = np.dot(np.linalg.inv(A), b)
		remove_dominated_arms(A, theta_hat, k, delta_conf)
		phase_num = phase_num + 1
		# print colored('hello', 'red'),
		print colored("\n*****************Phase No. : " + str(phase_num) + "***********************", "red")
		print "Current phase length : " + colored(str(curr_phase_length), "blue")
		print "Theta hat[Predicted Theta] : "
		print colored(str(theta_hat.reshape(theta_hat.shape[0],)), "blue")
		print "No. of dominant arms : "+ colored(str(len(dominant_arms)), "blue")
		print "The dominant arms : "
		for i in dominant_arms:
			print str(i) +"\t"


		# # i = 0
		# for i in range(len(dominant_arms)):
		# 	print colored(str(dominant_arms[i]), "blue")
		# 	# i+=1
		# print colored(str((dominant_arms)), "blue")
	

def remove_dominated_arms(A, theta, k, delta_conf):
	
	global dominant_arms

	remove_arms = np.array([])
	temp1 = np.sqrt(np.log(pow(k,2)/delta_conf)) # here log is to the base n. n possibly being phase length
	
	for x1 in dominant_arms :
		for x2 in dominant_arms :
			if x1 != x2 :
				diff = dominant_arms[x1] - dominant_arms[x2]
				temp2 = np.sqrt(np.matmul(np.matmul(np.transpose(diff), np.linalg.inv(A)), diff))
				temp3 = np.sum(np.multiply(-diff, theta))
				if temp1*temp2 <= temp3:
					remove_arms = np.append(remove_arms, x1)
					break

	if len(remove_arms) != 0:
		update_directions_set(remove_arms)



def initialize(n_arms, n_dim):
	global dominant_arms, dominant_arm, real_theta
	print("Initializing...")
	for i in range(n_arms):
		dominant_arms[i] = np.random.randint(n_dim, size=(n_dim,1))
		dominant_arm[i] = np.random.randint(n_dim, size=(n_dim,))

	real_theta = 0.5*np.ones((n_dim,1))




# print(sys.argv)
if len(sys.argv) == 3:
	n_arms = int(sys.argv[1])
	n_dim = int(sys.argv[2])
	initialize(n_arms, n_dim)

d = dominant_arms[0].shape[0]
initial_phase_length = (d*(d+1)) + 1
print colored("\n*****************Phase No. : 1***********************", "red")
print "Current phase length : " + colored(str(initial_phase_length), "blue")
print "Theta hat[Initial Theta] : "
print colored("[0.0 ...]", "blue")
print "No. of dominant arms : "+ colored(str(len(dominant_arms)), "blue")
print "The dominant arms : "
for i in dominant_arms:
	print str(i) +"\t",

compute_directions()
xy_adaptive()
print "Original Theta : "
print colored(str(real_theta.reshape(real_theta.shape[0],)), "blue")
print "Arms:", dominant_arm



