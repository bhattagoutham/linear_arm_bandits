import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

def plot_cones(arm_x, arm_y):

	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	
	ax.plot(arm_x[0], arm_y[0], 'ro', label='arm_0')
	ax.plot(arm_x[1], arm_y[1], 'go', label='arm_1')
	ax.plot(arm_x[2], arm_y[2], 'bo', label='arm_2')
	ax.legend()

	return fig

def animate(fig, x, y, w, h):

	ax = fig.axes[0]
	if len(ax.patches) != 0:
		rect = ax.patches[0]
		rect.remove()
	
	rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='black',facecolor='none', label='conf. set')
	ax.add_patch(rect)
	rect.set_zorder(3)
	

def get_cones():

	# initalize the arms
	arm = np.array([[1,0],[0,1],[1,1]])

	# set the display domain range
	x_r = 50; y_r = 50

	arm_x = {0:np.array([]),1:np.array([]),2:np.array([])}
	arm_y = {0:np.array([]),1:np.array([]),2:np.array([])}

	# caluclate the best arm at each point
	for x in range(-x_r, x_r+1):
		for y in range(-y_r, y_r+1):

			temp = np.array([])
			theta = np.array([x,y])
			
			for i in range(3):
				temp = np.append(temp, np.dot(arm[i], theta))
			
			bst_arm = np.argmax(temp, axis=0)
			arm_x[bst_arm] = np.append(arm_x[bst_arm], x)
			arm_y[bst_arm] = np.append(arm_y[bst_arm], y)
			# print('(', x,y,')',':',bst_arm, end='...')
		# print()

	# print(arm_x)
	return arm_x, arm_y




x, y = get_cones()
f = plot_cones(x,y)
for i in range(10):
	print(i)
	time.sleep(2)
	animate(f,i-5, 0-i, 15-i, 20-(2*i) )
	f.show(); plt.pause(1)