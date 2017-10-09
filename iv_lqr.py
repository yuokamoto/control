# Simulation for inverted pendulum
# Author: Yu Okamoto

import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as spl

# LQR function
def lqr(A, B, Q, R):

    # solve Riccati equation
    X = spl.solve_continuous_are(A, B, Q, R)

# compute the LQR gain
    K = spl.inv(R).dot(B.T).dot(X)
#eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    print "lqr gain ", K
    return K


# parameters for simulation
dt = 0.1
simtime = 12.0

# initial state
x = np.array([[0.0],  # x
              [0.0],  # x dot
              [np.deg2rad(45)],  # theta
              [0.0]])  # theta dot

# reference
y_ref = np.array([-3.0])    # x
y_size = 1
                  
# equation
L = 1.0   # dis0.0, 1.0, 0.0, 0.0 base to CoG [m]
g = 9.81  # gra0.0, 1.0, 0.0, 0.0 acceleration [m/s^2]
m = 0.5 # bar mass[kg]
M = 2.0 # bass mass[kg]

A = np.array([[0.0, 1.0,  0.0, 0.0],   # x
              [0.0, 0.0, -m/M*g, 0.0],   # x dot
              [0.0, 0.0, 0.0, 1.0],   # theta
              [0.0, 0.0, (M+m)*g/(M*L), 0.0]])  # theta dot

B = np.array([[0.0      ],
              [1.0/M    ],
              [0.0      ],
              [-1.0/(M*L)]])

C = np.array([[1.0, 0.0, 0.0, 0.0]])

# extend system for servo
As = np.hstack([np.vstack([A, -C]),
                np.vstack([np.zeros([A.shape[0], C.shape[0]]), np.zeros([C.shape[0], C.shape[0]])])])

Bs = np.vstack([B, np.zeros([C.shape[0], B.shape[1]])])

Cs = np.hstack([C, np.zeros([y_size, C.shape[0]])])

Is = np.vstack([np.zeros([A.shape[0], y_size]), np.identity(y_size)])

xs = np.vstack([x, np.zeros([y_size, 1])])

xs_init = xs

# LQR
Q = np.diag([1, 0.1, 1, 1, 0.1])
R = np.diag([1])
K = lqr(As, Bs, Q, R)

# variables for simulation and plot
t = np.arange(0, simtime + dt / 100, dt)
xp = np.zeros([int(simtime / dt) + 1, As.shape[0]])
up = np.zeros([int(simtime / dt) + 1, 1])
y = Cs.dot(xs)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4.0, 4.0))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():  
  line.set_data([], [])
  time_text.set_text('')
  global xs
  xs = xs_init
  return line, time_text

plotted = False
# simulation loop
def sim_loop(i):
	global xs, y, xp, up, plotted

	# calc input from lqr gain
	xse = xs.copy()
	xse[0] -= y_ref[0]
	u = -K.dot(xse)
	
	# update state 

	# dxs = As.dot(xs) + Bs.dot(u) + Is.dot(y_ref)

	# human friendly rename
	r = xs[0]
	r_dot = xs[1]
	theta = xs[2]
	theta_dot = xs[3]

	i0 = u + m*L*theta_dot**2*np.sin(theta)
	i1 = m*g*L*np.sin(theta)
	det = M*m*L**2 + m**2*L**2*np.sin(theta)

	dxs = np.zeros([As.shape[0],1])
	dxs[0] = xs[1]
	dxs[2] = xs[3]
	dxs[1] =  (m*L**2*i0 -m*L*np.cos(theta)*i1)/det
	dxs[3] = (-m*L*np.cos(theta)*i0 + (M+m)*i1)/det
	dxs[4] = y_ref[0] - y[0]
  
	xs = xs + dxs * dt
	y = Cs.dot(xs)

	# record for plot
	xp[i] = np.transpose(xs)
	up[i] = np.transpose(u)
	
	thisx = [xs[0], xs[0]+L*np.sin(xs[2])]
	thisy = [0,       L*np.cos(xs[2])]

	line.set_data(thisx, thisy)
	time_text.set_text(time_template % (i*dt))
	
	if i==len(xp[:,0])-1 and plotted != True:
		plt.figure()
		# plt.rcParams["font.family"] = "Times New Roman"
		plt.subplot(3, 1, 1)
		plt.plot(t, xp[:, 0])
		plt.plot([t[0],t[-1]], [y_ref[0], y_ref[0]], 'r:')
		plt.legend([r'$x$',r'$x_{\rm{ref}}$'])
		plt.ylabel(r'$x \rm{[m]}$')

		plt.subplot(3, 1, 2)
		plt.plot(t, xp[:, 2]*180/np.pi)
		plt.legend([r'$\theta$'], loc='upper right')
		plt.ylabel(r'$\theta$ [deg]')
		
		plt.subplot(3, 1, 3)
		plt.plot(t, up)
		plt.ylabel(r'$f \rm{[N]}$')
		plt.xlabel(r'$t \rm{[s]}$')

		plotted = True
		plt.show(block=False)

	return line, time_text
	
ani = animation.FuncAnimation(fig, sim_loop, frames=np.arange(0, len(xp[:,0])),
                              interval=5, blit=True, init_func=init)
plt.show()
