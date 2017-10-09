# Simulation for inverted pendulum
# Author: Yu Okamoto

import math
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as spl
import cvxpy
import time

#number of horizon
Nm = 30
Np = 3

if Nm < Np:
  print "Nm should be longer than Np"

dt = 0.1
dt_mpc = 0.1
simtime = 10.0

L = 0.77   # distance from base to c.o.m [m]
g = 9.81  # gravitational acceleration [m/s^2]

def mpc(A,B,Q,R,x0):

    xm = cvxpy.Variable(A.shape[1], Nm+1)
    um = cvxpy.Variable(B.shape[1], Np+1)

    cost = 0.0
    constr = []
    for t in range(Nm):
        if t > Np:
          u =  um[:,-1]
        else :
          u = um[:,t]

        # u = u + np.array([[0.0],[0.0],[0.0],[-g]])
        cost += cvxpy.quad_form(xm[:, t + 1], Q)
        cost += cvxpy.quad_form(u, R)
        constr += [xm[:, t + 1] == xm[:,t]+ (A * xm[:, t] + B * u)*dt_mpc ]
        constr += [xm[8,t+1]+1>=0.2] # Z
 
    constr += [xm[:, 0] == x0]
    constr += [cvxpy.abs(xm[0,:])<=40*np.pi/180.0] # r
    constr += [cvxpy.abs(xm[2,:])<=40*np.pi/180.0] # s
    constr += [cvxpy.abs(xm[10,:])<=70*np.pi/180.0] # gamma
    constr += [cvxpy.abs(xm[11,:])<=70*np.pi/180.0] # beta
    
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

    start = time.time()
    prob.solve(verbose=False)
    elapsed_time = time.time() - start
    # print("calc time:{0} [s]".format(elapsed_time)), prob.status

    if prob.status == cvxpy.OPTIMAL:
		# print um.value[:,:]
		# print (np.array(um.value[:,0]).flatten()), um.value[:,0]
		return (np.array(um.value[:,0]).flatten())
    else :
      return False

def draw_quad(x,z,beta):
	lc = L*np.cos(beta)
	ls = L*np.sin(beta)
	tx = [ x, x+0.5*lc ]
	ty = [ z, z-0.5*ls ]
	tx = np.vstack([tx, tx[-1] + 0.25*ls ])
	ty = np.vstack([ty, ty[-1] + 0.25*lc ])
	tx = np.vstack([tx, tx[-1] + 0.125*lc ])
	ty = np.vstack([ty, ty[-1] - 0.125*ls ])
	tx = np.vstack([tx, tx[-2] - 0.125*lc ])
	ty = np.vstack([ty, ty[-2] + 0.125*ls ])
	#back to center
	tx = np.vstack([tx, tx[-3] ])
	ty = np.vstack([ty, ty[-3] ])
	tx = np.vstack([tx, tx[-5] ])
	ty = np.vstack([ty, ty[-5] ])

	#left
	tx = np.vstack([tx, tx[0] - 0.5*lc ])
	ty = np.vstack([ty, ty[0] + 0.5*ls ])
	tx = np.vstack([tx, tx[-1] + 0.25*ls ])
	ty = np.vstack([ty, ty[-1] + 0.25*lc ])
	tx = np.vstack([tx, tx[-1] - 0.125*lc ])
	ty = np.vstack([ty, ty[-1] + 0.125*ls ])
	tx = np.vstack([tx, tx[-2] + 0.125*lc ])
	ty = np.vstack([ty, ty[-2] - 0.125*ls ])
	#back to center
	tx = np.vstack([tx, tx[-3] ])
	ty = np.vstack([ty, ty[-3] ])
	tx = np.vstack([tx, tx[-5] ])
	ty = np.vstack([ty, ty[-5] ])

	return tx, ty

def draw_pole(x,z,theta):
	tx = [x, x+2.0*L*np.sin(theta) ]
	ty = [z, z+2.0*L*np.cos(theta) ]

	return tx, ty


# Rotation array (matrix)
def R3x(rad):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(rad)[0], -np.sin(rad)[0]],
                     [0.0, np.sin(rad)[0], np.cos(rad)[0]]])


def R3y(rad):
    return np.array([[np.cos(rad)[0], 0.0, np.sin(rad)[0]],
                     [0.0, 1.0, 0.0],
                     [-np.sin(rad)[0], 0.0, np.cos(rad)[0]]])


def R3z(rad):
    return np.array([[np.cos(rad)[0], -np.sin(rad)[0], 0.0],
                     [np.sin(rad)[0], np.cos(rad)[0], 0.0],
                     [0.0, 0.0, 1.0]])


# parameters for simulation
L = 0.77   # distance from base to c.o.m [m]

A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # r
              [g / L, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0],  # r dot
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s
              [0, 0, g / L, 0, 0, 0, 0, 0, 0, 0, g, 0, 0],  # s dot
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # x
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g, 0],  # x dot
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # y
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],  # y dot
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # z
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # z dot
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # gamma
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # beta
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # alpha

B = np.array([[0, 0, 0, 0],   # u = [Wx,
              [0, 0, 0, 0],  # Wy,
              [0, 0, 0, 0],  # Wz,
              [0, 0, 0, 0],  # a]
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])

C = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])


# initial state
x = np.array([[20*np.pi/180.0],  # r
              [0.0],  # r dot
              [0.0],  # s
              [0.0],  # s dot
              [0.0],  # x
              [0.0],  # x dot
              [0.0],  # y
              [0.0],  # y dot
              [1.0],  # z
              [0.0],  # z dot
              [0.0],  # gamma
              [0.0],  # beta
              [0.0]])  # alpha

# reference
y_ref = np.array([[-0.5],    # x [m]
                  [0.0],    # y [m]
                  [1.25]])   # z [m]


# extend system for servo
As = np.hstack([np.vstack([A, -C]),
                np.vstack([np.zeros([A.shape[0], C.shape[0]]), np.zeros([C.shape[0], C.shape[0]])])])

Bs = np.vstack([B, np.zeros([C.shape[0], B.shape[1]])])

Cs = np.hstack([C, np.zeros([y_ref.shape[0], C.shape[0]])])

Is = np.vstack([np.zeros([A.shape[0], y_ref.shape[0]]), np.identity(y_ref.shape[0])])

xs = np.vstack([x, np.zeros([y_ref.shape[0], 1])])

xs_init = xs

dxs = np.zeros([As.shape[0], 1])

# LQR
Q = np.diag([1.0,   # r
             1.0,   # r_dot
             1.0,   # s
             1.0,   # s_dot
             1.2,   # x
             0.5,   # x_dot
             0.8,   # y
             0.8,   # y_dot
             1.2,   # z
             0.5,   # z_dot
             1.0,   # gamma
             1.0,   # beta
             0.01,   # alpha
             0.001,   # delta x
             0.001,   # delta y
             0.001    # delta z
             ]) 

R = np.diag([1.0,
             1.0,
             1.0,
             0.001])

# variables for simulation and plot
t = np.arange(0, simtime + dt / 100, dt)
xp = np.zeros([int(simtime / dt) + 1, As.shape[0]])
up = np.zeros([int(simtime / dt) + 1, 4])
dxp = np.zeros([int(simtime / dt) + 1, As.shape[0]])
# zp = np.zeros([int(simtime / dt) + 1, 1])

# Store initial xs and dxs (t = 0)
xp[0, :] = np.transpose(xs)
dxp[0, :] = np.transpose(dxs)
up[0, :] = 0
# zp[0] = math.sqrt(math.pow(L, 2) - math.pow(xs[0], 2) - math.pow(xs[2], 2))

y = Cs.dot(xs)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-4.0, 4.0))
ax.grid()
line_quad_ref, = ax.plot([], [], 'b:', lw=0.5)
line_pole_ref, = ax.plot([], [], 'r:', lw=2)
line_quad, = ax.plot([], [], 'b-', lw=1)
line_pole, = ax.plot([], [], 'ro-', lw=3)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
legend = ax.legend([r'$\rm{quad}_{\rm{ref}}$',r'$\rm{pole}_{\rm{ref}}$','quad','pole'], loc='lower right')

def init():  
  line_quad.set_data([], [])
  line_pole.set_data([], [])
  line_quad_ref.set_data([], [])
  line_pole_ref.set_data([], [])
  time_text.set_text('')
  global xs
  xs = xs_init
  return line_quad, line_pole, time_text

plotted = False
# simulation loop
def sim_loop(i):
	global xs, dxs, y, xp, dxp, zp, up, plotted
	xse = xs.copy()
	xse[4] -= y_ref[0]
	xse[6] -= y_ref[1]
	xse[8] -= y_ref[2]
	u = mpc(As,Bs,Q,R,xse)
	# if u==False:
	# 	print 'mpc diverged'
	# 	# u = u_prev
	# 	return False
	# u_prev = u
	
	# calc input from lqr gain
	# u = -K.dot(xs)  # [Wx, Wy, Wz, a]
	# print u
	# u[3] += g


	# creates variables for the xs(t-1) so it is easier to code
	r = xs[0]
	r_dot = xs[1]
	s = xs[2]
	s_dot = xs[3]
	x = xs[4]
	x_dot = xs[5]
	y = xs[6]
	y_dot = xs[7]
	z = xs[8]
	z_dot = xs[9]
	gamma = xs[10]
	beta = xs[11]
	alpha = xs[12]

	# create variables for dxs
	r_dd = dxs[1]  # r double dot
	s_dd = dxs[3]  # s double dot

	# create variables for inputs
	Wx = u[0]
	Wy = u[1]
	Wz = u[2]

	# gravity copensation
	Rov = (R3z(alpha).dot(R3y(beta))).dot(R3x(gamma))  # rotation matrix from o to v
	g_acc = np.array([[0],
	                  [0],
	                  [-g]])

	Rinv = np.linalg.inv(Rov)
	u[3] += np.linalg.norm(Rinv.dot(-g_acc))
	a = u[3]
	
	# update state equation

	## x_dd, y_dd, z_dd
	acc = np.array([[0],
	                [0],
	                [a]])

	trans_dd = Rov.dot(acc) + g_acc
	# print Wy, r*180.0/np.pi, r_dot*180.0/np.pi, x_dot, trans_dd[0]
	# print z_dot
	
	# create variables for translational double dot so it is easier to code
	x_dd = trans_dd[0]  # x double dot
	y_dd = trans_dd[1]  # y double dot
	z_dd = trans_dd[2]  # z double dot

	omega_vector = np.array([Wx,
	                         Wy,
	                         Wz])
	Rot = np.array([[1, np.sin(beta) * np.sin(gamma) / np.cos(beta), np.sin(beta) * np.cos(gamma) / np.cos(beta)],
	                [0, np.cos(gamma), -np.sin(gamma)],
	                [0, np.sin(gamma) / np.cos(beta), np.cos(gamma) / np.cos(beta)]])
	angle_dot = Rot.dot(omega_vector)

	# Create temporary buffer to store the nonlinear A.dot(xs) part
	dxs_temp = np.zeros([As.shape[0], 1])
	# dxs_temp = np.zeros([As.shape[0],1])

	# print math.pow(L, 2) - math.pow(r, 2) - math.pow(s, 2)
	zeta = math.sqrt(math.pow(L, 2) - math.pow(r, 2) - math.pow(s, 2))

	dxs_temp[0] = r_dot          # r_dot
	dxs_temp[2] = s_dot          # s_dot
	dxs_temp[4] = x_dot          # x_dot
	dxs_temp[6] = y_dot          # y_dot
	dxs_temp[8] = z_dot          # z_dot
	dxs_temp[10] = angle_dot[0]  # gamma_dot
	dxs_temp[11] = angle_dot[1]  # beta_dot
	dxs_temp[12] = angle_dot[2]  # alpha_dot
	dxs_temp[5] = x_dd    # x_dd
	dxs_temp[7] = y_dd    # y_dd
	dxs_temp[9] = z_dd    # z_dd

	# r_dd
	dxs_temp[1] = 1 / ((L * L - s * s) * zeta * zeta) * (-math.pow(r, 4) * x_dd
	                                                     - math.pow((L * L - s * s), 2) * x_dd
	                                                     - 2 * r * r *
	                                                     (s * r_dot * s_dot + (-L * L + s * s) * x_dd)
	                                                     + math.pow(r, 3) * (s_dot * s_dot +
	                                                                         s * s_dd - zeta * (g + z_dd))
	                                                     + r * (-L * L * s * s_dd
	                                                            + math.pow(s, 3) * s_dd
	                                                            + s * s *
	                                                            (r_dot * r_dot - zeta * (g + z_dd))
	                                                            + L * L * (-r_dot * r_dot - s_dot * s_dot + zeta * (g + z_dd))))

	# s_dd
	dxs_temp[3] = 1 / ((L * L - r * r) * zeta * zeta) * (-math.pow(s, 4) * y_dd
	                                                     - math.pow((L * L - r * r), 2) * y_dd
	                                                     - 2 * s * s *
	                                                     (r * r_dot * s_dot + (-L * L + r * r) * y_dd)
	                                                     + math.pow(s, 3) * (r_dot * r_dot +
	                                                                         r * r_dd - zeta * (g + z_dd))
	                                                     + s * (-L * L * r * r_dd
	                                                            + math.pow(r, 3) * r_dd
	                                                            + r * r *
	                                                            (s_dot * s_dot - zeta * (g + z_dd))
	                                                            + L * L * (-r_dot * r_dot - s_dot * s_dot + zeta * (g + z_dd))))

	# Create vector for error states
	error = - np.vstack([ np.zeros([ A.shape[0],1 ]), Cs.dot(xs) ]) + Is.dot(y_ref)
	# print -error 

	# Update
	dxs = dxs_temp + error
	xs = xs + dxs * dt
	y = Cs.dot(xs)

	# record for plot
	xp[i] = np.transpose(xs)
	dxp[i] = np.transpose(dxs)
	up[i] = np.transpose(u)
	# zp[i] = zeta

	# draw animation
	#right
	
	tx, ty = draw_quad(x, z, beta)
	line_quad.set_data(tx, ty)
	tx, ty = draw_pole(x, z, r)
	line_pole.set_data(tx, ty)
	tx, ty = draw_quad(y_ref[0], y_ref[2], 0)
	line_quad_ref.set_data(tx, ty)
	tx, ty = draw_pole(y_ref[0], y_ref[2], 0)
	line_pole_ref.set_data(tx, ty)
		

	time_text.set_text(time_template % (i*dt))
	
	if i==len(xp[:,0])-1 and plotted != True:
		plt.figure()
		# plt.rcParams["font.family"] = "Times New Roman"
		plt.subplot(3, 1, 1)
		plt.plot(t, xp[:, 4])
		# plt.plot(t, xp[:, 6])
		plt.plot(t, xp[:, 8])
		plt.plot([t[0],t[-1]], [y_ref[0], y_ref[0]], 'r:')
		plt.plot([t[0],t[-1]], [y_ref[2], y_ref[2]], 'g:')
		plt.legend([r'$x$',r'$z$',r'$x_{\rm{ref}}$',r'$z_{\rm{ref}}$'], loc='upper right')
		plt.ylabel(r'$x \rm{[m]}$')

		plt.subplot(3, 1, 2)
		plt.plot(t, xp[:, 0]*180/np.pi)
		plt.plot([t[0],t[-1]], [y_ref[1], y_ref[1]], 'r:')
		plt.legend([r'$\theta$',r'$\theta_{\rm{ref}}$'], loc='upper right')
		plt.ylabel(r'$\theta$ [deg]')
		
		plt.subplot(3, 1, 3)
		plt.plot(t, up)
		plt.ylabel(r'$f \rm{[N]}$')
		plt.xlabel(r'$t \rm{[s]}$')

		plotted = True
		plt.show(block=False)

	return line_quad_ref, line_pole_ref,line_quad, line_pole, time_text, legend
	
ani = animation.FuncAnimation(fig, sim_loop, frames=np.arange(0, len(xp[:,0])),
                              interval=1, blit=True, init_func=init)
# ani.save("pb_mpc.gif", writer = 'imagemagick')

plt.show()

