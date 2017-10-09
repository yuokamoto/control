# Simulation for inverted pendulum
# Author: Yu Okamoto

import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as spl
import cvxpy
import time

#number of horizon
Nm = 20
Np = 5

if Nm < Np:
  print "Nm should be longer than Np"

# parameters for simulation
dt = 0.1
simtime = 10.0

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
        cost += cvxpy.quad_form(xm[:, t + 1], Q)
        cost += cvxpy.quad_form(u, R)
        constr += [xm[:, t + 1] == xm[:,t]+ (A * xm[:, t] + B * u)*dt]

    constr += [xm[:, 0] == x0]
    constr += [cvxpy.abs(um)<=100]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

    start = time.time()
    prob.solve(verbose=False)
    elapsed_time = time.time() - start
    # print("calc time:{0} [s]".format(elapsed_time))

    if prob.status == cvxpy.OPTIMAL:
      return (np.array(um.value[0, :]).flatten())[0]
    else :
      return False


# initial state
x = np.array([[0.0],  # x
              [0.0],  # x dot
              [np.deg2rad(30)],  # theta
              [0.0]])  # theta dot

# reference
y_ref = np.array([[-3.0],    # x
                  [0.0]])   # theta

# equation
L = 1.0   # dis0.0, 1.0, 0.0, 0.0 base to CoG [m]
g = 9.81  # gra0.0, 1.0, 0.0, 0.0 acceleration [m/s^2]
m = 1.0 # bar mass[kg]
M = 2.0 # bass mass[kg]

A = np.array([[0.0, 1.0,  0.0, 0.0],   # x
              [0.0, 0.0, -m/M*g, 0.0],   # x dot
              [0.0, 0.0, 0.0, 1.0],   # theta
              [0.0, 0.0, (M+m)*g/(M*L), 0.0]])  # theta dot

B = np.array([[0.0      ],
              [1.0/M    ],
              [0.0      ],
              [-1.0/(M*L)]])

C = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]])

# extend system for servo
As = np.hstack([np.vstack([A, -C]),
                np.vstack([np.zeros([A.shape[0], C.shape[0]]), np.zeros([C.shape[0], C.shape[0]])])])

Bs = np.vstack([B, np.zeros([C.shape[0], B.shape[1]])])

Cs = np.hstack([C, np.zeros([y_ref.shape[0], C.shape[0]])])

Is = np.vstack([np.zeros([A.shape[0], y_ref.shape[0]]), np.identity(y_ref.shape[0])])

xs = np.vstack([x, np.zeros([y_ref.shape[0], 1])])

xs_init = xs

# LQR
Q = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([0.1])

# variables for simulation and plot
t = np.arange(0, simtime + dt / 100, dt)
xp = np.zeros([int(simtime / dt) + 1, As.shape[0]])
up = np.zeros([int(simtime / dt) + 1, 1])
y = Cs.dot(xs)

#init
xp[0, :] = np.transpose(xs)
up[0, :] = 0

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
  # calc input from mpc
  # print 'sim time ', t[i], '[s]'
  xse = xs.copy()
  xse[0] -= y_ref[0]
  u = mpc(As,Bs,Q,R,xse)
  if u==False:
    print 'mpc diverged'
    # u = u_prev
    return False
  u_prev = u

  # update state 

  # dxs = As.dot(xs) + Bs.dot(u) + Is.dot(y_ref)

  # human friendly rename
  r = xs[0,0]
  r_dot = xs[1,0]
  theta = xs[2,0]
  theta_dot = xs[3,0]

  i0 = u + m*L*theta_dot**2*np.sin(theta)
  i1 = m*g*L*np.sin(theta)
  det = M*m*L**2 + m**2*L**2*np.sin(theta)

  dxs = np.zeros([As.shape[0],1])
  dxs[0] = xs[1]
  dxs[2] = xs[3]
  dxs[1] =  (m*L**2*i0 -m*L*np.cos(theta)*i1)/det
  dxs[3] = (-m*L*np.cos(theta)*i0 + (M+m)*i1)/det
  dxs[4] = y_ref[0] - y[0]
  dxs[5] = y_ref[1] - y[1]

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
    plt.plot([t[0],t[-1]], [y_ref[1], y_ref[1]], 'r:')
    plt.legend([r'$\theta$',r'$\theta_{\rm{ref}}$'], loc='upper right')
    plt.ylabel(r'$\theta$ [deg]')
    
    plt.subplot(3, 1, 3)
    plt.plot(t, up)
    plt.ylabel(r'$f \rm{[N]}$')
    plt.xlabel(r'$t \rm{[s]}$')

    plotted = True
    plt.show(block=False)

  return line, time_text
  
ani = animation.FuncAnimation(fig, sim_loop, frames=np.arange(0, len(xp[:,0])),
                              interval=1, blit=True, init_func=init)
plt.show()