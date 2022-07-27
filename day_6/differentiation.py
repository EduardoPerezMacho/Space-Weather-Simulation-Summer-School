"""
Created on Wed Jul 20 15:14:41 2022

@author: Simone Servadio
"""

"""This file teaches first order numerical differentiation using 
finite difference methods"""

"Auxiliarly functions"

import sys
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """generic function"""
    return np.cos(x)+x*np.sin(x)

def func_dot(x):
    """Derivative of the generic function"""
    return x*np.cos(x)



"Dispaly function and its derivative"
n_points = 1000     # number of points
x_in = -6           # start 
x_fin = -x_in       # symmetric domain
x = np.linspace(x_in,x_fin,n_points) # independent variable
y = func(x) #dependent variable
y_dot = func_dot(x) # derivative

fig1 = plt.figure()
plt.plot(x,y,'-r')
plt.plot(x,y_dot,'b-')
plt.grid()
plt.xlabel('x',fontsize = 16)
plt.legend([r'$y$',r'$\dot y$'],fontsize=16)
#sys.exit() #exit form the script










"FINITE DIFFERENCE"
step_size = 0.25  #define interval step for differentiation

fig2 = plt.figure() #plot the correct solution 
plt.plot(x,y_dot,'-k')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot y$')
plt.legend([r'$\dot y$ truth'])

"Forward Finite Difference"
x0 = x_in                      # initialize first point
y_dot_forw = np.array([])      # initialize solution array 
x_forw = np.array([x_in])      # initialize step points

while x0 <= x_fin:
    current_value = func(x0)                              #f_k
    following_value = func(x0+step_size)                  #f_k+1
    slope = (following_value-current_value)/step_size     #(f_k+1 - f_k)/h
    x0 = x0 + step_size           
    x_forw = np.append(x_forw, x0)
    y_dot_forw = np.append(y_dot_forw, slope)
    
    
plt.plot(x_forw[:-1],y_dot_forw,'-r')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward'])
#sys.exit()









"Backward Finite Difference"
x0 = x_in                      # initialize first point
y_dot_back = np.array([])      # initialize solution array 
x_back = np.array([x_in])      # initialize step points

while x0 <= x_fin:
    current_value = func(x0)                              #f_k
    previous_value = func(x0-step_size)                   #f_k-1
    slope = (current_value-previous_value)/step_size      #(f_k - f_k-1)/h
    x0 = x0 + step_size
    x_back = np.append(x_back, x0)
    y_dot_back = np.append(y_dot_back, slope)
    
    
plt.plot(x_back[:-1],y_dot_back,'-b')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward',r'$\dot y$ backward'])
#sys.exit()









"Central Finite Difference"
x0 = x_in                      # initialize first point
y_dot_cent= np.array([])       # initialize solution array 
x_cent = np.array([x_in])      # initialize step points

while x0 <= x_fin:
    following_value = func(x0+step_size)                  #f_k+1
    previous_value = func(x0-step_size)                   #f_k-1
    slope = (following_value-previous_value)/step_size/2  #(f_k+1 - f_k-1)/2h
    x0 = x0 + step_size
    x_cent = np.append(x_cent, x0)
    y_dot_cent = np.append(y_dot_cent, slope)
    
    
plt.plot(x_cent[:-1],y_dot_cent,'-g')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward',
            r'$\dot y$ backward',r'$\dot y$ central'])
sys.exit()











































=======
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:46:54 2022

@author: edu_p
"""
# Solution from Simoni
import numpy as np
import matplotlib.pyplot as plt


def function1(x):
    "Return the result of a generic function"""
    return np.cos(x)+x*np.sin(x)

def function2(x):
    """Derivative of a generic function"""
    return x*np.cos(x)

n_points=1000       # number of function
x_in = -6           # start
x_fin = -x_in       # end (symmetric domain)
x = np.linspace(x_in, x_fin, n_points) # independent variable
y= function1(x) 
y_dot = function2(x)

fig1=plt.figure()
plt.plot(x,y,'-r')
plt.plot(x,y_dot,'b')
plt.grid()
plt.xlabel('x',fontsize=16)


#%%
print(list(range(-6,7,1)))

#%%
# My Solution

def function1(x):
    "Return the result of a generic function"""
    return np.cos(x)+x*np.sin(x)

def function2(x):
    """Derivative of a generic function"""
    return x*np.cos(x)

x_in=-6
x_end=-x_in
x=np.linspace(x_in,x_end,1000)
y1=function1(x)
y2=function2(x)
plt.plot(x,y1)
plt.plot(x,y2)

#%%
# Evaluation Forwards

x_in=-6 # initial x
x_end=-x_in # end x (symmetric domain)
h=0.25 # finite difference
fSize=14 # fontsizes
x=[]
new_function=[]
while x_in<=x_end:
    new_function.append((function1(x_in+h)-function1(x_in))/h) # evaluation
    x.append(x_in)
    x_in+=h

plt.plot(x,new_function,'r',label='evaluation')
plt.plot(x,function2(x),'b',label='derivative')
plt.xlabel('x',fontsize=fSize)
plt.legend(fontsize=fSize)
plt.grid()

#%%
# Evaluation Backwards

x_in=-6 # initial x
x_end=-x_in # end x (symmetric domain)
h=0.25 # finite difference
fSize=14 # fontsizes
x=[]
new_function=[]
while x_in<=x_end:
    new_function.append((function1(x_in)-function1(x_in-h))/h) # evaluation
    x.append(x_in)
    x_in+=h

plt.plot(x,new_function,'r',label='evaluation')
plt.plot(x,function2(x),'b',label='derivative')
plt.xlabel('x',fontsize=fSize)
plt.legend(fontsize=fSize)
plt.grid()

#%%
# Evaluation Forwards and Backwards

x_in=-6 # initial x
x_end=-x_in # end x (symmetric domain)
h=0.25 # finite difference
fSize=14 # fontsizes
x=[]
new_function1=[]
new_function2=[]
while x_in<=x_end:
    new_function1.append((function1(x_in+h)-function1(x_in))/h) # forwards
    new_function2.append((function1(x_in)-function1(x_in-h))/h) # backwards
    x.append(x_in)
    x_in+=h

fig1=plt.figure()
plt.plot(x,new_function1,'r--',label='forwards')
plt.plot(x,new_function2,'g--',label='bakwards')
plt.plot(x,function2(x),'b-',label='derivative')
plt.legend(fontsize=fSize)
plt.xlabel('x',fontsize=fSize)
plt.grid()

#%%
# Evaluation Central

x_in=-6 # initial x
x_end=-x_in # end x (symmetric domain)
h=0.25 # finite difference
fSize=14 # fontsizes
x=[]
new_function3=[]
while x_in<=x_end:
    new_function3.append((function1(x_in+h)-function1(x_in-h))/(2*h)) # central
    x.append(x_in)
    x_in+=h

fig1=plt.figure()
plt.plot(x,new_function3,'r--',label='central')
plt.plot(x,function2(x),'b-',label='derivative', alpha=0.5)
plt.legend(fontsize=fSize)
plt.xlabel('x',fontsize=fSize)
plt.ylabel('y',fontsize=fSize)
plt.grid()

#%%
# Integration - Euler Method or RK1

from scipy.integrate import odeint
def RHS(x, t):
    """ ODE Right hand side"""
    return -2*x

# Set the problem
y0=2 # initial condition
t0=0 #initial time
tf=2 # final time

# Evaluate exact solution
time=np.linspace(t0,tf) # time spaned
y_true=odeint(RHS,y0,time) # solution

fig1=plt.figure()
plt.plot(time,y_true,'k-',linewidth=2)
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend(['Truth'])
plt.grid()

#%%
# Plot Exact Solution (real) and Euler Methods (RK1 and RK2)

# First Order Runge-Kutta (RK1)
step_size = 0.2
y0 = 3
current_time = t0
current_value = y0
fSize = 14 # fontsizes
time_1 = np.array([t0])
new_function4 = np.array([y0])
while current_time < tf-step_size:
    current_value = current_value + step_size*(-2*current_value)
    new_function4 = np.append(new_function4, current_value)
    current_time += step_size
    time_1 = np.append(time_1, current_time)

# Second Order Runge-Kutta (RK2)
current_time = t0
current_value = y0
time_2 = np.array([t0])
new_function5 = np.array([y0])
while current_time < tf-step_size:
    current_value = current_value + step_size*(-2*(current_value+step_size/2*(-2*current_value)))
    new_function5 = np.append(new_function5, current_value)
    current_time += step_size
    time_2 = np.append(time_2, current_time)


# Exact solution (real)
time_real=np.linspace(t0,tf) # time spaned
y_true=odeint(RHS,y0,time_real) # solution

fig1=plt.figure()
plt.plot(time_1, new_function4, 'r-', linewidth=2, label='Runge-Kutta 1', alpha=0.7)
plt.plot(time_2, new_function5, 'g-', linewidth=2, label='Runge-Kutta 2', alpha=0.7)
plt.plot(time_real, y_true, 'b--', linewidth=2, label='real1', alpha=0.9)
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.grid()

#%%
# Plot Exact Solution (real) and Euler Methods (RK1, RK2, RK3, and RK4)  ***NOT FINISHED***

# All Orders Runge-Kutta (RK1)
step_size = 0.2
y0 = 3
current_time = t0
current_value = y0
fSize = 14 # fontsizes
time = np.array([t0])
RK1 = np.array([y0])
RK2 = np.array([y0])
RK3 = np.array([y0])
RK4 = np.array([y0])

while current_time < tf-step_size:
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*step_size/2, current_time + step_size/2)
    k3 = RHS(current_value + k2*step_size/2, current_time + step_size/2)
    k4 = RHS(current_value + k3*step_size/2, current_time + step_size/2)
    next_value = current_value + (k1+2*k2+2*k3_k4)*step_size/6
    next_time = current_time _step_size
    timeline

# Exact solution (real)
time_real=np.linspace(t0,tf) # time spaned
y_true=odeint(RHS,y0,time_real) # solution

fig1=plt.figure()
plt.plot(time, RK1, 'r-', linewidth=2, label='Runge-Kutta 1', alpha=0.7)
plt.plot(time, RK2, 'g-', linewidth=2, label='Runge-Kutta 2', alpha=0.7)
plt.plot(time_real, y_true, 'b--', linewidth=2, label='real1', alpha=0.9)
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.grid()

#%%
# Nonlinear pendulun

def pendulum_free(x, t):
    """ Dynamics of the pendulum without any constraint """
    g = 9.81 # gravity constant
    l = 3 #lenght of pendulum
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = -g/l*np.sin(x[0])
    return x_dot

def pendulum_damped(x, t):
    """ Dynamics of the pendulum with a damper """
    g = 9.81 # gravity constant
    l = 3 #lenght of pendulum
    damp = 0.3 #damper coefficient
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = -g/l*np.sin(x[0]) - damp*x[1]
    return x_dot

x0 = np.array([np.pi/3, 0])
t0 = 0.0
tf = 15.0
n_points=1000
time = np.linspace(t0, tf, n_points)
#x1 = teta
#x2 = teta_dot

solution1 = odeint(pendulum_free, x0, time)
solution2 = odeint(pendulum_damped, x0, time)

fig2=plt.figure()
plt.subplot(2,1,1)
plt.plot(time, solution1[:,0])
plt.plot(time, solution2[:,0])
plt.subplot(2,1,2)
plt.plot(time, solution1[:,1])
plt.plot(time, solution2[:,1])

#%%
# Lorentz63 system

from mpl_toolkits.mplot3d import Axes3D

def Lorentz63(x, t, sigma, ro, beta):
    """ Lorentz63 """
    x_dot = np.zeros(3)
    x_dot[0] = sigma*(x[1]-x[0])
    x_dot[1] = x[0]*(ro-x[2])-x[1]
    x_dot[2] = x[0]*x[1] - beta*x[2]
    return x_dot


x0 = np.array([5, 5, 5])
t0 = 0.0
tf = 20.0
n_points=1000
time = np.linspace(t0, tf, n_points)
sigma = 10
ro = 28
beta = 8/3
y = 5
z = 5

y = solution=odeint(Lorentz63, x0, time, args=(sigma, ro, beta))

fig3 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(y[:,0], y[:,1], y[:,2])

for condition in range(20):
    x0[0] = np.random.choice(range(-20,20))
    x0[1] = np.random.choice(range(-30,30))
    x0[2] = np.random.choice(range(0,50))
    y = solution=odeint(Lorentz63, x0, time, args=(sigma, ro, beta))
    ax.plot3D(y[:,0], y[:,1], y[:,2])

