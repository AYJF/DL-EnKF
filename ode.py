#############################################################################################
#                                                                                           #
#             Modules for time integration of three versions of Lorenz 96 model             #
#                             using 4th-order Runge-Kutta scheme                            #
#                                                                                           #
#                             written by Tadashi Tsuyuki (MRI, Japan) on February 4, 2022   #
#############################################################################################


import numpy as np


#######################
#   Lorenz 96 model   #
#######################

def func (t, x, n, F):

    dxdt = np.zeros(n)

    dxdt[0] = (x[1]-x[n-2])*x[n-1] - x[0] + F
    dxdt[1] = (x[2]-x[n-1])*x[0  ] - x[1] + F
    for i in range(2, n-1):
        dxdt[i] = (x[i+1]-x[i-2])*x[i-1] - x[i] + F
    dxdt[n-1] = (x[0]-x[n-3])*x[n-2] - x[n-1] + F
    
    return dxdt

def runge (x0, n, dt, nstep, F):
    
    k1 = np.zeros(n)  
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    x  = np.zeros(n)
    X  = np.zeros((nstep+1, n))

    istep = 0
    X[istep, :] = x0[:]

    x[:] = x0[:]
    for istep in range(nstep):
        t  = istep*dt       
        k1 = dt*func(t,          x,          n, F)
        k2 = dt*func(t + dt/2.0, x + k1/2.0, n, F)
        k3 = dt*func(t + dt/2.0, x + k2/2.0, n, F)
        k4 = dt*func(t + dt,     x + k3,     n, F)
        
        istep += 1
        x += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0    
        X[istep, :] = x[:]

    return X


#################################
#   Two-scale Lorenz 96 model   #
#################################

def func0 (t, x, nl, ns, F, h, c, b):
    
    n = nl + nl*ns
    dxdt = np.zeros(n)

    # Large-scale
    dxdt[0] = (x[1]-x[nl-2])*x[nl-1] - x[0] + F - h*c/b*sum(x[nl   :nl+  ns])
    dxdt[1] = (x[2]-x[nl-1])*x[0   ] - x[1] + F - h*c/b*sum(x[nl+ns:nl+2*ns])
    for i in range(2, nl-1):
        dxdt[i] = (x[i+1]-x[i-2])*x[i-1] - x[i] + F - h*c/b*sum(x[nl+i*ns:nl+(i+1)*ns])
    dxdt[nl-1] = (x[0]-x[nl-3])*x[nl-2] - x[nl-1] + F - h*c/b*sum(x[nl+(nl-1)*ns:n])

    # Small-scale
    dxdt[nl ] = c*b*(x[n-1]-x[nl+2])*x[nl+1] - c*x[nl ] + h*c/b*x[0]
    for i in range(nl+1, n-2):
        dxdt[i] = c*b*(x[i-1]-x[i+2])*x[i+1] - c*x[i  ] + h*c/b*x[int((i-nl)/ns)]
    dxdt[n-2] = c*b*(x[n-3]-x[nl  ])*x[n-1 ] - c*x[n-2] + h*c/b*x[nl-1]
    dxdt[n-1] = c*b*(x[n-2]-x[nl+1])*x[nl  ] - c*x[n-1] + h*c/b*x[nl-1]
                                                                  
    return dxdt

def runge0 (x0, nl, ns, dt, nstep, F, h, c, b):

    n = nl + nl*ns    
    
    k1 = np.zeros(n)  
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    x  = np.zeros(n)
    X  = np.zeros((nstep+1, n))

    istep = 0
    X[istep, :] = x0[:]

    x[:] = x0[:]
    for istep in range(nstep):
        t  = istep*dt       
        k1 = dt*func0(t,          x,          nl, ns, F, h, c, b)
        k2 = dt*func0(t + dt/2.0, x + k1/2.0, nl, ns, F, h, c, b)
        k3 = dt*func0(t + dt/2.0, x + k2/2.0, nl, ns, F, h, c, b)
        k4 = dt*func0(t + dt,     x + k3,     nl, ns, F, h, c, b)
        
        istep += 1
        x += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0    
        X[istep, :] = x[:]

    return X


######################################
#   Parameterized Lorenz 96 model   #
######################################

def func1 (t, x, n, F, p0, p1):

    # Parameterization of forcing by linear function
    G = np.zeros(n)
    G[:] = p0*x[:] + p1

    # Large-scale 
    dxdt = np.zeros(n)
    dxdt[0] = (x[1]-x[n-2])*x[n-1] - x[0] + F + G[0]
    dxdt[1] = (x[2]-x[n-1])*x[0  ] - x[1] + F + G[1]
    for i in range(2, n-1):
        dxdt[i] = (x[i+1]-x[i-2])*x[i-1] - x[i] + F + G[i]
    dxdt[n-1] = (x[0]-x[n-3])*x[n-2] - x[n-1] + F + G[n-1]
    
    return dxdt

def runge1 (x0, n, dt, nstep, F, p0, p1):
    
    k1 = np.zeros(n)  
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)
    x  = np.zeros(n)
    X  = np.zeros((nstep+1, n))

    istep = 0
    X[istep, :] = x0[:]

    x[:] = x0[:]
    for istep in range(nstep):
        t  = istep*dt       
        k1 = dt*func1(t,          x,          n, F, p0, p1)
        k2 = dt*func1(t + dt/2.0, x + k1/2.0, n, F, p0, p1)
        k3 = dt*func1(t + dt/2.0, x + k2/2.0, n, F, p0, p1)
        k4 = dt*func1(t + dt,     x + k3,     n, F, p0, p1)
        
        istep += 1
        x += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0    
        X[istep, :] = x[:]

    return X