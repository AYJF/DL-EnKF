#############################################################################################
#                                                                                           #
#                      Modules for Serial EnSRF and stochastic EnKF                         #
#               with covariance localization ond adaptive covariance inflation              #
#                                                                                           #
#                             written by Tadashi Tsuyuki (MRI, Japan) on February 4, 2022   #
#############################################################################################


import numpy as np


#############################################################
#   Analysis ensemble by serial EnSRF for Lorenz 96 model   #
#############################################################

def EnSRF (x_f_ens, x_f_mean, y_o, H, n, m, N, sigma_o, obs_dist, r_local):

    h       = np.zeros(n)
    x_f     = np.zeros(n)
    work    = np.zeros(n)
    e_f     = np.zeros(N)
    x_a     = np.zeros(n)
    E_f     = np.zeros((n, N))
    x_a_ens = np.zeros((n, N))

    x_f[:] = x_f_mean[:]
    for k in range(N):
        E_f[:, k] = (x_f_ens[:, k] - x_f[:])/np.sqrt(N-1)
    
    for j in range(m):
     
        # Observation operator
        h[:] = H[j, :] 

        # Location of observation
        j_loc = obs_dist*np.where(h != 0)[0][0]

        # Kalman gain
        for k in range(N):
            work[:] = E_f[:, k]
            e_f[k] = np.dot(h, work)
        alpha = np.dot(e_f, e_f) + sigma_o**2
        K = E_f@e_f/alpha

        # Covariance localization
        for i in range(n):
            cor = localize(i, j_loc, n, r_local)
            K[i] = cor*K[i]       
        
        # Analysis
        d = y_o[j] - np.dot(h, x_f)
        x_a[:] = x_f[:] + K[:]*d

        # Analysis ensemble
        K1  = K/(1.0 + sigma_o/np.sqrt(alpha))
        E_a = (np.eye(n) -np.outer(K1, h))@E_f
        
        x_f = x_a
        E_f = E_a

    for k in range(N):
        x_a_ens[:, k] = x_a[:] + np.sqrt(N-1)*E_a[:, k]


    return x_a_ens


#######################################################################
#   Analysis ensemble by serial EnSRF for two-scale Lorenz 96 model   #
#######################################################################

def EnSRF_2scale (x_f_ens, x_f_mean, y_o, H, nl, ns, m, N, sigma_o, obs_dist, r_local, icor_ls):

    n = nl + nl*ns

    h       = np.zeros(n)
    x_f     = np.zeros(n)
    work    = np.zeros(n)
    e_f     = np.zeros(N)
    x_a     = np.zeros(n)
    E_f     = np.zeros((n, N))
    x_a_ens = np.zeros((n, N))

    x_f[:] = x_f_mean[:]
    for k in range(N):
        E_f[:, k] = (x_f_ens[:, k] - x_f[:])/np.sqrt(N-1)
    
    for j in range(m):
     
        # Observation operar
        h[:] = H[j, :] 

        # Location of observation
        j_loc = obs_dist*np.where(h != 0)[0][0]

        # Kalman gain
        for k in range(N):
            work[:] = E_f[:, k]
            e_f[k] = np.dot(h, work)
        alpha = np.dot(e_f, e_f) + sigma_o**2
        K = E_f@e_f/alpha

        # Covariance localization
        for i in range(nl):
            cor = localize(i, j_loc, nl, r_local)
            K[i] = cor*K[i]
        if icor_ls == 1:
            K[         nl:nl+j*ns] = 0.0
            K[nl+(j+1)*ns:n      ] = 0.0
        else:           
            K[nl:] = 0.0
        
        # Analysis
        d = y_o[j] - np.dot(h, x_f)
        x_a[:] = x_f[:] + K[:]*d

        # Analysis ensemble
        K1  = K/(1.0 + sigma_o/np.sqrt(alpha))
        E_a = (np.eye(n) -np.outer(K1, h))@E_f
        
        x_f = x_a
        E_f = E_a

    for k in range(N):
        x_a_ens[:, k] = x_a[:] + np.sqrt(N-1)*E_a[:, k]


    return x_a_ens


################################################################
#   Analysis ensemble by stochastic EnKF for Lorenz 96 model   #
################################################################

def EnKF (x_f_ens, x_f_mean, y_o, H, n, m, N, sigma_o, r_local):

    y_o_member = np.zeros(m)
    x_f_member = np.zeros(n)
    x_a_member = np.zeros(n)
    x_a_ens    = np.zeros((n, N))
    K          = np.zeros((n, m))
    
    # Foracast covariance
    P_f      = np.cov(x_f_ens)

    # Covariance localization
    for i in range(n):
        for j in range(n):
            cor = localize(i, j, n, r_local)
            P_f[i, j] = cor*P_f[i, j]

    # Kalman gain
    K = P_f@(H.T)@np.linalg.inv((sigma_o**2)*np.eye(m) + H@P_f@(H.T))

    # Analysis ensemble
    for k in range(N):
        y_o_member[0:m] = y_o[0:m] + sigma_o*np.random.randn(m)
        x_f_member[0:n] = x_f_ens[0:n, k]
        x_a_member = x_f_member + K@(y_o_member - H@x_f_member)
        x_a_ens[0:n, k] = x_a_member[0:n]
        
    return x_a_ens


###############################
#   Covariance localization   #
###############################

def localize (i, j, n, r_local):

    if r_local < 0:
        
        cor = 1.0

    else:

        n_local = 2*r_local
        n_dist  = i - j

        if (i > n_local and i < n - n_local) or (j > n_local and j < n - n_local):
            r = abs(n_dist)
        else:
            r = n
            for k in range(-1, 2):
                r = min(abs(n_dist + k*n), r)
                
        if r_local > 0:
            x = float(r)/float(r_local)
        else:
            x = 0.0

        if   r <= r_local:
            cor = -1.0/4.0*x**5 + 1.0/2.0*x**4 + 5.0/8.0*x**3 - 5.0/3.0*x**2 + 1.0
        elif r <= n_local:
            cor = 1.0/12.0*x**5 - 1.0/2.0*x**4 + 5.0/8.0*x**3 + 5.0/3.0*x**2 - 5.0*x + 4.0 - 2.0/3.0/x
        else:
            cor = 0.0

    return cor


#####################################
#   Adaptive covariance inflation   #
#####################################

def inflation_rev (x_f_mean, P_f, y_o, H, sigma_o, n, m, rho_min, rho_max, kappa, rho, ratio):
    
    d = y_o - H@x_f_mean    
    work = (np.dot(d, d) - float(m)*sigma_o**2)/np.trace(H@P_f@(H.T))
    if work < rho_min:
        work = rho_min
    elif work > rho_max:
        work = rho_max

    rho   = (rho + ratio*work)/(1.0 + ratio)
    ratio = kappa*ratio/(1.0 + ratio)

    return rho, ratio