import numpy as np
from ode import runge
from analysis import EnKF, EnSRF, inflation_rev
from miscellaneous import extract
import matplotlib.pyplot as plt 
import csv


def main():
    
    # Parameter setting
    F = 8.0         # Forcing paramter
    n = 40          # No of state variables
    m = 40          # No of observations
    N = 10          # Ensemble size
    dt = 0.01       # Time step of integration
    t_end = 2050.0  # Length of time integration
    obs_int = 0.50  # Time interval between observations
    sigma_o = 1.0   # Standard deviation of observation error
    sigma_f = 2.0   # Standard deviation of forecast error at initial time

    r_local = 4     # Covariance localization radius
    rho_min = 0.9   # Lower bound for adaptive covariance inflation
    rho_max = 5.0   # Upper bound for adaptive covariance inflation

    r_extr  = 10    # maximum input radius of datasets

    nstep    = int(t_end/dt + 0.5)       # No of time steps of integration
    obs_step = int(obs_int/dt + 0.5)     # No of time steps between observations
    n_DA     = int(nstep/obs_step + 0.5) # No of executions of data assimilation after initial time
    out_int  = int(1.0/obs_int + 0.5)    # Output interval (unit: time interval betwen observations)
    n_out    = int(n_DA/out_int + 0.5)   # No of file outputs after initial time
    obs_dist = int(n/m + 0.5)            # grid interval of observations
    n_extr   = 2*r_extr + 1              # Size of each input to DNN  
    
    print('nstep =', nstep, '     obs_step =', obs_step, '     n_DA =', n_DA, '     out_int =', out_int, \
      '     n_out =', n_out)
    print('obs_dist =', obs_dist, '     n_extr =', n_extr,     '     n_data =', 3*n_extr)

    # Initialization of arrays
    x0      = np.zeros(n)
    x_t     = np.zeros(n)
    y_w     = np.zeros(n)
    x_f_ens = np.zeros((n, N))
    x_a_ens = np.zeros((n, N))

    RMSE_f   = np.zeros(n_out+1) 
    RMSE_a   = np.zeros(n_out+1)
    Spread_f = np.zeros(n_out+1)
    Spread_a = np.zeros(n_out+1)
    rho_list = np.zeros(n_out+1)

    X_t = np.zeros((n*(n_out+1), n_extr))
    Y_o = np.zeros((n*(n_out+1), n_extr))
    X_f = np.zeros((n*(n_out+1), n_extr))
    X_a = np.zeros((n*(n_out+1), n_extr))

    # Parameters of adaptive covariance inflation at initial time
    kappa = 1.1
    ratio = 1.0
    rho   = 1.0   
    
   
    ###################################
    #   True state and observations   #
    ###################################

    # Initialization of random number sequence
    np.random.seed(0)     # for training

    # True state
    x0[0:n] = F*np.ones(n) + np.random.randn(n)
    X_hist = runge(x0, n, dt, nstep, F)
    


    # Observations
    Y_hist = np.zeros((nstep+1, n))
    for istep in range(nstep+1):
        error_o = np.random.randn(n)
        Y_hist[istep,:] = X_hist[istep,:] + sigma_o*error_o[:]

    # Observation operator
    H = np.zeros((m, n), dtype='int')
    for i in range(m):
        H[i, obs_dist*i] = 1
        
    print('H.shape =', H.shape)
    
    
    #########################################
    #   Data assimilation at initial time   #
    #########################################

    # Initialization of random number sequence
    np.random.seed(12345)

    i_DA  = 0
    i_out = 0

    x0[0:n] = F*np.ones(n) + np.random.randn(n)

    # True stata and obsetvations
    x_t[:] = X_hist[i_DA*obs_step, :]
    y_w[:] = Y_hist[i_DA*obs_step, :]
    y_o    = H@y_w

    print('x_t =', x_t)
    print('y_o =', y_o)
    
    
    # Forecast ensemble
    for k in range(N):
        error_f = sigma_f*np.random.randn(n)
        x_f_ens[:, k] = x0[:] + error_f[:]

    # Mean and covariance of forecast
    x_f_mean = np.average(x_f_ens, axis=1)
    P_f      = np.cov(x_f_ens)

    print('x_f_mean =', x_f_mean)
    print('P_f =', '\n', P_f)
    
    
    # Analysis ensemble
    if m == 0:
        x_a_ens = np.copy(x_f_ens)
    else:
        x_a_ens = EnSRF(x_f_ens, x_f_mean, y_o, H, n, m, N, sigma_o, obs_dist, r_local)

    # Mean and covariance of analysis
    x_a_mean = np.average(x_a_ens, axis=1)
    P_a      = np.cov(x_a_ens)

    print('x_a_mean =', x_a_mean)
    print('P_a =', '\n', P_a)
    
    # Historical data
    for i in range(n):
        x_extr = extract(x_t, n, i, r_extr)
        X_t[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
        x_extr = extract(y_o, n, i, r_extr)
        Y_o[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
        x_extr = extract(x_f_mean, n, i, r_extr)
        X_f[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
        x_extr = extract(x_a_mean, n, i, r_extr)
        X_a[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
        
    # RMSEs
    RMSE_f[i_out] = np.sqrt(np.average((x_f_mean - x_t)**2))
    RMSE_a[i_out] = np.sqrt(np.average((x_a_mean - x_t)**2))

    # Spreads
    Spread_f[i_out] = np.sqrt(np.trace(P_f)/float(n))
    Spread_a[i_out] = np.sqrt(np.trace(P_a)/float(n))
                
    # Inflation factor
    rho_list[i_out] = rho
    
    ############################################
    #   Data assimilation after initial time   #
    ############################################

    t = i_DA*obs_step*dt
    print('t =', '{0:6.1f}'.format(t), '  RMSE_f =', '{0:6.3f}'.format(RMSE_f[i_out]), \
    ' RMSE_a =', '{0:6.3f}'.format(RMSE_a[i_out]), ' Spread_f =', '{0:6.3f}'.format(Spread_f[i_out]), \
    ' Spread_a =', '{0:6.3f}'.format(Spread_a[i_out]), ' rho =', '{0:6.3f}'.format(rho), ' m =', m)

    while i_DA < n_DA:

        i_DA += 1

        # True state and observations
        x_t[:] = X_hist[i_DA*obs_step, :]
        y_w[:] = Y_hist[i_DA*obs_step, :]
        y_o    = H@y_w
        
        # Forcast ensemble
        for k in range(N):
            work = runge(x_a_ens[:, k], n, dt, obs_step, F)
            x_f_ens[:,k] = work[obs_step,:]   
        
        # Mean and covariance of forecast
        x_f_mean = np.average(x_f_ens, axis=1)
        P_f      = np.cov(x_f_ens)
        
        # Covariance inflation
        rho, ratio = inflation_rev(x_f_mean, P_f, y_o, H, sigma_o, n, m, rho_min, rho_max, kappa, rho, ratio)
        for k in range(N):
            x_f_ens[:, k] = x_f_mean[:] + np.sqrt(rho)*(x_f_ens[:, k] - x_f_mean[:])

        # Analysis ensemble
        if m == 0:
            x_a_ens = np.copy(x_f_ens)
        else:
            x_a_ens = EnSRF(x_f_ens, x_f_mean, y_o, H, n, m, N, sigma_o, obs_dist, r_local)

        # Mean and covarina of analysis
        x_a_mean = np.average(x_a_ens, axis=1)
        P_a      = np.cov(x_a_ens)

        if i_DA%out_int == 0:

            i_out += 1
            
            # Historical data
            for i in range(n):
                x_extr = extract(x_t, n, i, r_extr)
                X_t[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
                x_extr = extract(y_o, n, i, r_extr)
                Y_o[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
                x_extr = extract(x_f_mean, n, i, r_extr)
                X_f[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
                x_extr = extract(x_a_mean, n, i, r_extr)
                X_a[n*i_out + i, 0:n_extr] = x_extr[0:n_extr]
            
            # RMSEs
            RMSE_f[i_out] = np.sqrt(np.average((x_f_mean - x_t)**2))
            RMSE_a[i_out] = np.sqrt(np.average((x_a_mean - x_t)**2))

            # Spreads
            Spread_f[i_out] = np.sqrt(np.trace(P_f)/float(n))
            Spread_a[i_out] = np.sqrt(np.trace(P_a)/float(n))

            # Inflation factor
            rho_list[i_out] = rho
            
        if i_DA%(50*out_int) == 0:
            t = i_DA*obs_step*dt
            print('t =', '{0:6.1f}'.format(t), '  RMSE_f =', '{0:6.3f}'.format(RMSE_f[i_out]), \
            ' RMSE_a =', '{0:6.3f}'.format(RMSE_a[i_out]), ' Spread_f =', '{0:6.3f}'.format(Spread_f[i_out]), \
            ' Spread_a =', '{0:6.3f}'.format(Spread_a[i_out]), ' rho =', '{0:6.3f}'.format(rho), ' m =', m)
            
    ##################
    #   Statistics   #
    ##################

    n_start = 1050 + 1     # Start index of validation data

    # Time average of RMSEs
    RMSE_f_mean = np.sqrt(np.average(RMSE_f[n_start:]**2))
    RMSE_a_mean = np.sqrt(np.average(RMSE_a[n_start:]**2))

    # Time average of spreads
    Spread_f_mean    = np.sqrt(np.average(Spread_f[n_start:]**2))
    Spread_a_mean    = np.sqrt(np.average(Spread_a[n_start:]**2))

    # Time average of inflation factor
    rho_mean = np.average(rho_list[n_start:])

    print('r_local =', '{0:3d}'.format(r_local), '   rho =', '{0:6.3f}'.format(rho_mean), \
    '   RMSE_f =', '{0:6.3f}'.format(RMSE_f_mean), '   RMSE_a =', '{0:6.3f}'.format(RMSE_a_mean), \
    '   Spread_f =', '{0:6.3f}'.format(Spread_f_mean), '   Spread_a =', '{0:6.3f}'.format(Spread_a_mean))        
    
    # # plot lines
    # plt.plot(y_o, label = "Observation")
    # plt.plot(x_t, label = "True state")
    # plt.plot(x_f_mean, label = "Forecast")
    # plt.plot(x_a_mean, label = "Analisis")
    # plt.legend()
    # plt.show()
    
    ###########################################################
    #   Preparation of datasets for training and validation   #
    ###########################################################

    N_start  = (50+1)*n               # Start index of original data
    N_middle = (n_start-1)*n          # Middle index of original data
    N_out    = n_out*n                # End index of original data
    N_sample = (N_out+n) - N_start    # No of samples

    print('N_start =', N_start, '     N_out =', N_out, '     N_sample =', N_sample)

    X_t_mean = np.average(X_t[N_start:N_middle+n]) 
    X_t_std  = np.std    (X_t[N_start:N_middle+n])

    print('X_t_mean =', X_t_mean)
    print('X_t_std =',  X_t_std)

    nn = np.zeros(r_extr+1, dtype='int')
    n0 = np.zeros(r_extr+1, dtype='int')
    n1 = np.zeros(r_extr+1, dtype='int')
    for i in range(r_extr+1):
        nn[i] = 2*i + 1
        n0[i] = r_extr - i
        n1[i] = n0[i] + nn[i] 
    print('nn =', nn)
    print('n0 =', n0)
    print('n1 =', n1)

    data0   = np.zeros((N_sample, 3*nn[ 0]), dtype='float32')
    data1   = np.zeros((N_sample, 3*nn[ 1]), dtype='float32')
    data2   = np.zeros((N_sample, 3*nn[ 2]), dtype='float32')
    data3   = np.zeros((N_sample, 3*nn[ 3]), dtype='float32')
    data4   = np.zeros((N_sample, 3*nn[ 4]), dtype='float32')
    data5   = np.zeros((N_sample, 3*nn[ 5]), dtype='float32')
    data6   = np.zeros((N_sample, 3*nn[ 6]), dtype='float32')
    data7   = np.zeros((N_sample, 3*nn[ 7]), dtype='float32')
    data8   = np.zeros((N_sample, 3*nn[ 8]), dtype='float32')
    data9   = np.zeros((N_sample, 3*nn[ 9]), dtype='float32')
    data10  = np.zeros((N_sample, 3*nn[10]), dtype='float32')
    targets = np.zeros((N_sample, 1))

    data0 [0:N_sample,        0:  nn[ 0]] = (Y_o[N_start:N_out+n, n0[ 0]:n1[ 0]] - X_t_mean)/X_t_std
    data0 [0:N_sample,   nn[ 0]:2*nn[ 0]] = (X_f[N_start:N_out+n, n0[ 0]:n1[ 0]] - X_t_mean)/X_t_std
    data0 [0:N_sample, 2*nn[ 0]:3*nn[ 0]] = (X_a[N_start:N_out+n, n0[ 0]:n1[ 0]] - X_t_mean)/X_t_std

    data1 [0:N_sample,        0:  nn[ 1]] = (Y_o[N_start:N_out+n, n0[ 1]:n1[ 1]] - X_t_mean)/X_t_std
    data1 [0:N_sample,   nn[ 1]:2*nn[ 1]] = (X_f[N_start:N_out+n, n0[ 1]:n1[ 1]] - X_t_mean)/X_t_std
    data1 [0:N_sample, 2*nn[ 1]:3*nn[ 1]] = (X_a[N_start:N_out+n, n0[ 1]:n1[ 1]] - X_t_mean)/X_t_std

    data2 [0:N_sample,        0:  nn[ 2]] = (Y_o[N_start:N_out+n, n0[ 2]:n1[ 2]] - X_t_mean)/X_t_std
    data2 [0:N_sample,   nn[ 2]:2*nn[ 2]] = (X_f[N_start:N_out+n, n0[ 2]:n1[ 2]] - X_t_mean)/X_t_std
    data2 [0:N_sample, 2*nn[ 2]:3*nn[ 2]] = (X_a[N_start:N_out+n, n0[ 2]:n1[ 2]] - X_t_mean)/X_t_std

    data3 [0:N_sample,        0:  nn[ 3]] = (Y_o[N_start:N_out+n, n0[ 3]:n1[ 3]] - X_t_mean)/X_t_std
    data3 [0:N_sample,   nn[ 3]:2*nn[ 3]] = (X_f[N_start:N_out+n, n0[ 3]:n1[ 3]] - X_t_mean)/X_t_std
    data3 [0:N_sample, 2*nn[ 3]:3*nn[ 3]] = (X_a[N_start:N_out+n, n0[ 3]:n1[ 3]] - X_t_mean)/X_t_std

    data4 [0:N_sample,        0:  nn[ 4]] = (Y_o[N_start:N_out+n, n0[ 4]:n1[ 4]] - X_t_mean)/X_t_std
    data4 [0:N_sample,   nn[ 4]:2*nn[ 4]] = (X_f[N_start:N_out+n, n0[ 4]:n1[ 4]] - X_t_mean)/X_t_std
    data4 [0:N_sample, 2*nn[ 4]:3*nn[ 4]] = (X_a[N_start:N_out+n, n0[ 4]:n1[ 4]] - X_t_mean)/X_t_std

    data5 [0:N_sample,        0:  nn[ 5]] = (Y_o[N_start:N_out+n, n0[ 5]:n1[ 5]] - X_t_mean)/X_t_std
    data5 [0:N_sample,   nn[ 5]:2*nn[ 5]] = (X_f[N_start:N_out+n, n0[ 5]:n1[ 5]] - X_t_mean)/X_t_std
    data5 [0:N_sample, 2*nn[ 5]:3*nn[ 5]] = (X_a[N_start:N_out+n, n0[ 5]:n1[ 5]] - X_t_mean)/X_t_std

    data6 [0:N_sample,        0:  nn[ 6]] = (Y_o[N_start:N_out+n, n0[ 6]:n1[ 6]] - X_t_mean)/X_t_std
    data6 [0:N_sample,   nn[ 6]:2*nn[ 6]] = (X_f[N_start:N_out+n, n0[ 6]:n1[ 6]] - X_t_mean)/X_t_std
    data6 [0:N_sample, 2*nn[ 6]:3*nn[ 6]] = (X_a[N_start:N_out+n, n0[ 6]:n1[ 6]] - X_t_mean)/X_t_std

    data7 [0:N_sample,        0:  nn[ 7]] = (Y_o[N_start:N_out+n, n0[ 7]:n1[ 7]] - X_t_mean)/X_t_std
    data7 [0:N_sample,   nn[ 7]:2*nn[ 7]] = (X_f[N_start:N_out+n, n0[ 7]:n1[ 7]] - X_t_mean)/X_t_std
    data7 [0:N_sample, 2*nn[ 7]:3*nn[ 7]] = (X_a[N_start:N_out+n, n0[ 7]:n1[ 7]] - X_t_mean)/X_t_std

    data8 [0:N_sample,        0:  nn[ 8]] = (Y_o[N_start:N_out+n, n0[ 8]:n1[ 8]] - X_t_mean)/X_t_std
    data8 [0:N_sample,   nn[ 8]:2*nn[ 8]] = (X_f[N_start:N_out+n, n0[ 8]:n1[ 8]] - X_t_mean)/X_t_std
    data8 [0:N_sample, 2*nn[ 8]:3*nn[ 8]] = (X_a[N_start:N_out+n, n0[ 8]:n1[ 8]] - X_t_mean)/X_t_std

    data9 [0:N_sample,        0:  nn[ 9]] = (Y_o[N_start:N_out+n, n0[ 9]:n1[ 9]] - X_t_mean)/X_t_std
    data9 [0:N_sample,   nn[ 9]:2*nn[ 9]] = (X_f[N_start:N_out+n, n0[ 9]:n1[ 9]] - X_t_mean)/X_t_std
    data9 [0:N_sample, 2*nn[ 9]:3*nn[ 9]] = (X_a[N_start:N_out+n, n0[ 9]:n1[ 9]] - X_t_mean)/X_t_std

    data10[0:N_sample,        0:  nn[10]] = (Y_o[N_start:N_out+n, n0[10]:n1[10]] - X_t_mean)/X_t_std
    data10[0:N_sample,   nn[10]:2*nn[10]] = (X_f[N_start:N_out+n, n0[10]:n1[10]] - X_t_mean)/X_t_std
    data10[0:N_sample, 2*nn[10]:3*nn[10]] = (X_a[N_start:N_out+n, n0[10]:n1[10]] - X_t_mean)/X_t_std

    with open('data_M10T50R00_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data0 ) 
    with open('data_M10T50R01_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data1 )   
    with open('data_M10T50R02_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data2 )   
    with open('data_M10T50R03_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data3 )    
    with open('data_M10T50R04_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data4 )    
    with open('data_M10T50R05_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data5 )   
    with open('data_M10T50R06_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data6 )   
    with open('data_M10T50R07_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data7 )   
    with open('data_M10T50R08_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data8 )  
    with open('data_M10T50R09_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data9 )  
    with open('data_M10T50R10_constant.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data10)
                    
    targets[0:N_sample, 0] = (X_t[N_start:N_out+n, r_extr] - X_t_mean)/X_t_std  
    with open('targets_R00.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(targets)

    factor = np.array([X_t_mean, X_t_std])
    with open('factor.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(factor)

    print('data0 =',   '\n', data0  [0:n, :])
    print('targets =', '\n', targets[0:n, 0])
    


if __name__ == "__main__":
    main()