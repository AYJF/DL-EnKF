#############################################################################################
#                                                                                           #
#                                  Miscellaneous modules                                    #
#                                                                                           #
#                             written by Tadashi Tsuyuki (MRI, Japan) on February 4, 2022   #
#############################################################################################


import numpy as np
import matplotlib.pyplot as plt


###########################################
#   Supplement of misssing observations   #
###########################################

def EnKF_observation (y_o, x_a_mean, H, m, n):

    h = np.zeros(n)
    y_o_full = np.zeros(n)
    H_full = np.zeros((n, n))
    
    y_o_full[:] = x_a_mean[:]
    
    for j in range(m):
        h[:] = H[j, :]    
        for i in range(n):
            if h[i] > 0:
                y_o_full[i] = y_o[j]
                H_full[i, i] = 1

    return y_o_full, H_full


###################################
#   Random observation operator   #
###################################

def random_observation (n, p_c):

    m = 0
    
    for i in range(n):
        p = np.random.random()
        if p >= p_c:
            H = np.zeros(n)
            H[i] = 1
            m = m + 1
            break

    for i in range(i+1, n):
        p = np.random.random()
        if  p >= p_c:
            m = m + 1
            H_loc = np.zeros(n)
            H_loc[i] = 1
            H = np.block([[H], [H_loc]])

    return H, m


#############################################
#   Extraction of input variables for DNN   #
#############################################

def extract (x, n, i, r_extr):
    
    x_extr = np.zeros(2*r_extr + 1)
    
    for l in range(2*r_extr + 1):
        i1 = i - r_extr + l
        if   i1 < 0:
            i1 = i1 + n
        elif i1 > n - 1:
            i1 = i1 - n
        x_extr[l] = x[i1]
        
    return x_extr


#############################
#   Plot of loss function   #
#############################

def loss_plot(x, y, title, ymin, ymax):
    
    fig, ax = plt.subplots()

    ax.plot(x, label="training")
    ax.plot(y, label="validation")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss function')
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.legend()

    plt.show()