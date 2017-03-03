# TS
# 2D
# Sicheng He, All Rights Reserved
import numpy as np
import math as math

def timeder_mat(T,n):

    D = np.matrix(np.zeros((n,n)))

    if (n%2==0):
        # even
        for i in xrange(n):
            for j in xrange(n):
                if (i!=j):
                    D[i,j] = 0.5*(-1)**(i-j)/math.tan(np.pi*(i-j)/n)

    else:
        # odd
        for i in xrange(n):
            for j in xrange(n):
                if (i!=j):
                    D[i,j] = 0.5*(-1)**(i-j)/math.sin(np.pi*(i-j)/n)

    D = D*2.0*np.pi/T

    return D

def sort_N(N):

    # it will return things in the following order
    # for FFT
    # eg. N=5
    # return 0,1,2,-2,1
    #     N=6
    # return -2,-1,0,1,2,3

    N_vec = []
    N_vec.append(0)

    if (N%2==1):
        # odd
        N_half = (N-1)/2

        for i in xrange(N_half):
            N_vec.append(i+1)
        for i in xrange(N_half):
            N_vec.append(i-N_half)

    else:
        # even
        N_half = N/2

        for i in xrange(N_half):
            N_vec.append(i+1)
        for i in xrange(N_half-1):
            N_vec.append(i-N_half+1)

    return N_vec



def TS_evaluate(coef,t,period):

    t_scaled = t/period*(2.0*np.pi)

    N = len(coef)
    coef = coef/N # rescaled the coef

    N_vec = sort_N(N)

    y = 0.0
    for i in xrange(N):
        N_loc = N_vec[i]
        y += coef[i]*np.exp(N_loc*1j*t_scaled)

    return np.real(y)









# test
# y = sin(t)
# T = 2pi/1 = 2pi
# N = 3
if (1==0):
    N = 10
    t_vec = (np.linspace(0,2.0*np.pi,N+1))[:N]
    y_vec = np.matrix(np.zeros((N,1)))
    ydot_analy_vec = np.matrix(np.zeros((N,1)))
    for i in xrange(N):
        t_loc = t_vec[i]
        y_vec[i,0] = math.sin(t_loc)
        ydot_analy_vec[i,0] = math.cos(t_loc)

    D = timeder_mat(2.0*np.pi,N)
    ydot_vec = D.dot(y_vec)

    err = abs(ydot_analy_vec-ydot_vec)
    print '++++++++++err is++++++++++', err
