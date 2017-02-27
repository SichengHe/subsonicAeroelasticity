import numpy as np


def FourierCoeff_md(U):
    "here U could be 2 * ntimeintervalsspectral matrix"
    dim, n = U.shape
    omega_N = (n - 1) / 2

    c_0 = np.matrix(np.zeros( (dim,1) ))
    c_c = []
    c_s = []

    for i in xrange(omega_N):

        c_c.append([])
        c_s.append([])


    for i in xrange(dim):

        u_one_dim = U[i, :]

        c_0_loc, c_c_loc, c_s_loc = FourierCoeff(u_one_dim)

        c_0[i] = c_0_loc

        for j in xrange(omega_N):

            c_c[j].append([c_c_loc[j]])
            c_s[j].append([c_s_loc[j]])

    for i in xrange(omega_N):

        c_c[i] = np.matrix(c_c[i])
        c_s[i] = np.matrix(c_s[i])

    return c_0, c_c, c_s


def FourierCoeff(u_mat):

    " u = c_0 + sum_i (c_ci cos(ix) + c_si sin(ix)) "
    " Only keep the real part!"
    " assume 'u' is of odd number"
    " assume 'u' is 1 * ntimeintervalsspectral matrix"

    n = u_mat.shape[1]
    omega_N = (n - 1) / 2 # number of nonzero freq

    # convert u to a list for convinience
    u = []
    for i in xrange(n):
        u.append(u_mat[0, i])

    coeff_freq = np.fft.fft(u)

    c_0 = np.real(coeff_freq[0]) / n
    c_c = []
    c_s = []

    for i in xrange(omega_N):

        ind = i + 1

        c_pls = coeff_freq[ind]
        c_mns = coeff_freq[-ind]

        c_c.append(np.real(c_pls + c_mns) / n)
        c_s.append(np.real( 1j*(c_pls - c_mns)) / n)

    return c_0, c_c, c_s
