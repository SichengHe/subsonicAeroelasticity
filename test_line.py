import numpy as np
import matplotlib.pyplot as plt
import fouriercoeff as fc

disp_dest = np.matrix([[-0.01973829],
        [-0.01128884],
        [-0.01102459],
        [-0.00653375],
        [-0.01636958],
        [-0.0095091 ]])

disp_perturbed = np.matrix([[-0.01932677],
        [-0.01058071],
        [-0.01030624],
        [-0.0063929 ],
        [-0.01669807],
        [-0.00883528]])

dR_dDisp = np.matrix([[-0.05189822, -0.21498957,  0.56334142,  1.45001791, -0.44352816,
         -1.23374967],
        [ 0.05619416,  0.19428259, -0.43382364, -1.20557572,  0.32362776,
          0.99105109],
        [-0.48679267, -1.38477662, -0.05212285, -0.1963437 ,  0.49005996,
          1.42705916],
        [ 0.35747079,  1.12104524,  0.05660407,  0.17827739, -0.37457199,
         -1.1799468 ],
        [ 0.53864932,  1.59961138, -0.5108814 , -1.25506402, -0.04605436,
         -0.1926532 ],
        [-0.41365664, -1.33079865,  0.37679195,  1.01471388,  0.05104366,
          0.17418463]])

dx = np.matrix([[ 0.00230943],
        [ 0.00102368],
        [ 0.00286399],
        [ 0.00030676],
        [ 0.0050458 ],
        [-0.00127906]])



print()
alpha_list = [1e-5, 10**-4.5, 1e-4, 10**-3.5, 1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.7, 1.0]
alpha_list_log = []
err_list = []

original_dff = disp_perturbed - disp_dest
original_err = np.log10(np.sqrt(np.transpose(original_dff).dot(original_dff)[0, 0]))
original_err_list = []

for i in xrange(len(alpha_list)):

	alpha = alpha_list[i]
	alpha_list_log.append(np.log10(alpha))

	disp_alpha = alpha * dx + disp_perturbed

	err_alpha = disp_alpha - disp_dest

	err_alpha_norm_log = np.log10(np.sqrt(np.transpose(err_alpha).dot(err_alpha)[0, 0]))

	err_list.append(err_alpha_norm_log)


	original_err_list.append(original_err)

plt.plot(alpha_list_log, err_list, '-o')
plt.plot(alpha_list_log, original_err_list, '--')
plt.xlabel('log(stepsize/unit Newton step )')
plt.ylabel('log(norm(disp-target disp))')
plt.show()



dx_norm = np.sqrt(np.transpose(dx).dot(dx)[0, 0])
perturb = disp_perturbed - disp_dest
perturb_norm = np.sqrt(np.transpose(perturb).dot(perturb)[0, 0])
print dx/dx_norm, perturb/perturb_norm


def disp2disphat(disp, invPhi, ntimeintervalsspectral):

    "convert disp (2*ntimeintervalsspectral,1) to modal space and conduct fourier transform"

    disp_reorganized = disp.reshape((2, ntimeintervalsspectral), order='F')

    disp_reduced_space = np.matrix(np.zeros((2, ntimeintervalsspectral)))

    for i in xrange(ntimeintervalsspectral):
        disp_reduced_space[:, i] = invPhi.dot(disp_reorganized[:, i])

    disp_reduced_space_0, disp_reduced_space_c, disp_reduced_space_s = fc.FourierCoeff_md(disp_reduced_space)

    return disp_reduced_space_0, disp_reduced_space_c, disp_reduced_space_s

def disphat2disp(disp_reduced_space_0, disp_reduced_space_c, disp_reduced_space_s, Phi, ntimeintervalsspectral):

    "convert disphat (modal form in freq domain) to disp (time domain displacement)"

    disp_regular_space_0 = Phi.dot(disp_reduced_space_0)
    disp_regular_space_c = []
    disp_regular_space_s = []

    omega_N = (ntimeintervalsspectral - 1) / 2

    for i in xrange(omega_N):

        disp_regular_space_c.append(Phi.dot(disp_reduced_space_c[i]))
        disp_regular_space_s.append(Phi.dot(disp_reduced_space_s[i]))


    disp = np.matrix(np.zeros((2*ntimeintervalsspectral, 1)))

    for i in xrange(ntimeintervalsspectral):

        disp[2*i,   0] = disp_regular_space_0[0, 0]
        disp[2*i+1, 0] = disp_regular_space_0[1, 0]

        for j in xrange(omega_N):

            phase = (j + 1) * (1.0 * i) / ntimeintervalsspectral * (2.0 * np.pi)

            disp[2*i,   0] += disp_regular_space_c[j][0] * np.cos(phase) + disp_regular_space_s[j][0] * np.sin(phase)
            disp[2*i+1, 0] += disp_regular_space_c[j][1] * np.cos(phase) + disp_regular_space_s[j][1] * np.sin(phase)

    return disp










disp_for_CSD = np.matrix([[-0.01973829],
 [-0.01128884],
 [-0.01102459],
 [-0.00653375],
 [-0.01636958],
 [-0.0095091 ]])

disp_perturbed_Newton = disp_perturbed + dx

invPhi = np.matrix(np.eye(2))
Phi = np.matrix(np.eye(2))

disp_reduced_space_0, disp_reduced_space_c, disp_reduced_space_s = disp2disphat(disp_for_CSD, invPhi, 3)
disp_reduced_space_0_p, disp_reduced_space_c_p, disp_reduced_space_s_p = disp2disphat(disp_perturbed, invPhi, 3)
disp_reduced_space_0_N, disp_reduced_space_c_N, disp_reduced_space_s_N = disp2disphat(disp_perturbed_Newton, invPhi, 3)




dT = 2 * np.pi / 3

T_list = []
hb_list = []
alpha_list = []

for i in xrange(3):

    T_list.append(i * dT)

    hb_list.append(disp_for_CSD[2 * i, 0])

    alpha_list.append(disp_for_CSD[2 * i + 1, 0])


n = 120

dT2 = 2 * np.pi / n


T_list0 = []

hb_list0 = []
hb_listp = []
hb_listN = []

alpha_list0 = []
alpha_listp = []
alpha_listN = []

for i in xrange(n):

    T_loc = i * dT2

    T_list0.append(T_loc)

    hb_loc0 = disp_reduced_space_0[0, 0] + disp_reduced_space_c[0][0, 0] * np.cos(T_loc) \
    + disp_reduced_space_s[0][0, 0] * np.sin(T_loc)
    hb_locp = disp_reduced_space_0_p[0, 0] + disp_reduced_space_c_p[0][0, 0] * np.cos(T_loc) \
    + disp_reduced_space_s_p[0][0, 0] * np.sin(T_loc)
    hb_locN = disp_reduced_space_0_N[0, 0] + disp_reduced_space_c_N[0][0, 0] * np.cos(T_loc) \
    + disp_reduced_space_s_N[0][0, 0] * np.sin(T_loc)

    alpha_loc0 = disp_reduced_space_0[1, 0] + disp_reduced_space_c[0][1, 0] * np.cos(T_loc) \
    + disp_reduced_space_s[0][1, 0] * np.sin(T_loc)
    alpha_locp = disp_reduced_space_0_p[1, 0] + disp_reduced_space_c_p[0][1, 0] * np.cos(T_loc) \
    + disp_reduced_space_s_p[0][1, 0] * np.sin(T_loc)
    alpha_locN = disp_reduced_space_0_N[1, 0] + disp_reduced_space_c_N[0][1, 0] * np.cos(T_loc) \
    + disp_reduced_space_s_N[0][1, 0] * np.sin(T_loc)

    hb_list0.append(hb_loc0)
    hb_listp.append(hb_locp)
    hb_listN.append(hb_locN)

    alpha_list0.append(alpha_loc0)
    alpha_listp.append(alpha_locp)
    alpha_listN.append(alpha_locN)




plt.figure()
plt.plot(T_list, hb_list, 'bo', label='Target h')
plt.plot(T_list, alpha_list, 'ro', label='Target alpha')
plt.plot(T_list0, hb_list0, 'b-', label='Target h')
plt.plot(T_list0, alpha_list0, 'r-', label='Target alpha')
plt.plot(T_list0, hb_listp, 'b--', label='Perturbed h')
plt.plot(T_list0, alpha_listp, 'r--', label='Perturbed alpha')
plt.plot(T_list0, hb_listN, 'b-.',label='Newton h')
plt.plot(T_list0, alpha_listN, 'r-.', label='Newton alpha')

plt.legend(loc = 8)

plt.xlabel('phase')
plt.ylabel('displacement')

plt.show()