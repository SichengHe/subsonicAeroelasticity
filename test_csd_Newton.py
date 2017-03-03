import numpy
import data_extracter as d_e
import pystructts as pystructts
import matplotlib.pyplot as plt
import copy

# utils
def ind_gen(line_begin,line_end,ntimeintervalsspectral):

    d_ind = (line_end-line_begin)*1.0/ntimeintervalsspectral

    ind_vec = []

    for i in xrange(ntimeintervalsspectral):

        ind = int(d_ind*i)
        ind_vec.append(ind)

    return ind_vec


# get the load
def set_TS_data(data_un,ind_vec):

    data_TS = []

    for i in xrange(len(ind_vec)):

        data_TS.append(data_un[ind_vec[i]])

    return data_TS


"main part"
b = 0.5

ntimeintervalsspectral = 3


# history interval
line_begin = 2931
line_end = 3375

ind_vec = ind_gen(line_begin,line_end,ntimeintervalsspectral)


# get the load history
filename_load = "sumbHistory.dat"
t_col = 2
Cl_col = -4
Cm_col = -2

t_vec = d_e.fetch_data(filename_load,line_begin,line_end,t_col)
Cl_vec = d_e.fetch_data(filename_load,line_begin,line_end,Cl_col)
Cm_vec = d_e.fetch_data(filename_load,line_begin,line_end,Cm_col)

Cm_vec = [x*(-1) for x in Cm_vec]

t_TS = set_TS_data(t_vec,ind_vec)
Cl_TS = set_TS_data(Cl_vec,ind_vec)
Cm_TS = set_TS_data(Cm_vec,ind_vec)



f = []
for i in xrange(ntimeintervalsspectral):
    f.append(Cl_TS[i])
    f.append(Cm_TS[i])


b = 1.0/2.0

dt = 3.0

wf = 100.0
xa = 1.8
r2a = 3.48
mu = 60.0
wh = wf
wa = wf
wh_wa = wh/wa

Vs = 0.525

Ma = 0.85
omega = 81.07#87.07


CSDSolver = pystructts.pyStructTS(xa,r2a,wa,wh_wa,mu,b, \
    Vs,\
    Ma, omega,
    ntimeintervalsspectral)

CSDSolver.solve(f)

if (1==0):
	# [[-0.01962922]
#  [-0.01124823]
#  [-0.01237087]
#  [-0.00726651]
#  [-0.0126078 ]
#  [-0.00748893]]
	disp = CSDSolver.disp_vec
else:
	disp = numpy.matrix([[-0.01962922+0.0003],
	 [-0.01124823+0.0003],
	 [-0.01237087+0.0003],
	 [-0.00726651+0.0003],
	 [-0.0126078+0.0003 ],
	 [-0.00748893+0.0003]])


res = CSDSolver.set_disp_and_evaluate(disp)

print "res", res


# sens analysis:
epsilon = 1e-8
dres_ddisp = numpy.matrix(numpy.zeros((2*ntimeintervalsspectral, 2*ntimeintervalsspectral)))
for i in xrange(ntimeintervalsspectral*2):

	# +

	disp_perturbed_plus = copy.deepcopy(disp)
	disp_perturbed_plus[i, 0] *= (1 + epsilon)

	res_plus = CSDSolver.set_disp_and_evaluate(disp_perturbed_plus)

	# -

	disp_perturbed_minus = copy.deepcopy(disp)
	disp_perturbed_minus[i, 0] *= (1 - epsilon)

	res_minus = CSDSolver.set_disp_and_evaluate(disp_perturbed_minus)



	dres_ddisp[:, i] = (res_plus - res_minus) / (disp_perturbed_plus[i, 0] - disp_perturbed_minus[i, 0])

# reset
CSDSolver.set_disp_and_evaluate(disp)


print "dres_ddisp", dres_ddisp


ddisp = numpy.linalg.solve(dres_ddisp, -res)
disp_new = disp + ddisp
res_new = CSDSolver.set_disp_and_evaluate(disp_new)


print "res old", res, "res_new", res_new, "disp", disp, "disp_new", disp_new

