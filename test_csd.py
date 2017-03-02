import numpy
import data_extracter as d_e
import pystructts as pystructts
import matplotlib.pyplot as plt


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


# get the displacement history
filename_disp = "history.dat"
plg_col = 1
ptch_col = 2

plg_vec = d_e.fetch_data(filename_disp,line_begin,line_end,plg_col)
ptch_vec = d_e.fetch_data(filename_disp,line_begin,line_end,ptch_col)

plg_vec = [x*b for x in plg_vec] # get dimensinal plg


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

disp = CSDSolver.disp_vec


# compare with unsteady results
plg_TS = []
ptch_TS = []
for i in xrange(ntimeintervalsspectral):

    plg_TS.append(0.5*disp[2*i, 0])
    ptch_TS.append(disp[2*i+1, 0])


print("omega_cal",2.0*numpy.pi/(t_vec[-1]-t_vec[0]))
print("plg_TS",plg_TS,"plg_un",set_TS_data(plg_vec,ind_vec))
print("ptch_TS",ptch_TS,"ptch_un",set_TS_data(ptch_vec,ind_vec))

plt.figure(1)

plt.subplot(211)
plt.plot(t_vec,plg_vec, label="unsteady")
plt.plot(t_TS,plg_TS,'o', label="TS")
plt.xlabel('time')
plt.ylabel('plunging')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,\
       ncol=2, mode="expand", borderaxespad=0.)

plt.subplot(212)
plt.plot(t_vec,ptch_vec, label="unsteady")
plt.plot(t_TS,ptch_TS,'o', label="TS")
plt.xlabel('time')
plt.ylabel('pitching')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,\
       ncol=2, mode="expand", borderaxespad=0.)

plt.show()
