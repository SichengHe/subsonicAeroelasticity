import numpy as np
from mpi4py import MPI
from sumb import SUMB
import pyaerots as pyaerots
from pywarp import MBMesh
from baseclasses import *
import copy
import matplotlib.pyplot as plt
import random
import fouriercoeff as fc
import linesearch as LS
import data_extracter as d_e

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

def dispCSD2dispCFD(dispCSD, b, ntimeintervalsspectral):

    dispCFD = np.matrix(np.zeros((2 * ntimeintervalsspectral, 1)))

    for i in xrange(ntimeintervalsspectral):

        dispCFD[2 * i,    0] = dispCSD[2 * i,     0] * b
        dispCFD[2 * i +1, 0] = dispCSD[2 * i + 1, 0]

    return dispCFD

def loadCFD2loadCSD(loadCFD, ntimeintervalsspectral):

    "maps CFD load (cl, cm) to csd load (-cl, 2cm)"

    loadCSD = np.matrix(np.zeros((2*ntimeintervalsspectral, 1)))

    for i in xrange(ntimeintervalsspectral):

        loadCSD[2*i,   0] = loadCFD[2*i,   0] * (-1.0)
        loadCSD[2*i+1, 0] = loadCFD[2*i+1, 0] * 2.0

    return loadCSD

def assembleHat2Vec(uhat_0, uhat_c, uhat_s, ntimeintervalsspectral):

    omega_N = (ntimeintervalsspectral - 1) / 2

    uhat_vec = np.matrix(np.zeros((2 * ntimeintervalsspectral, 1)))

    "res_vec is organized in: 0, c1, s1, c2, s2, ..."

    uhat_vec[0:2,0] = uhat_0

    for i in xrange(omega_N):

        uhat_vec[4*i + 2: 4*i + 4, 0] = uhat_c[i]
        uhat_vec[4*i + 4: 4*i + 6, 0] = uhat_s[i]

    return uhat_vec

def dissembleVec(uhat_vec, ntimeintervalsspectral):

    omega_N = (ntimeintervalsspectral - 1) / 2

    uhat_0 = uhat_vec[0:2, 0]

    uhat_c = []
    uhat_s = []

    for i in xrange(omega_N):

        uhat_c.append(uhat_vec[4*i + 2: 4*i + 4, 0])
        uhat_s.append(uhat_vec[4*i + 4: 4*i + 6, 0])

    return uhat_0, uhat_c, uhat_s


def set_disp_res(disp_for_CSD, b, ntimeintervalsspectral, TS_CFDSolver, loadCFD_dest):

    disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)
    loadCFD = TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD)

    res = loadCFD - loadCFD_dest

    return res


def sens_disp(epsilon, disp_for_CSD, b, ntimeintervalsspectral, TS_CFDSolver):

    var_num = disp_for_CSD.shape[0]

    dLoad_dDisp = np.matrix( np.zeros((var_num, var_num)) )


    for i in xrange(var_num):

        # +

        disp_for_CSD_new_plus = copy.deepcopy(disp_for_CSD)

        disp_for_CSD_new_plus[i, 0] *= (1 + epsilon)

        disp_for_CFD_new_plus = dispCSD2dispCFD(disp_for_CSD_new_plus, b, ntimeintervalsspectral)

        loadCFD_new_plus = TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD_new_plus)


        # -

        disp_for_CSD_new_minus = copy.deepcopy(disp_for_CSD)

        disp_for_CSD_new_minus[i, 0] *= (1 - epsilon)

        disp_for_CFD_new_minus = dispCSD2dispCFD(disp_for_CSD_new_minus, b, ntimeintervalsspectral)

        loadCFD_new_minus = TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD_new_minus)


        dLoad_dDisp[:, i] =  (loadCFD_new_plus - loadCFD_new_minus) / (disp_for_CSD_new_plus[i, 0] - disp_for_CSD_new_minus[i, 0]) 

    # reset

    disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)
    TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD)


    return dLoad_dDisp


def Newton_disp(disp_for_CSD, res, TS_CFDSolver, ntimeintervalsspectral, b):

  epsilon = 1e-4
  dRdx = sens_disp(epsilon, disp_for_CSD, b, ntimeintervalsspectral, TS_CFDSolver)

  dvar = np.linalg.solve(dRdx, -res)

  return disp_for_CSD + dvar

def Newton(disp_for_CSD, loadCFD_dest):

  global TS_CFDSolver
  global ntimeintervalsspectral
  global b

  res_norm_log_list = []

  for i in xrange(4):

    disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)
    loadCFD = TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD)

    res = loadCFD - loadCFD_dest
    res_norm = np.sqrt(np.transpose(res).dot(res)[0, 0])
    res_norm_log_list.append(np.log10(res_norm))


    disp_for_CSD = Newton_disp(disp_for_CSD, res, TS_CFDSolver, ntimeintervalsspectral, b)



  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if (rank == 0):

    print "disp_for_CSD", disp_for_CSD, "loadCFD_dest", loadCFD_dest, "loadCFD", loadCFD

    plt.plot(res_norm_log_list, '-o', markersize = 15)
    plt.xlabel('iteration index')
    plt.ylabel('res norm')
    plt.show()






"main part"
outputDirectory = "./"
ntimeintervalsspectral = 3

Ma = 0.85
#Ma = 0.3
omega = 81.07

"parameters"
wf = 100.0

xa = 1.8
r2a = 3.48
wh = wf
wa = wf
wh_wa = wh/wa

mu = 60.0

Vf = 0.525

gamma = 1.4
R = 287.870
P = 101325.0
b = 0.5
a2 = (Vf*b*wa/Ma)**2.0 * mu
T = a2 / (gamma*R)
Va = np.sqrt(a2) / (wa * b * 2)



print("++", 4.0/(np.pi * mu) * Ma**2 * Va**2, Vf**2/np.pi)
if (4.0/(np.pi * mu) * Ma**2 * Va**2 != Vf**2/np.pi):
    print("Error: Check Vf, kc: not consistent!")

# ======================================================================
#         GEOMETRY SOLVER INPUTS
# ======================================================================

areaRef = 1.0
chordRef = 1.0
xRef = -0.5
yRef = 0.0
xRot = -0.5
yRot = 0.0

b = chordRef/2.0

# ======================================================================
#         MOTION SOLVER INPUTS
# ======================================================================

alpha_0 = 1.0#1.01
alpha_m = 0.0

omega = 81.07#12.82
deltaAlpha = -alpha_m*np.pi/180.0
# ======================================================================
#         MESH INPUTS
# ======================================================================
gridFile = 'INPUT/naca64A010_euler-L2.cgns'
# meshOptions = {
#       # Files
#       'gridFile':gridFile,
# }
meshOptions = {
      # Files
      'gridFile':gridFile,

      # Warp Type
      'warpType':"algebraic",
      'solidSolutionType':'linear', # or nonLinear or steppedLinear
      'fillType':'linear', # 'cubic' does not work temporarily
      'solidWarpType':'n',
      'n':7,

      # Solution Parameters
      'nSolutionSteps':5, # only for nonLinear or steppedLinear
      'ksp_its':25,
      'warp_tol':1e-10, # overall tolerance for nonLinear/steppedLinear
      'reduction_factor_for_reform':0.5,
      'useSolutionMonitor':False,
      'useKSPMonitor':False,

      # Physical parameters
      'nu':0.0,
      'e_exp':1.0,
      'skew_exp':0.0,
      'useRotationCorrection':False,
}
# Create mesh object
mesh0 = MBMesh(options=meshOptions)
mesh_vec = []
for i in xrange(ntimeintervalsspectral):
    mesh_vec.append(MBMesh(options=meshOptions))

# ======================================================================
#         CFD INPUTS
# ======================================================================

MGCycle = '2w'
name = 'aero_struct'

aeroOptions = {
    # Common Parameters
    'gridFile':gridFile,
    'outputDirectory':'OUTPUT',

    # Physics Parameters
    'equationType':'euler',
    'equationMode':'time spectral',

    # Common Parameters
    #'smoother':'dadi',
    'CFL':1.0,
    'CFLCoarse':1.0,
    'MGCycle':"sg",#MGCycle,
    'MGStartLevel':-1,
    'nCyclesCoarse':3000,
    'nCycles':100000,
    'monitorvariables':['resrho','cl','cmz'],
    'useNKSolver':True,
    'nkswitchtol':1e-8,

    # Convergence Parameters
    'L2Convergence':1e-14,
    'L2ConvergenceCoarse':1e-2,

    # TS
    #'TSStability': True,
    'timeIntervals': ntimeintervalsspectral,
    'qmode':True,
    'alphaFollowing': False,

    'surfacevariables':['cp','vx','vy','vz','mach','rvx','rvy','rvz'],
    }

# Aerodynamic problem description

ap = AeroProblem(name=name, alpha=alpha_0,
             mach=Ma, T=T, P=P, R=R,
             areaRef=areaRef, chordRef=chordRef,
             xRef=xRef,xRot=xRot,
             degreePol=0,coefPol=[0.0],
             degreeFourier=1,omegaFourier=omega,
             cosCoefFourier=[0.0,0.0],sinCoefFourier=[deltaAlpha],
             evalFuncs=['cl','cd','cmx','cmy','cmz'])



# structural displacemenet

def ind_gen(line_begin,line_end,ntimeintervalsspectral):

  d_ind = (line_end-line_begin)*1.0/ntimeintervalsspectral

  ind_vec = []

  for i in xrange(ntimeintervalsspectral):

      ind = int(d_ind*i)
      ind_vec.append(ind)

  return ind_vec


line_begin = 2931
line_end = 3375
filename = "history.dat"
plg_col = 1
ptch_col = 2

ind_vec_list = ind_gen(line_begin,line_end,ntimeintervalsspectral)

plg_vec = d_e.fetch_data(filename,line_begin,line_end,plg_col)
ptch_vec = d_e.fetch_data(filename,line_begin,line_end,ptch_col)

disp_for_CSD = np.matrix(np.zeros((2*ntimeintervalsspectral, 1)))

plg_vec_TS = []
ptch_vec_TS = []

for i in xrange(ntimeintervalsspectral):

  ind = ind_vec_list[i]

  plg_vec_TS.append(plg_vec[ind])
  ptch_vec_TS.append(ptch_vec[ind])


for i in xrange(ntimeintervalsspectral):

  disp_for_CSD[2*i, 0] = plg_vec_TS[i]
  disp_for_CSD[2*i+1, 0] = ptch_vec_TS[i]



# displacement for CFD

disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)



# construct CFD solver

CFDSolver = SUMB(options=aeroOptions)
CFDSolver.addFamilyGroup('full_surface',['wall'])



# construct the TS CFD sovler

TS_CFDSolver = pyaerots.pyAeroTS(CFDSolver,ntimeintervalsspectral,mesh0,mesh_vec,xRot,yRot,ap)
TS_CFDSolver.setUndeformedSurfmesh()
loadCFD_dest = TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD)


Cl_TS_list = []
Cm_TS_list = []
for i in xrange(ntimeintervalsspectral):

  Cl_TS_list.append(loadCFD_dest[2*i, 0])
  Cm_TS_list.append(-loadCFD_dest[2*i + 1, 0])

print "Cl_TS_list", Cl_TS_list, "len(Cl_TS_list)", len(Cl_TS_list)

filename2 = "sumbHistory.dat"
t_col = 2
Cl_col = -4
Cm_col = -2


def postProcess(ind_vec_list,Cl_TS_list,Cm_TS_list,\
filename2,line_begin,line_end,t_col,Cl_col, Cm_col):

    import matplotlib.pyplot as plt

    # get the load from unsteady sol

    t_vec = d_e.fetch_data(filename2,line_begin,line_end,t_col)
    Cl_vec = d_e.fetch_data(filename2,line_begin,line_end,Cl_col)
    Cm_vec = d_e.fetch_data(filename2,line_begin,line_end,Cm_col)

    n_cfd_eval = len(ind_vec_list)

    print("+++++++++++++++++++++++++===n_cfd_eval",n_cfd_eval)

    plt.figure(1)

    plt.subplot(311)
    plt.plot(t_vec,Cl_vec, label="unsteady")
    plt.xlabel('time')
    plt.ylabel('Cl')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,\
           ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(312)
    plt.plot(t_vec,Cm_vec, label="unsteady")
    plt.xlabel('time')
    plt.ylabel('Cm')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,\
           ncol=2, mode="expand", borderaxespad=0.)

    marker_dict = {0: "bo", 1: "b^", 2:"bv"}

    ntimeintervalsspectral_list = []
    Cl_err_inf_norm = []
    Cm_err_inf_norm = []

    for i in xrange(n_cfd_eval):

        ind_vec = ind_vec_list[i]
        Cl_TS = Cl_TS_list[i]
        Cm_TS = Cm_TS_list[i]

        # get the time for TS
        ntimeintervalsspectral = len(Cl_TS)
        ntimeintervalsspectral_list.append(ntimeintervalsspectral)

        TS_t_vec = []
        for j in xrange(ntimeintervalsspectral):
            TS_t_vec.append(t_vec[ind_vec[j]])

        # Cm, Cl from time accurate simulation
        Cl_proj = []
        Cm_proj = []
        for j in xrange(ntimeintervalsspectral):
            Cl_proj.append(Cl_vec[ind_vec[j]])
            Cm_proj.append(Cm_vec[ind_vec[j]])

        Cl_err = []
        Cm_err = []
        for j in xrange(ntimeintervalsspectral):
            Cl_err.append(abs(Cl_proj[j]-Cl_TS[j]))
            Cm_err.append(abs(Cm_proj[j]-Cm_TS[j]))

        Cl_err_inf_norm.append(max(Cl_err))
        Cm_err_inf_norm.append(max(Cm_err))






        # Cl
        plt.subplot(311)
        plt.plot(TS_t_vec,Cl_TS,marker_dict[i], label="TS")


        # Cm
        plt.subplot(312)
        plt.plot(TS_t_vec,Cm_TS,marker_dict[i], label="TS")


    plt.subplot(313)
    plt.plot(ntimeintervalsspectral_list,Cl_err_inf_norm,'o',label="Cl error")
    plt.plot(ntimeintervalsspectral_list,Cm_err_inf_norm,'v',label="Cm error")
    plt.xlabel('ntimeintervalsspectral')
    plt.ylabel('|err|_{max}')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,\
           ncol=2, mode="expand", borderaxespad=0.)

    plt.show()


postProcess([ind_vec_list],[Cl_TS_list],[Cm_TS_list],\
filename2,line_begin,line_end,t_col,Cl_col, Cm_col)

