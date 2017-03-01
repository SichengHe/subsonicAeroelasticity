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


def sens(Ma, omega, disp_for_CSD, TS_CFDSolver, b, ntimeintervalsspectral):

    epsilon_Ma = 1e-7
    epsilon_omega = 1e-7
    epsilon = 1e-4


    # Ma

    Ma_perturbed_plus = (1 + epsilon_Ma) * Ma

    loadCFD_perturbed_plus = TS_CFDSolver.set_mach_and_solve(Ma_perturbed_plus)

    Ma_perturbed_minus = (1 - epsilon_Ma) * Ma

    loadCFD_perturbed_minus = TS_CFDSolver.set_mach_and_solve(Ma_perturbed_minus)

    dR_o_dMa = (loadCFD_perturbed_plus - loadCFD_perturbed_minus) / (Ma_perturbed_plus - Ma_perturbed_minus)


    # reset
    TS_CFDSolver.set_mach_and_solve(Ma)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if (rank == 0):
        print "++++++++++++++++++++++++1", TS_CFDSolver.ap.mach, TS_CFDSolver.ap.omegaFourier


    # omega

    omega_perturbed_plus = (1 + epsilon_omega) * omega

    loadCFD_perturbed_plus = TS_CFDSolver.set_omega_and_solve(omega_perturbed_plus)

    omega_perturbed_minus = (1 - epsilon_omega) * omega

    loadCFD_perturbed_minus = TS_CFDSolver.set_omega_and_solve(omega_perturbed_minus)

    dR_o_domega = (loadCFD_perturbed_plus - loadCFD_perturbed_minus) / (omega_perturbed_plus - omega_perturbed_minus)

    # reset
    TS_CFDSolver.set_omega_and_solve(omega)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if (rank == 0):
        print "++++++++++++++++++++++++2", TS_CFDSolver.ap.mach, TS_CFDSolver.ap.omegaFourier



    # disp

    dLoad_dDisp = sens_disp(epsilon, disp_for_CSD, b, ntimeintervalsspectral, TS_CFDSolver)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if (rank == 0):
        print "++++++++++++++++++++++++3",TS_CFDSolver.ap.mach, TS_CFDSolver.ap.omegaFourier


    # assemble matrix

    dR_dx = np.matrix( np.zeros((2*ntimeintervalsspectral, 2*ntimeintervalsspectral)) )
    dR_dx[:, 2:] = dLoad_dDisp[:, 2:]
    dR_dx[:, 0] = dR_o_dMa
    dR_dx[:, 1] = dR_o_domega

    

    return dR_dx

def Newton_step_total(res, disp_for_CSD, Ma, omega, TS_CFDSolver, b, ntimeintervalsspectral):

  dRdx = sens(Ma, omega, disp_for_CSD, TS_CFDSolver, b, ntimeintervalsspectral)

  dvar = np.linalg.solve(dRdx, -res)

  d_disp_for_CSD = np.matrix(np.zeros((2 * ntimeintervalsspectral, 1)))

  d_disp_for_CSD[2:, 0] = dvar[2:, 0]
  dMa = dvar[0, 0]
  domega = dvar[1, 0]

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  if (rank == 0):

    print "dRdx", dRdx, "res", res, "d_disp_for_CSD", d_disp_for_CSD, "dMa", dMa, "domega", domega

  return dMa, domega, d_disp_for_CSD

def Newton_bigger(disp_for_CSD, Ma, omega, loadCFD_dest):

  global TS_CFDSolver
  global ntimeintervalsspectral
  global b

  res_norm_log_list = []

  for i in xrange(10):

    disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)
    loadCFD = TS_CFDSolver.set_disp_mach_omega_and_solve(disp_for_CFD, Ma, omega)

    res = loadCFD - loadCFD_dest
    res_norm = np.sqrt(np.transpose(res).dot(res)[0, 0])
    res_norm_log_list.append(np.log10(res_norm))

    dMa, domega, d_disp_for_CSD = Newton_step_total(res, disp_for_CSD, Ma, omega, TS_CFDSolver, b, ntimeintervalsspectral)

    #break

    if (i<=3):

      Ma += dMa

    else:

      disp_for_CSD += d_disp_for_CSD
      Ma += dMa
      omega += domega

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if (rank == 0):

      print "dMa", dMa, "domega", domega, "d_disp_for_CSD", d_disp_for_CSD




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

disp_for_CSD = np.matrix([
 [-0.01973829],
 [-0.01128884],
 [-0.01102459],
 [-0.00653375],
 [-0.01636958],
 [-0.0095091 ]])



# displacement for CFD

disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)



# construct CFD solver

CFDSolver = SUMB(options=aeroOptions)
CFDSolver.addFamilyGroup('full_surface',['wall'])



# construct the TS CFD sovler

TS_CFDSolver = pyaerots.pyAeroTS(CFDSolver,ntimeintervalsspectral,mesh0,mesh_vec,xRot,yRot,ap)
TS_CFDSolver.setUndeformedSurfmesh()
loadCFD_dest = TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD)






# Test 1: under disp history under Ma, omega, we solve for the load; 
# then we perturb the disp but keep Ma and omega unchanged. We are going to search for the original disp.

disp_for_CSD_perturbed = np.matrix([
 [-0.01973829],
 [-0.01128884],
 [-0.01102459+0.0003],
 [-0.00653375+0.0004],
 [-0.01636958+0.0005],
 [-0.0095091+0.0006 ]])


Ma_perturbed = Ma + 0.01
omega_perturbed = omega + 5.0

Newton_bigger(disp_for_CSD_perturbed, Ma_perturbed, omega_perturbed, loadCFD_dest)










