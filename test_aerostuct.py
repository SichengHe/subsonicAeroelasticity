import numpy as np
from mpi4py import MPI
import copy
import matplotlib.pyplot as plt

from sumb import SUMB
from pywarp import MBMesh
from baseclasses import *



import pyaerots as pyaerots
import pystructts as pystructts
import pyaerostructts as pyaerostructts

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

disp_for_CSD = np.matrix([
 [-0.01962922],
 [-0.01124823],
 [-0.01237087],
 [-0.00726651],
 [-0.0126078 ],
 [-0.00748893]])


# displacement for CFD

disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)



# construct CFD solver

CFDSolver = SUMB(options=aeroOptions)
CFDSolver.addFamilyGroup('full_surface',['wall'])



# construct the TS CFD sovler

TS_CFDSolver = pyaerots.pyAeroTS(CFDSolver,ntimeintervalsspectral,mesh0,mesh_vec,xRot,yRot,ap)
TS_CFDSolver.setUndeformedSurfmesh()
TS_CFDSolver.set_disp_and_deform_and_solve(disp_for_CFD)

# construct the TS CSD solver
TS_CSDSolver = pystructts.pyStructTS(xa,r2a,wa,wh_wa,mu,b, \
    Vf,\
    Ma, omega,
    ntimeintervalsspectral)


TS_CSDSolver.preprocess()

FSISolver = pyaerostructts.pyAeroStructTS(TS_CFDSolver, TS_CSDSolver, ntimeintervalsspectral)
res = FSISolver.set_disp_and_evaluate(disp_for_CSD)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if (rank == 0):

  print "res", res


