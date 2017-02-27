import numpy as np
from mpi4py import MPI
from baseclasses import *
from sumb import SUMB
from pywarp import MBMesh
import copy

class pyAeroTS(object):

    def __init__(self,aeroSolver,ntimeintervalsspectral,mesh0,mesh_vec,xRot,yRot,ap):

        self.aeroSolver = aeroSolver
        self.ntimeintervalsspectral = ntimeintervalsspectral

        self.mesh0 = mesh0 # undeformed mesh
        self.mesh_vec = mesh_vec

        self.xRot = xRot # rotation center x coord
        self.yRot = yRot # rotation center y coord

        self.ap = ap # aero problem

        self.aeroSolver.setMesh(self.mesh0)

    def getLoad(self):

        # [Cl_0,Cm_0,...,Cl_{Ts-1},Cm_{Ts-1}]
        # notice here we take into account that Sumb gives the Cm in the opposite direction
        # as in aeroelasticity -- so we convert the cm to be in accordance with aeroelasticity
        # convention.

        self.load_vec = []

        for sps in xrange(self.ntimeintervalsspectral):
            sol = self.aeroSolver.getSolution(sps+1)

            cm = sol['cmz']
            cl = sol['cl']

            self.load_vec.append(cl)
            self.load_vec.append(-cm)

        self.load_vec = np.transpose( np.matrix(self.load_vec) )


    # info <- CSD solver
    def set_disp_and_deform_and_solve(self, disp_hist):

        self.setDispHist(disp_hist)
        self.spectral_mesh_warping()

        self.solve()

        return self.load_vec

    def set_omega_and_solve(self, omega):

        self.ap.omegaFourier = omega

        self.solve()
        return self.load_vec

    def set_mach_and_solve(self, mach):

        self.ap.mach = mach

        self.solve()
        return self.load_vec

    def set_disp_mach_omega_and_solve(self, disp_hist, mach, omega):

        self.setDispHist(disp_hist)
        self.spectral_mesh_warping()

        self.ap.mach = mach

        self.ap.omegaFourier = omega

        self.solve()
        
        return self.load_vec






    def setDispHist(self,disp_hist):

        # [h_0, alpha_0,...,h_{Ts-1},alpha_{Ts-1}]

        self.h_vec = []
        self.alpha_vec = []

        for i in xrange(self.ntimeintervalsspectral):

            self.h_vec.append(disp_hist[2*i, 0])
            self.alpha_vec.append(disp_hist[2*i+1, 0])

    ## mesh warping
    def spectral_mesh_warping(self):

        #self.aeroSolver.setMesh(self.mesh0)

        for i in xrange(self.ntimeintervalsspectral):

            ptch_loc = self.alpha_vec[i]
            plg_loc = self.h_vec[i] # positive downwards

            new_coords = self.coords0.copy()

            cc = np.cos(ptch_loc)
            ss = np.sin(ptch_loc)

            for j in xrange(len(self.coords0)):
                new_coords[j,0] =   cc*(self.coords0[j,0] - self.xRot) + ss* self.coords0[j,1] + self.xRot
                new_coords[j,1] = - ss*(self.coords0[j,0] - self.xRot) + cc* self.coords0[j,1] - plg_loc

            self.aeroSolver.mesh.setSurfaceCoordinates(new_coords)

            self.aeroSolver.mesh.warpMesh()

            m = self.aeroSolver.mesh.getSolverGrid()

            self.aeroSolver.setGridSpectralWarp(m,sps=i+1)


        flag = True
        self.aeroSolver.setTSmanualWarp(flag)


    def setUndeformedSurfmesh(self):

        # set the neutral angle of attack (with spring relaxed)

        self.coords0 = self.mesh0.getSurfaceCoordinates()

    def solve(self):
        self.aeroSolver(self.ap)

        self.getLoad()
        return self.load_vec
