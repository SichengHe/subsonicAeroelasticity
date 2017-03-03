import numpy as np

import pystructts
import pyaerots

b = 0.5

def dispCSD2dispCFD(dispCSD, b, ntimeintervalsspectral):

    dispCFD = np.matrix(np.zeros((2 * ntimeintervalsspectral, 1)))

    for i in xrange(ntimeintervalsspectral):

        dispCFD[2 * i,    0] = dispCSD[2 * i,     0] * b
        dispCFD[2 * i +1, 0] = dispCSD[2 * i + 1, 0]

    return dispCFD


class pyAeroStructTS(object):


    def __init__(self, TS_aero_prb, TS_struct_prb,ntimeintervalsspectral):

        # two solvers
        self.TS_aero_prb = TS_aero_prb
        self.TS_struct_prb = TS_struct_prb

        # fundamental settings
        self.ntimeintervalsspectral = ntimeintervalsspectral


    def set_disp_and_evaluate(self, disp_for_CSD):

        global b

        TS_aero_prb = self.TS_aero_prb
        TS_struct_prb = self.TS_struct_prb
        ntimeintervalsspectral = self.ntimeintervalsspectral

        disp_for_CFD = dispCSD2dispCFD(disp_for_CSD, b, ntimeintervalsspectral)

        f = TS_aero_prb.set_disp_and_deform_and_solve(disp_for_CFD)
        TS_struct_prb.set_load(f)
        TS_struct_prb.set_scaled_load()

        res = TS_struct_prb.set_disp_and_evaluate(disp_for_CSD)

        return res

    def set_omega_and_evaluate(self, omega):

        TS_aero_prb = self.TS_aero_prb
        TS_struct_prb = self.TS_struct_prb

        f = TS_aero_prb.set_omega_and_solve(omega)
        TS_struct_prb.set_load(f)
        TS_struct_prb.set_scaled_load()

        res = TS_struct_prb.set_omega_and_evaluate(omega)

        return res

    def set_Ma_and_evaluate(self, Ma):

        TS_aero_prb = self.TS_aero_prb
        TS_struct_prb = self.TS_struct_prb

        f = TS_aero_prb.set_mach_and_solve(Ma)
        TS_struct_prb.set_load(f)

        res = TS_struct_prb.set_Ma_and_evaluate(Ma)

        return res


    def res(self):

        TS_aero_prb = self.TS_aero_prb
        TS_struct_prb = self.TS_struct_prb

        f = TS_aero_prb.solve()

        TS_struct_prb.set_load(f)
        TS_struct_prb.set_scaled_load()

        res = TS_struct_prb.res_reduced()

        return res

















