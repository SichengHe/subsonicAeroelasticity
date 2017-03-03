# TS CSD solver for aerostructural problem
# It has the following functions:
# (1). Set up the CSD solver
# (2). Serves as the interface with CFD solver:
#      (2.1) takes in load
#      (2.2) outputs the disp


import numpy as np
from sumb import SUMB
import TS as TS

class pyStructTS(object):

    def __init__(self,xa,r2a,wa,wh_wa,mu,b, \
    Vf,\
    Ma, omega,
    ntimeintervalsspectral):

        self.xa = xa
        self.r2a = r2a
        self.wa = wa
        self.wh_wa = wh_wa
        self.mu = mu
        self.b = b

        self.Vf = Vf

        self.Ma = Ma
        self.omega = omega

        self.M = np.matrix([[1.0,self.xa],[self.xa,self.r2a]])
        self.K = np.matrix([[wh_wa**2,0.0],[0.0,r2a]])

        self.ntimeintervalsspectral = ntimeintervalsspectral

        self.pro_size = ntimeintervalsspectral*4


        omega = self.omega
        ntimeintervalsspectral = self.ntimeintervalsspectral
        wa = self.wa
        
        # setup the time dev mat
        TimePeriod = 2.0 * np.pi / omega 

        self.dim_dscalar = TS.timeder_mat(TimePeriod, ntimeintervalsspectral)
        self.dimless_dscalar = self.dim_dscalar/wa

        self.disp_velo_hist_vec = np.matrix(np.zeros((4*ntimeintervalsspectral, 4*ntimeintervalsspectral)))









    def preprocess(self):

        # LHS

        self.set_P_mat()
        self.set_glo_K()
        self.set_glo_M()

        self.set_glo_M_inv()

        # RHS

        self.get_set_MaIndependent_coeff()








    def set_omega(self, omega):

        wa = self.wa

        self.omega = omega

        TimePeriod = 2.0 * np.pi / omega

        self.dim_dscalar = TS.timeder_mat(TimePeriod, ntimeintervalsspectral)
        self.dimless_dscalar = self.dim_dscalar/wa

    def set_omega_and_evaluate(self, omega):

        # Notice: we must re-evaluate CFD to get new f before this step

        self.set_omega(omega)

        self.set_P_mat()

        res_reduced = self.res_reduced()

        return res_reduced


    def set_disp_and_evaluate(self,disp):

        # Notice: we must re-evaluate CFD to get new f before this step

        ntimeintervalsspectral = self.ntimeintervalsspectral
        dimless_dscalar = self.dimless_dscalar

        hb = np.matrix( np.zeros((ntimeintervalsspectral, 1)) )
        alpha = np.matrix( np.zeros((ntimeintervalsspectral, 1)) )

        for i in xrange(ntimeintervalsspectral):

            hb[i, 0] = disp[2*i, 0]
            alpha[i, 0] = disp[2*i+1, 0]

        hb_velo = dimless_dscalar.dot(hb)
        alpha_velo = dimless_dscalar.dot(alpha)

        for i in xrange(ntimeintervalsspectral):

            self.disp_velo_hist_vec[4*i  , 0] = hb[i, 0]
            self.disp_velo_hist_vec[4*i+1, 0] = alpha[i, 0]
            self.disp_velo_hist_vec[4*i+2, 0] = hb_velo[i, 0]
            self.disp_velo_hist_vec[4*i+3, 0] = alpha_velo[i, 0]

        res_reduced = self.res_reduced()

        return res_reduced


    def set_Ma_and_evaluate(self, Ma):

        # Notice: we must re-evaluate CFD to get new f before this step

        self.Ma = Ma

        self.set_scaled_load()

        res_reduced = self.res_reduced()

        return res_reduced




    """ LHS """

    def set_glo_M(self):

        # id_M = |1, 0, 0, 0|
        #        |0, 1, 0, 0|
        #        |0, 0,     |
        #        |0, 0,    M|

        id_M = np.matrix(np.zeros((4,4)))
        id_M[0,0] = 1.0
        id_M[1,1] = 1.0
        id_M[2:4,2:4] = self.M



        # glo_M = |id_M    0  ...    0|
        #         |0    id_M  ...  ...|
        #         |...  ...   ...    0|
        #         |0    ...     0 id_M|

        self.glo_M = np.matrix(np.zeros((self.ntimeintervalsspectral*4,self.ntimeintervalsspectral*4)))
        for i in range(self.ntimeintervalsspectral):
            self.glo_M[i*4:(i+1)*4,i*4:(i+1)*4] = id_M

    def set_glo_M_inv(self):
        self.glo_M_inv = np.linalg.inv(self.glo_M)

    def set_glo_K(self):

        # id_K = |0, 0, -1, 0|
        #        |0, 0, 0, -1|
        #        |      0,  0|
        #        |   K, 0,  0|

        id_K = np.matrix(np.zeros((4,4)))
        id_K[0,2] = -1.0
        id_K[1,3] = -1.0
        id_K[2:4,0:2] = self.K

        # glo_K = |id_K    0  ...    0|
        #         |0    id_K  ...  ...|
        #         |...  ...   ...    0|
        #         |0    ...     0 id_K|

        self.glo_K = np.matrix(np.zeros((self.ntimeintervalsspectral*4,self.ntimeintervalsspectral*4)))
        for i in range(self.ntimeintervalsspectral):
            self.glo_K[i*4:(i+1)*4,i*4:(i+1)*4] = id_K

    def set_permutation(self):

        self.S = np.matrix(np.zeros((4*self.ntimeintervalsspectral,4*self.ntimeintervalsspectral)))
        for i in range(4):
            for j in range(self.ntimeintervalsspectral):
                self.S[i*self.ntimeintervalsspectral+j,j*4+i] = 1.0
        self.ST = np.transpose(self.S)


    def set_glo_D(self):

        self.glo_D = np.matrix(np.zeros((self.ntimeintervalsspectral*4,self.ntimeintervalsspectral*4)))

        for i in range(4):
            self.glo_D[i*self.ntimeintervalsspectral:(i+1)*self.ntimeintervalsspectral,\
            i*self.ntimeintervalsspectral:(i+1)*self.ntimeintervalsspectral] = self.dimless_dscalar


    def set_P_mat(self):

        self.set_permutation()
        self.set_glo_D()

        self.P_mat = self.ST.dot(self.glo_D).dot(self.S)





    """ RHS """

    def get_set_MaIndependent_coeff(self):

        Vf = self.Vf
        Ma = self.Ma

        Ma_Indepdendent_coeff = (Vf**2/np.pi) / Ma**2  

        self.Ma_Indepdendent_coeff = Ma_Indepdendent_coeff

    def set_scaled_load(self):

        Ma = self.Ma
        Ma_Indepdendent_coeff = self.Ma_Indepdendent_coeff
        coeff = Ma_Indepdendent_coeff * Ma**2

        Cl_coeff = -coeff
        Cm_coeff = 2.0 * coeff

        self.f_scaled = np.matrix(np.zeros((self.pro_size,1)))

        for i in xrange(self.ntimeintervalsspectral):

            self.f_scaled[4*i+2,0] = Cl_coeff*self.f[2*i]
            self.f_scaled[4*i+3,0] = Cm_coeff*self.f[2*i+1]


    def res(self):

        "get the residual of original eqn: [M][P]w+[K]w=f"

        disp_velo_hist_vec_der = self.P_mat.dot(self.disp_velo_hist_vec)
        self.Res = self.glo_M.dot(disp_velo_hist_vec_der) + self.glo_K.dot(self.disp_velo_hist_vec)-self.f_scaled

        self.Res_norm = np.sqrt(np.transpose(self.Res).dot(self.Res)[0,0])

    def res_reduced(self):

        "smaller size res: drop the part that define dot(u) = D u since this part is forced to be true"

        self.res()
        Res = self.Res
        ntimeintervalsspectral = self.ntimeintervalsspectral

        res_reduced = np.matrix(np.zeros((2*ntimeintervalsspectral, 1)))

        for i in xrange(ntimeintervalsspectral):
            res_reduced[2*i,   0] = Res[4*i + 2, 0]
            res_reduced[2*i+1, 0] = Res[4*i + 3, 0]

        return res_reduced








    ## interface  ##########################
    # info <- CFD solver
    def set_load(self,f):
        # f: [Cl_0,Cm_0,...,Cl_{Ts-1},Cm_{Ts-1}]
        self.f = f


    # info -> CFD solver\
    def get_disp_4CFD(self):

        # disp_vec: [h_0/b, alpha_0,...,h_{Ts-1}/b,alpha_{Ts-1}]
        self.disp_vec = np.matrix(np.zeros((self.ntimeintervalsspectral*2,1)))

        for i in xrange(self.ntimeintervalsspectral):

            self.disp_vec[2*i,0] = self.disp_velo_hist_vec[4*i,0]
            self.disp_vec[2*i+1,0] = self.disp_velo_hist_vec[4*i+1,0]



    ## a direct solver (for validation usage)

    test_mode = 1

    if (test_mode):

        def set_G_mat(self):

            self.set_G_mat_imp()


        def set_G_mat_imp(self):
            # matrix used to solve for new disp
            # MPw+Kw=f
            # implicit scheme:
            # G = inv(I+dt(P+inv(M)K))

            dt = 10.0

            self.set_P_mat()
            self.set_glo_K()
            self.set_glo_M()

            self.set_glo_M_inv()

            pro_size = self.pro_size
            glo_K = self.glo_K
            glo_M_inv = self.glo_M_inv
            P_mat = self.P_mat



            I_mat = np.matrix(np.identity(pro_size))

            G_mat = P_mat+glo_M_inv.dot(glo_K)
            G_mat *= dt
            G_mat += I_mat
            G_mat = np.linalg.inv(G_mat)
            
            self.G_mat = G_mat


        def struct_one_iter(self):

            dt = 10.0

            # implicit

            disp_velo_hist_vec = self.disp_velo_hist_vec
            glo_M_inv = self.glo_M_inv
            f_scaled = self.f_scaled
            G_mat = self.G_mat



            prev_step_f = disp_velo_hist_vec + dt*glo_M_inv.dot(f_scaled)
            disp_velo_hist_vec = G_mat.dot(prev_step_f)

            self.disp_velo_hist_vec = disp_velo_hist_vec


        def struct_solver(self, f):

            self.set_load(f)

            self.set_scaled_load()

            self.struct_one_iter()


        def solve(self, f):

            self.set_G_mat()

            ntimeintervalsspectral = self.ntimeintervalsspectral

            self.disp_velo_hist_vec = np.matrix(np.zeros((4*ntimeintervalsspectral, 1)))

            self.get_set_MaIndependent_coeff()

            for i in xrange(100):

                self.struct_solver(f)

                self.res()

                #print "Res_norm", self.Res_norm

            self.get_disp_4CFD()















