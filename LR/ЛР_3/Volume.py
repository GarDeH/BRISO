import numpy as np

class Volume():
    def __init__(self, INIT):
        self.D_n = INIT[0]
        self.l_ob = INIT[1]
        self.delta_ob = INIT[2]
        self.delta_zks = INIT[3]
        self.b_1 = INIT[4]
        self.b_2 = INIT[5]
        self.delta_dn1 = INIT[6]
        self.delta_dn2 = INIT[7]
        self.delta_p1 = INIT[8]
        self.delta_p2 = INIT[9]
        self.d = INIT[10]
        self.D_vh = INIT[11]
        self.D_vih = INIT[12]
        self.D_kr = INIT[13]
        self.tetta_1 = INIT[14]
        self.tetta_2 = INIT[15]
        self.l_kr = INIT[16]
        self.delta_s1 = INIT[17]
        self.delta_s2 = INIT[18]
        self.delta_st = INIT[19]
        self.delta_ps1 = INIT[20]
        self.delta_ps2 = INIT[21]
        self.delta_vks = INIT[22]

    def cil_ks(self):
        d_ob = self.D_n - 2 * self.delta_ob
        S_ob = np.pi / 4 * (self.D_n ** 2 - d_ob ** 2)
        V_ob = S_ob * self.l_ob
        d_zks = d_ob - 2 * self.delta_zks
        S_zks = np.pi / 4 * (d_ob ** 2 - d_zks ** 2)
        V_zks = S_zks * self.l_ob
        return np.array([V_ob, V_zks])

    def ell_pd(self):
        V_dn1 = 2 / 3 * np.pi * self.delta_dn1 * \
            (self.D_n**2/4+self.D_n*self.b_1-(self.D_n+self.b_1)
             * self.delta_dn1 + self.delta_dn1**2)
        V_p1 = 2 / 3 * np.pi * self.delta_p1 * ((self.D_n - 2 * self.delta_dn1)**2/4+(
            self.D_n-2*self.delta_dn1)*(self.b_1-self.delta_dn1)-(self.D_n - 2*self.delta_dn1+self.b_1 - self.delta_dn1) * self.delta_p1 + self.delta_p1**2)
        return np.array([V_dn1, V_p1])

    def ell_sd(self):
        V_dn2 = np.pi / 6 * self.D_n**2*self.b_1*(1-(self.d/self.D_n)**2)**(3/2) - np.pi / 6 * (
            self.D_n - 2 * self.delta_dn1)**2*(self.b_1-self.delta_dn1)*(1 - self.d**2 / (self.D_n - 2 * self.delta_dn1)**2)**(3/2)
        V_p2 = np.pi / 6 * (self.D_n - 2 * self.delta_dn1)**2*(self.b_1-self.delta_dn1)*(1-(self.d/(self.D_n - 2*self.delta_dn1))**2)**(3/2) - np.pi / 6 * (
            (self.D_n - 2*self.delta_dn1) - 2 * self.delta_p1)**2*(self.b_1 - self.delta_dn1-self.delta_p1)*(1 - self.d**2 / (self.D_n - 2 * self.delta_dn1 - 2*self.delta_p1)**2)**(3/2)
        return np.array([V_dn2, V_p2])

    def cil_st(self):
        d_st = self.D_kr + 2 * self.delta_vks
        S_st = np.pi / 4 * ((self.D_kr + 2 * self.delta_st +
                            2 * self.delta_vks) ** 2 - d_st ** 2)
        V_st = S_st * self.l_kr
        D_vks = self.D_kr + 2 * self.delta_vks
        S_vks = np.pi / 4 * (D_vks ** 2 - self.D_kr ** 2)
        V_vks = S_vks * self.l_kr
        return np.array([V_st, V_vks])

    def kon_1(self):
        l_kon_1 = (self.D_vh - self.D_kr) / 2 / np.tan(self.tetta_1)
        V_s1 = np.pi * l_kon_1 * self.delta_s1 / np.cos(self.tetta_1) * (
            (self.D_vh + 2 * self.delta_ps1 / np.cos(self.tetta_1) + self.D_kr + 2 * self.delta_ps1 / np.cos(self.tetta_1)) / 2 + self.delta_s1 / np.cos(self.tetta_1))
        V_ps1 = np.pi * l_kon_1 * self.delta_ps1 / \
            np.cos(self.tetta_1) * ((self.D_vh + self.D_kr) /
                                    2 + self.delta_ps1 / np.cos(self.tetta_1))
        return np.array([V_s1, V_ps1])

    def kon_2(self):
        l_kon_2 = (self.D_vih - self.D_kr) / 2 / np.tan(self.tetta_2)
        V_s2 = np.pi * l_kon_2 * self.delta_s2 / np.cos(self.tetta_2) * (
            (self.D_vih + 2 * self.delta_ps2 / np.cos(self.tetta_2) + self.D_kr + 2 * self.delta_ps2 / np.cos(self.tetta_2)) / 2 + self.delta_s2 / np.cos(self.tetta_2))
        V_ps2 = np.pi * l_kon_2 * self.delta_ps2 / np.cos(self.tetta_2) * (
            (self.D_vih + self.D_kr) / 2 + self.delta_ps2 / np.cos(self.tetta_2))
        return np.array([V_s2, V_ps2])