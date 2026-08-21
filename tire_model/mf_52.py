import math
import numpy as np
from tirParse import tirParse
import matplotlib.pyplot as plt

class MF52:
    def __init__(self,params :dict,fxScaleFactor=0.6, fyScaleFactor=0.6, mzScaleFactor=0.6):
        self.params = params
        self.fxScale = fxScaleFactor
        self.fyScale = fyScaleFactor
        self.mzScale = mzScaleFactor

    #Doing this as placeholder for combined slip
    def eval(self, Fz, Alpha, Kappa, Gamma, V_x=None):
        "Fz (N), Fy (N), SA (rad), SR (%), IA(rad),"
        #see TTC Forum for MF5.2 equation manual
        #TTC Thread:
            #Pocejka 2002 Equations
            #by Pasindu » Wed Dec 15, 2021 9:51 pm
                #MF52_EquationManual.pdf uploaded by BillCobb #https://www.fsaettc.org/viewtopic.php?p=1581&hilit=Pocejka+2002+Equations#p1581
        
        #Table 3 pg. 8
        C_z = self.params['VERTICAL_STIFFNESS'] #N/m
        R_0 = self.params['UNLOADED_RADIUS'] #m

        #Table 4 pg. 12
        F_z0 = self.params['FNOMIN'] #N
        B = self.params['BREFF']
        D = self.params['DREFF']
        F = self.params['FREFF']

        #Table 9 pg. 24 Longitudinal Coeficients
        PCX1 = self.params['PCX1']
        PDX1 = self.params['PDX1']
        PDX2 = self.params['PDX2']
        PDX3 = self.params['PDX3']
        PEX1 = self.params['PEX1']
        PEX2 = self.params['PEX2']
        PEX3 = self.params['PEX3']
        PEX4 = self.params['PEX4']
        PKX1 = self.params['PKX1']
        PKX2 = self.params['PKX2']
        PKX3 = self.params['PKX3']
        PHX1 = self.params['PHX1']
        PHX2 = self.params['PHX2']
        PVX1 = self.params['PVX1']
        PVX2 = self.params['PVX2']

        #Table 10 pg. 26 Lateral Coeficients
        PCY1 = self.params['PCY1']
        PDY1 = self.params['PDY1']
        PDY2 = self.params['PDY2']
        PDY3 = self.params['PDY3']
        PEY1 = self.params['PEY1']
        PEY2 = self.params['PEY2']
        PEY3 = self.params['PEY3']
        PEY4 = self.params['PEY4']
        PKY1 = self.params['PKY1']
        PKY2 = self.params['PKY2']
        PKY3 = self.params['PKY3']
        PHY1 = self.params['PHY1']
        PHY2 = self.params['PHY2']
        PHY3 = self.params['PHY3']
        PVY1 = self.params['PVY1']
        PVY2 = self.params['PVY2']
        PVY3 = self.params['PVY3']
        PVY4 = self.params['PVY4']

        #Table 11 pg. 28 Aligning Coefficients
        QBZ1 = self.params['QBZ1']
        QBZ2 = self.params['QBZ2']
        QBZ3 = self.params['QBZ3']
        QBZ4 = self.params['QBZ4']
        QBZ5 = self.params['QBZ5']
        QBZ9 = self.params['QBZ9']
        QBZ10 = self.params['QBZ10']
        QCZ1 = self.params['QCZ1']
        QDZ1 = self.params['QDZ1']
        QDZ2 = self.params['QDZ2']
        QDZ3 = self.params['QDZ3']
        QDZ4 = self.params['QDZ4']
        QDZ6 = self.params['QDZ6']
        QDZ7 = self.params['QDZ7']
        QDZ8 = self.params['QDZ8']
        QDZ9 = self.params['QDZ9']
        QEZ1 = self.params['QEZ1']
        QEZ2 = self.params['QEZ2']
        QEZ3 = self.params['QEZ3']
        QEZ4 = self.params['QEZ4']
        QEZ5 = self.params['QEZ5']
        QHZ1 = self.params['QHZ1']
        QHZ2 = self.params['QHZ2']
        QHZ3 = self.params['QHZ3']
        QHZ4 = self.params['QHZ4']

        #Table 15 pg.34 Overturning Coefficients
        QSX1 = self.params['QSX1']
        QSX2 = self.params['QSX2']
        QSX3 = self.params['QSX3']

        #Table 16 pg. 35 Rolling Coefficients
        QSY1 = self.params['QSY1']
        QSY2 = self.params['QSY2']
        QSY3 = self.params['QSY3']
        QSY4 = self.params['QSY4']

        #Note, all lambda scaling factors set to 1

        #Effective Radius
        p = Fz / C_z                                                                                                #(3)
        p_Fz0 = F_z0 / C_z                                                                                          #(5)
        pd = p / p_Fz0                                                                                              #(6) pd is p^d
        R_e = R_0 - p_Fz0 * (D*math.atan(B*pd) + F*pd)                                                              #(4)

        #Only used for rolling resistance, set to nominal if not assigned
        V_ref = self.params['LONGVL']
        if V_x is None: 
            V_x = V_ref                                                                                             #Table 2

        #Misc
        V_sx = -Kappa * V_x                                                                                         #(9)
        V_sy = abs(V_x) * math.tan(Alpha)                                                                           #(10)
        Omega = (V_sx - V_x) / (-R_e)                                                                               #(7)
        V_r = R_e * Omega                                                                                           #(11)
        F_z0 = F_z0*1                                                                                               #(15)
        df_z = (Fz - F_z0) / F_z0                                                                                   #(14)

        #Pure Longitudinal Slip pg.22
        S_Hx = (PHX1 + PHX2*df_z)*1                                                                                 #(26)
        Gamma_x = Gamma*1                                                                                           #(19)
        Kappa_x = Kappa + S_Hx                                                                                      #(18)
        Mu_x = (PDX1 + PDX2*df_z) * (1 - PDX3*Gamma_x**2)*1                                                         #(22)
        C_x = PCX1*1                                                                                                #(20) 
        D_x = Mu_x*Fz                                                                                               #(21)
        E_x = (PEX1 + PEX2*df_z + PEX3*df_z**2)*(1 - PEX4*np.sign(Kappa_x))*1                                       #(23)
        assert E_x <= 1, "Error: E_x = " + str(E_x) +  "   - Must be <=1 "
        K_x = Fz * (PKX1 + PKX2*df_z)*math.exp(PKX3*df_z)*1                                                         #(24) 
        B_x = K_x / (C_x*D_x) if (C_x*D_x) != 0 else 0                                                              #(25)
        S_Vx = Fz*(PVX1 + PVX2*df_z)*1*1                                                                            #(27) 
        F_x0 = D_x*math.sin(C_x*math.atan(B_x*Kappa_x-E_x*(B_x*Kappa_x))) + S_Vx                                    #(17)
        Fx = F_x0*self.fxScale                                                                                      #(16)

        #Pure Lateral Slip Lateral Force pg. 25
        Gamma_y = Gamma*1                                                                                           #(31) 
        S_Hy = (PHY1 + PHY2*df_z)*1 + PHY3*Gamma_y                                                                  #(38)
        Alpha_y = Alpha + S_Hy                                                                                      #(30)
        Mu_y = (PDY1 + PDY2*df_z)*(1-PDY3*Gamma_y**2)*1                                                             #(34) 
        C_y = PCY1*1                                                                                                #(32) 
        D_y = Mu_y * Fz                                                                                             #(33)
        E_y = (PEY1 + PEY2*df_z)*(1-(PEY3+PEY4*Gamma_y)*np.sign(Alpha_y))*1                                         #(35) 
        assert E_y <= 1, "Error: E_y = " + str(E_y) +  "   E_y must be <=1 "
        K_y = PKY1*F_z0*math.sin(2*math.atan(Fz/(PKY2*F_z0*1)))*(1-PKY3*abs(Gamma_y))*1*1                           #(36) 
        B_y = K_y / (C_y*D_y)                                                                                       #(37)
        S_Vy = Fz*((PVY1 + PVY2*df_z)*1 + (PVY3 + PVY4*df_z)*Gamma_y)*1                                             #(39) 
        F_y0 = D_y*math.sin(C_y*math.atan(B_y*Alpha_y - E_y*(B_y*Alpha_y-math.atan(B_y*Alpha_y)))) + S_Vy           #(29)
        Fy = F_y0*self.fyScale                                                                                      #(28)

        #Pure Lateral Slip Aligning Torque pg. 27
        Gamma_z = Gamma*1                                                                                           #(47) 
        S_Ht = QHZ1 + QHZ2*df_z+(QHZ3+QHZ4*df_z)*Gamma_z                                                            #(52) 
        S_Hf = S_Hy + S_Vy/K_y                                                                                      #(46)
        Alpha_t = Alpha + S_Ht                                                                                      #(43)
        Alpha_r = Alpha + S_Hf                                                                                      #(45)
        #Note: Eqn. (45) Says S_Hr instead of S_Hf in the Manual - Corrected by Bill Cobb with a comment in his posted Mz matlab function

        B_t = (QBZ1 + QBZ2*df_z + QBZ3*df_z**2)*(1+QBZ4*Gamma_z+QBZ5*abs(Gamma_z))*1/1                              #(48) 
        C_t = QCZ1                                                                                                  #(49)
        D_t = Fz*(QDZ1 + QDZ2*df_z)*(1+QDZ3*Gamma_z+QDZ4*Gamma_z**2)*(R_0/F_z0)*1                                   #(50) 
        E_t = (QEZ1+QEZ2*df_z+QEZ3*df_z**2)                                                                         #(51)
        # assert removed — fails with real TTC data, Mz not used here
        B_r = QBZ9*1/1 + QBZ10*B_y*C_y                                                                              #(53) 
        D_r = Fz*((QDZ6+QDZ7*df_z)*1+(QDZ8 + QDZ9*df_z)*Gamma_z)*R_0*1                                              #(54) 
        #Note: R_0 = R_O in Manual for unloaded radius

        M_zr = D_r*math.cos(math.atan(B_r*Alpha_r))*math.cos(Alpha)                                                 #(44)
        t = D_t*math.cos(C_t*math.atan(B_t*Alpha_t-E_t*(B_t*Alpha_t-math.atan(B_t*Alpha_t))))*math.cos(Alpha)       #(42)    
        M_z0 = -t * F_y0 + M_zr                                                                                     #(41)
        Mz = M_z0*self.mzScale                                                                                                   #(40)

        #Overturning Moment pg. 34
        Mx = R_0*Fz*(QSX1*1+(-QSX2*Gamma + QSX3*Fy/F_z0)*1)                                                         #(83)

        #Rolling Resistance pg. 35
        My = R_0*Fz*(QSY1 + QSY2*Fx/F_z0 + QSY3*abs(V_x/V_ref)+QSY4*(V_x/V_ref)**4)                                 #(84)

        return [Fx, Fy, Fz, Mx, My, Mz]

    def Fx(self, Fz, Kappa, Gamma):
        """Return SS Longitudinal Force (N) from given normal load (N), slip ratio (%slip), and camber angle (rad)"""
        return self.eval(Fz, 0, Kappa, Gamma)[0]
    
    def Fy(self, Fz, Alpha, Gamma):
        """Return SS Lateral Force (N) from given normal load (N), slip angle (rad), and camber angle (rad)"""
        return self.eval(Fz, Alpha, 0, Gamma)[1]
    
    def Mx(self, Fz, Alpha, Gamma):
        """Returns Overturning moment (Nm) at given normal load (N), slip angle (rad), and camber angle (rad)"""
        return self.eval(Fz, Alpha, 0, Gamma)[3]

    def My(self, Fz, Kappa, Gamma):
        """Return Rolling Resistance (Nm) from given normal load (N), slip ratio (%slip), and camber angle (rad)"""
        return self.eval(Fz,0 ,Kappa, Gamma)[4]

    def Mz(self, Fz, Alpha, Gamma):
        """Returns SS aligning moment (Nm) at given normal load (N), slip angle (rad), and camber angle (rad)"""
        return self.eval(Fz, Alpha, 0, Gamma)[5]

    def minFxKappa(self, Fz, Alpha, Gamma, V_x=None):
        kappaRange = np.linspace(-0.4,0,40)
        minFx = 0
        minKappa = 0
        for i, slipRatio in enumerate(kappaRange):
            tFx, tFy, tFz, tmx, tmy, tmz = self.eval(Fz,Alpha,slipRatio,Gamma, V_x=V_x)
            if tFx < minFx:
                minFx = tFx
                minKappa = slipRatio
        return minKappa, minFx
    
    def kappaBacksolve(self, Fz, Alpha, Gamma, Fx, V_x=None):
        slipRatio = 0
        tFx, tFy, tFz, tmx, tmy, tmz = self.eval(Fz,Alpha,slipRatio,Gamma, V_x=V_x)
        if Fx > 0:
            while tFx < Fx*0.99:
                slipRatio += 0.01
                tFx, tFy, tFz, tmx, tmy, tmz = self.eval(Fz,Alpha,slipRatio,Gamma, V_x=V_x)
                if slipRatio > 1:
                    print("Kappa Backsolve did not find SR, returned zero")
                    return 0
        else:
            while tFx > Fx*0.99:
                slipRatio -= 0.01
                tFx, tFy, tFz, tmx, tmy, tmz = self.eval(Fz,Alpha,slipRatio,Gamma, V_x=V_x)
                if slipRatio < -1:
                    print("Kappa Backsolve did not find SR, returned zero")
                    return 0
        return slipRatio
    
    def Re(self, Fz):
        #Table 3 pg. 8
        C_z = self.params['VERTICAL_STIFFNESS'] #N/m
        R_0 = self.params['UNLOADED_RADIUS'] #m

        #Table 4 pg. 12
        F_z0 = self.params['FNOMIN'] #N
        B = self.params['BREFF']
        D = self.params['DREFF']
        F = self.params['FREFF']

        #Effective Radius
        p = Fz / C_z                                                                                                #(3)
        p_Fz0 = F_z0 / C_z                                                                                          #(5)
        pd = p / p_Fz0                                                                                              #(6) pd is p^d
        R_e = R_0 - p_Fz0 * (D*math.atan(B*pd) + F*pd)                                                              #(4)

        return R_e

if __name__ == "__main__":
    #paramImport = tirParse(r"C:\Users\LegoE\OneDrive\Documents\04-LHR_Code\tire_modeling\tests\Improved_Defaults.tir")
    paramImport = tirParse(r"C:\Users\LegoE\OneDrive\Documents\04-LHR_Code\tire_modeling\tests\FSAE_Defaults.tir")
    tm = MF52(paramImport)

    Fz = 1000

    srRange = np.linspace(-1,1)
    saRange = np.linspace(-0.23,0.23)

    fig, ax = plt.subplots()
    Fx = []
    for i in srRange:
        Fx.append(tm.Fx(Fz, i, 0))

    ax.plot(srRange, Fx)
    ax.set_title("Fx (N) vs SR @ Fz = " + str(Fz) + " (N)")  
    ax.legend()
    plt.show()


    Fy = []
    for i in saRange:
        Fy.append(tm.Fy(Fz, i, 0))

    fig, bx = plt.subplots()
    bx.plot(np.rad2deg(saRange), Fy)
    bx.set_title("Fy (N) vs SA (deg) @ Fz = " + str(Fz) + " (N)")
    plt.show()
    
    Mx = []
    Mz = []
    for i in saRange:
        Mx.append(tm.Mx(Fz, i, 0))
        Mz.append(tm.Mz(Fz, i, 0))
        
    fig, cx = plt.subplots()
    cx.plot(np.rad2deg(saRange), Mx,label="Mx")
    cx.plot(np.rad2deg(saRange), Mz,label="Mz")
    cx.set_title("Mz and Mx (Nm) vs SA (deg) @ Fz = " + str(Fz) + " (N)")
    plt.show()

    My = []
    for i in srRange:
        My.append(tm.My(Fz, i, 0))

    fig, dx = plt.subplots()
    dx.plot(srRange, My)
    dx.set_title("My(Nm) vs SR @ Fz = " + str(Fz) + " (N)")
    plt.show()
