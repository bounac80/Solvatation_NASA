# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:38:11 2026

@author: Paes1
"""
import numpy as np
from sub_model import molar_volume

# gas constant [=] bar.L/(mol K)
R = 8.31446261815324e-2

"""
#==============================================================================  
# Function: psat_pure(T, Tc, Pc, w, L, M, N, c0)  
#==============================================================================  
# This function calculates the saturation pressure (Psat) of a pure compound  
# at a given temperature using an equation of state (EoS) approach.  
# The calculation is performed iteratively using Newton’s method.  
#  
# INPUTS:  
# -- T  : Temperature [K]  
# -- Tc : Critical temperature [K]  
# -- Pc : Critical pressure [bar]  
# -- w  : Acentric factor [-]  
# -- L, M, N, c0 : EOS-specific parameters  
#  
# OUTPUT:  
# -- Psat : Saturation pressure [bar]  
#==============================================================================  
"""
def psat_pure(T,a,b,c,P0):

    #tolerance (stop criteria)
    Tol = 1e-7
    # Calculation of Psat by Newton's method
    Dev_psi = 1e3
    Dev_Psat = 1e3
    P = P0

    # function to calculate fugacity of a pure compound
    def fugPR(a_PR,b_PR,Tk,Pbar,Z,R):
        A = a_PR*Pbar/(R*Tk)**2
        B = b_PR*Pbar/(R*Tk)
        return (Z - 1. - np.log(Z-B) - A/(2*B*np.sqrt(2))*np.log((Z+(1+np.sqrt(2))*B)/(Z+(1-np.sqrt(2))*B)))
    
    while Dev_psi  > Tol or Dev_Psat > Tol:
        # calculation of molar volumes (liq and vap)
        Zliq,Vliq = molar_volume(P,T,a,b,1)
        Zvap,Vvap = molar_volume(P,T,a,b,2)
        # calculation of fugacities and Volumes at T and P (step k)
        lnFUGliq = fugPR(a,b,T,P,Zliq,R)
        lnFUGvap = fugPR(a,b,T,P,Zvap,R)
        # calculation of psi = lnFUGvap - lnFUGliq
        psi = lnFUGvap - lnFUGliq
        
        # check if the solution is in a VLE region.
        # if it's the case, we change the initialization
        if Vvap == Vliq:
            Dev_psi = 1e3
            P0 = 0.9*P0
            P = P0
            P_new = P
        else:
            P_new = P - R*T*(lnFUGvap - lnFUGliq)/(Vvap - Vliq)
        
        if P_new < 0:
            P_new = P*np.exp(-(R*T/P)*(lnFUGvap - lnFUGliq)/(Vvap - Vliq))
        #stop criteria
        Dev_psi = np.max(np.abs(psi))
        Dev_Psat = np.abs(P_new - P)/P
        
        #update value
        P = P_new
        if np.isnan(P):
            Dev_psi = 1e3
            P0 = 0.9*P0
            P = P0
    Psat = P
    return Psat
