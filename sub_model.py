# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:22:49 2026

@author: Paes1
"""

import numpy as np

#************************************
#************************************
# Peng-Robinson EoS
#************************************
#************************************

# gas constant [=] bar.L/(mol K)
R = 8.31446261815324e-2;

"""
#==============================================================================
# Function: a,b,c = par_twu91(T,Tc,Pc,w,L,M,N):
#==============================================================================
# Calculation of EoS parameters of pure compounds using critical properties, 
# acentric factors, and Twu91 parameters, based on the Twu-91 alpha function
# INPUTS
# -- T = temperature in K: real[1,1]
# -- Tc = critical temperature in K: real[1,1]
# -- Pc = critical pressure in bar: real[1,1]
# -- w = acentric factor: real[1,1]
# -- L, M, and N = Twu91 alpha-function parameters (if available): real[1,1]
# OUTPUTS
# -- a = attractive parameter at T: real[NC,1]
# -- b = co-volume: real[NC,1]
# -- c = volume translation: real[NC,1]
# NOTES
# (1) If L, M, and N are not available, make than equal to 8888
#     In this case, a generalized correlation will be used instead
# -- L = 8888
# -- M = 8888
# -- M = 8888
# (2) Similarly, the volume translation is calculated by a generalized correlation
#     if c0 = 8888
"""
def par_twu91(T,Tc,Pc,w,L,M,N,c0):
    
    # Reduced temperature
    Tr = T / Tc
    # Calculation of Twu91 alpha-function at Tr
    if L == 8888 or M == 8888 or N == 8888:
        L = 0.0544 + 0.7536 * w + 0.0297 * w**2
        M = 0.8678 - 0.1785 * w + 0.1401 * w**2
        N = 2
    alpha = Tr**(N * (M - 1)) * np.exp(L * (1 - Tr**(M * N)))
    # Attractive parameter a(T)
    ac = 0.45724 * (R * Tc)**2 / Pc
    a = ac * alpha
    # Co-volume (b)
    b = 0.0778 * R * Tc / Pc
    # volume translation (c)
    if c0 == 8888:
        F = -0.014471 + 0.067498 * w - 0.084852 * w**2 + 0.067298 * w**3 - 0.017366 * w**4
        c = (R * Tc / Pc) * F
    else:
        c = c0
    return a, b, c

#----------------
# function used for entropy scaling
def Tder_twu(T,Tc,Pc,w,L,M,N):

    # reduced temperature
    Tr = T/Tc
    # Calculation of Twu91 alpha-function at Tr
    if L == 8888 or M == 8888 or N == 8888:
        L = 0.0544 + 0.7536 * w + 0.0297 * w**2
        M = 0.8678 - 0.1785 * w + 0.1401 * w**2
        N = 2
    alpha = Tr**(N * (M - 1)) * np.exp(L * (1 - Tr**(M * N)))
    # Attractive parameter at Tc
    ac = 0.45724 * (R * Tc)**2 / Pc
    # 1st derivative of the attractive parameter wrt Temperature
    coeff_0 = N*(M-1)
    coeff_1 = - (L*M*N)
    tau = Tr**(M*N)
    dalpha_dTr = (alpha/Tr) * (coeff_0 + coeff_1*tau)
    da_dT = (ac/Tc) * dalpha_dTr
    return da_dT

"""
#==============================================================================
# Function: a,b,c = par_soave72(T,Tc,Pc,w):
#==============================================================================
# Calculation of EoS parameters of pure compounds using critical properties, 
# acentric factors, considering the soave alpha function
# INPUTS
# -- T = temperature in K: real[1,1]
# -- Tc = critical temperature in K: real[1,1]
# -- Pc = critical pressure in bar: real[1,1]
# -- w = acentric factor: real[1,1]
# OUTPUTS
# -- a = attractive parameter at T: real[NC,1]
# -- b = co-volume: real[NC,1]
# -- c = volume translation: real[NC,1]
# NOTES
# (1) Here again the volume translation is calculated by a generalized correlation.
#     This value can be used if a optimal value for c is not available
"""
def par_soave72(T,Tc,Pc,m):

    # Reduced temperature
    Tr = T / Tc
    # Calculation of Twu91 alpha-function at Tr
    alpha = (1+m*(1-np.sqrt(Tr)))**2
    # Attractive parameter a(T)
    ac = 0.45724 * (R * Tc)**2 / Pc
    a = ac * alpha
    # Co-volume (b)
    b = 0.0778 * R * Tc / Pc
    # volume translation (c)
    c = 0.
    return a, b, c

#----------------
# function used for entropy scaling
def Tder_soave(T, Tc, Pc, m):
    # a_c
    ac = 0.45724 * (R * Tc)**2 / Pc
    # f(T)
    f = 1 + m * (1 - np.sqrt(T / Tc))
    # da/dT
    da_dT = -ac * m * f / np.sqrt(T * Tc)
    return da_dT

"""
#==============================================================================
# Function: (a,b) = par_soaveGC(T,occ,par_gc)
#==============================================================================
# Calculation of EoS parameters of pure compounds
# INPUTS
# -- T = temperature in K: real[1,1]
# -- occ = occurances matrix of UNIFAC groups: real[NC,Ngr]
# -- par_gc = group contributions real[Ngr,1]
#             par_gc.b = contributions for the co-volume: real[Ngr,1]
#             par_gc.ac = contributions for the atractive parameter at T = Tc: real[Ngr,1]
#             par_gc.a0 = contributions for the atractive parameter at T = 0K: real[Ngr,1]
# OUTPUTS
# -- a = attractive parameter at T: real[NC,1]
# -- b = co-volume: real[NC,1]
"""
def par_soaveGC(T,occ,ceosGC):

    # CEoS constants
    omega_a = 0.45724
    omega_b = 0.0778
    OmhR_a = omega_a*R**2
    OmhR_b = omega_b*R
    # expoents
    expb = 0.80
    expac = 0.67
    expa0 = 0.48
    ctes_b = ceosGC[:,0]
    ctes_b = ctes_b.reshape((len(ctes_b), 1))
    ctes_ac = ceosGC[:,1]
    ctes_ac = ctes_ac.reshape((len(ctes_ac), 1))
    ctes_a0 = ceosGC[:,2]
    ctes_a0 = ctes_a0.reshape((len(ctes_a0), 1))
    # GC calculations for b, ac, and a0
    b  = OmhR_b*(np.matmul(np.transpose(occ),ctes_b))**(1/expb)
    ac = OmhR_a*(np.matmul(np.transpose(occ),ctes_ac))**(1/expac)
    a0 = OmhR_a*(np.matmul(np.transpose(occ),ctes_a0))**(1/expa0)
    # critical constants
    Tc = (omega_b/omega_a*1/R)*(ac/b)
    # soave alpha function
    m = np.sqrt(((a0))/(ac)) - 1
    alpha = (1 + m*(1 - np.sqrt(T/Tc)))**2
    # atractive parameter at T
    a = ac*alpha
    # volume translation = 0
    c = 0.
    return a, b, c

"""
#==============================================================================
# Function: a_mix, b_mix, c_mix, abRT_mix, sum_zb, der_abRT_mix = mix_rule(T,z,lnGam,a,b,c,q1,s)
#==============================================================================
# Calculation of the CEoS parameters for a mixture at a given temperature and composition
# INPUTS
# -- T = temperature in K: real[1,1]
# -- z = composition of the system: real[NC,1]
# -- lnGam = ln of activity coefficients: real[NC,1]
# -- a = pure compounds attractive parameter at T: real[NC,1]
# -- b = pure compounds co-volume: real[NC,1]
# -- c = pure compounds volume translation: real[NC,1]
# -- q1 and s = mixing rule constants
# OUTPUTS
# -- a_mix = mixture attractive parameter at T: real[1,1]
# -- b_mix = mixture co-volume in L/mol: real[NC,1]
# -- c_mix = volume translation of the mixture in L/mol: real[1,1]
# -- abRT_mix = a_mix(T)/(b_mix*R*T): real[1,1]
# -- sum_zb = linear form associated with the quadratic function bmix: real[NC,1]
# -- der_abRT_mix = Derivative of abRT_mix wrt. z(i): real[NC,1]
"""
def mix_rule(T,z,lnGam,a,b,c,q1,s):

    # constant mixing rule (HV)
    NC = len(z)
    # volume translation (c_mix)
    c_mix = np.dot(np.transpose(z),c)
    bij = np.zeros((NC, NC))
    for i in range(NC):
        for j in range(i, NC):
            bij[i, j] = ((b[i].item() ** (1 / s) + b[j].item() ** (1 / s)) / 2) ** s
            if i != j:
                bij[j, i] = bij[i, j]
    b_mix = np.dot(np.matmul(np.transpose(z), bij), z)
    # Linear form associated with the quadratic function bmix
    sum_zb = np.transpose(np.dot(np.transpose(z), bij))
    # attractive parameter
    Ge_RT = np.dot(np.transpose(z),lnGam)
    abRT = a/(b*R*T)
    sum_abRT = np.dot(np.transpose(z),abRT)
    abRT_mix = Ge_RT/q1 + sum_abRT
    a_mix = abRT_mix*(b_mix*R*T)
    # Derivative of abRT_mix wrt. z(i)
    der_abRT_mix = abRT + lnGam / q1
    return a_mix.item(), b_mix.item(), c_mix.item(), abRT_mix.item(), sum_zb, der_abRT_mix

"""
#==============================================================================
# Function: Z_res, V_res = molar_volume(P,T,a_mix,b_mix)
#==============================================================================
# Function that solves the cubic EoS for the molar volume 
# The volume translation is not taken into account here...
# INPUTS
# -- P = pressure in bar: real[1,1]
# -- T = temperature in K: real[1,1]
# -- a = mixture attractive parameter at T: real[NC,1]
# -- b = mixture co-volume: real[NC,1]
# OUTPUTS
# -- Z_res = compressibility factor: real[NC,1]
# -- V_res = molar volume of the mixture in L/mol: real[NC,1]
# Calculation of EoS parameters of mixtures
"""
def molar_volume(P,T,a_mix,b_mix,phase):

    # Dimensionless EoS parameters
    A = a_mix *P/(R*T)**2
    B = b_mix *P/(R*T)
    # Compressibility factor
    coeff1 = float(1.)
    coeff2 = float(-(1. - B))
    coeff3 = float(A - 3. * B**2 - 2. * B)
    coeff4 = float(-(A * B - B**2 - B**3))
    coefficients = [coeff1, coeff2, coeff3, coeff4]
    Z = np.roots(coefficients)
    ZR = np.array([z.real for z in Z if np.isreal(z) and z.real > 0])
    
    if phase == 1:
        # Liquid phase
        Z_res = np.min(ZR)

    elif phase == 2:
        # Vapor phase
        Z_res = np.max(ZR)
    
    elif phase == 0:
        # most stable root = lower value for the fugacity coefficient of the mixture (phi)
        phi = np.exp((ZR - 1) - np.log(ZR - B) - A/(2 * B * np.sqrt(2)) * np.log((ZR + (1 + np.sqrt(2)) * B)/(ZR + (1 - np.sqrt(2)) * B)))
        idx = np.argmin(phi)
        Z_res = float(ZR[idx])
        
    V_res = float((R * T / P) * Z_res)
    return Z_res.real, V_res.real

"""
#==============================================================================
# Function: FUG = fugacity(P, T, V_res, b_mix, sum_zb, der_abRT_mix, c_mix)
#==============================================================================
# Function that calculates the fugacity of solute in the mixture
# INPUTS
# -- P = pressure in bar: real[1,1]
# -- T = temperature in K: real[1,1]
# -- V_res = molar volume of the mixture in L/mol: real[NC,1]
# -- sum_zb = linear form associated with the quadratic function bmix: real[NC,1]
# -- der_abRT_mix = Derivative of abRT_mix wrt. z(i): real[NC,1]
# -- c_mix = volume translation of the mixture in L/mol: real[1,1]
# OUTPUTS
# -- FUG = ln of fugacity coefficients of a compound in a mixture : real[NC,1]
"""
def fugacity(P, T, V_res, b_mix, sum_zb, der_abRT_mix, c_mix):

    # ln(fugacity coefficient) = FUG
    F1 = np.log(R * T / (P * (V_res - b_mix)))
    F2 = (2 * sum_zb / b_mix - 1) * (P * V_res / (R * T) - 1)
    sqrt2 = np.sqrt(2)
    F3 = - der_abRT_mix / (2 * sqrt2) * np.log((V_res + (1 + sqrt2) * b_mix) / (V_res + (1 - sqrt2) * b_mix))
    FUG = F1 + F2 + F3
    FUG = FUG + P * c_mix / (R * T)
    return FUG

"""
#==============================================================================
# Function: sr = entropy(T,v,da_dT,b):
#==============================================================================
# Function that calculates residual entropies with the PR-EoS
# INPUTS
# -- T = temperature in K: real[1,1]
# -- v = molar volume in L/mol: real[NC,1]
# -- da_dT = T-derivative of the attractive parameter "a": real[NC,1]
# -- b = co-volume of the non-translated PR-EoS in L/mol: real[NC,1]
# OUTPUTS
# -- sr = dimensionless residual entropy: real[NC,1]
"""
def entropy(T,v,da_dT,b):

    # PR EoS constants
    r1 = -1 - np.sqrt(2);
    r2 = -1 + np.sqrt(2);
    # residual entropy
    sr = R*np.log((v-b)/v) - 1/(b*(r1-r2)) * np.log((v-b*r1)/(v-b*r2)) * da_dT
    sr = sr/R
    return sr

#************************************
#************************************
#            COSMO-RS
#************************************
#************************************

"""
#==============================================================================
# Function: Em298, Ehb298 = interaction_mtx(par)
#==============================================================================
# This function generates the interaction matrix at 298 K
# INPUTS
# -- par = list containing COSMO-RS universal constants
# OUTPUTS
# -- Em298 = misfit interaction matrix: real[sigma_max,sigma_max]
# -- Ehb298 = hydrogen bonding interaction matrix: real[sigma_max,sigma_max]
"""
def interaction_mtx(par):
    # Function to calculate the misfit interaction between two segments "i" and "j"
    def Emisfit(sigma_i, sigma_j, alpha):
        # Misfit (or electrostatic) energy
        Emis = (alpha / 2.) * (sigma_i + sigma_j)**2  # [J/mol/A2]
        return Emis
    # Function to calculate the hydrogen-bonding interaction between two segments "i" and "j"
    def Ehydbon(sigma_i, sigma_j, sigma_hb, Chb):
        # HB donnor
        sigma_don = min(sigma_i, sigma_j)
        # HB acceptor
        sigma_acc = max(sigma_i, sigma_j)
        # Calculation of HB energy
        term1 = min([0, (sigma_don + sigma_hb)])
        term2 = max([0, (sigma_acc - sigma_hb)])
        Ehb = Chb * min([0, (term1 * term2)])
        return Ehb
    sigma = par['sigma']
    N = len(sigma)
    # Interaction matrix
    Em298 = np.zeros((N, N))
    Ehb298 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Interaction energies between sigma_i and sigma_j
            # -- Misfit (or electrostatic) energy
            Em298[i, j] = Emisfit(sigma[i], sigma[j], par['alpha'])
            # -- Hydrogen-bond energy
            Ehb298[i, j] = Ehydbon(sigma[i], sigma[j], par['sigma_hb'], par['Chb'])

    return Em298, Ehb298

"""
#==============================================================================
# Function: Et = T_correction_mtx(Tk,par)
#==============================================================================
# This function corrects the interaction matrix to temperatures 
# different of 298 K
# INPUTS
# -- Tk = temperature in K: real[1,1]
# -- par = list containing COSMO-RS universal constants
# OUTPUTS
# -- Et = total interaction matrix at Tk: real[sigma_max,sigma_max]
"""
def T_correction_mtx(Tk,par): 
    # T-dep function for hydrogen bonding
    R = par['RSI']
    termT = Tk * np.log(1 + np.exp(par['ChbT1'] / (R * Tk)) / par['ChbT2'])
    term298 = 298.15 * np.log(1 + np.exp((par['ChbT1'] ) / (R * 298.15)) / par['ChbT2'])
    fhbT = termT / term298
    Ehb = fhbT * par['Ehb298']
    # T-dep function for the electrostatic interaction
    fm = 1.
    Em = fm * par['Em298']
    # total interaction energy matrix
    Et = Em + Ehb
    return Et

"""
#==============================================================================  
# Function: pot_res = sigma_potential(Tk, profile, Aint, par)  
#==============================================================================  
# This function calculates the sigma-potential of a compound based on its  
# sigma-profile and interaction matrix. The sigma-potential is crucial for  
# determining intermolecular interactions in thermodynamic models.  
#  
# INPUTS:  
# -- Tk       : Temperature [K]  
# -- profile  : Sigma-profile (distribution of surface charge density)  
# -- Aint     : Interaction matrix from interaction_mtx function  
# -- par      : Dictionary containing model parameters:  
#               - 'RSI'  : Parameter related to interaction strength  
#               - 'aeff' : Effective molecular area  
#  
# OUTPUT:  
# -- pot_res  : Sigma-potential in kJ/mol/nm² (real[1, N])  
#==============================================================================  
"""
def sigma_potential(Tk, profile, Aint, potential_0, par):
    
    # Normalization of sigma-profiles
    sum_profile = np.sum(profile)
    psigma_calc = profile / sum_profile
    N = len(profile)  # Number of points of the discretized sigma-profile
    
    # Defintion of the matrix MU = psigma/Aint (Aint is calculated in interaction_mtx)
    MU = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            MU[i,j] = psigma_calc[j].item()/Aint[i,j].item()
    
    # Initializations
    aeffRT_inv = (par['RSI'] * Tk) / par['aeff']
    Z_old = np.exp(potential_0/(100.*aeffRT_inv))
    Z_old = np.ones((N,1))
    
    # Tolerances for the numerical solutions
    error = 1e3
    Tol   = 1e-9 # sucessive substitution only
        
    # Solving for Z using successive substitution
    while error > Tol:  # error in kcal/mol/nm^2
        Z = np.matmul(MU,Z_old)
        Z = 1/Z
        # Deviation results
        error = np.max(np.abs(Z - Z_old))
        Z_old = (Z + Z_old)/2.
        
    # Calculating sigma potentials from the Z-vector
    pot = 100.* aeffRT_inv * np.log(Z) # in kJ/mol/nm2
    pot_res = pot.reshape(1, -1)
    return pot_res
    
"""
#==============================================================================  
# Function: lngam = lngamcosmo(Tk,z,profiles_pures, pot_pure_init, pot_mix_init, par)  
#==============================================================================  
# This function calculates the logarithm of the activity coefficient (ln(γ))  
# for each compound in a mixture using the COSMO-RS model. The calculation  
# is based on sigma-potentials and interaction matrices.  
#  
# INPUTS:  
# -- Tk             : Temperature [K]  
# -- z              : Mole fraction of each compound in the mixture (array of size NC)  
# -- profiles_pures : Sigma-profiles of pure compounds [nm²] (real[NC, Nsigma])  
# -- par           : Dictionary containing model parameters:  
#                    - 'aeff'  : Effective molecular area  
#                    - 'RSI'   : Scaling factor for interactions 
# -- pot_pure_init, pot_mix_init : these inputs are initialization for the calculations of
#                                  sigma-potentials with COSMO-RS
#  
# OUTPUT:  
# -- lngam : Logarithm of the activity coefficient of each compound in the mixture (real[NC,1])  
"""
def lngamcosmo(Tk,z,profiles_pures, pot_pure_init, pot_mix_init, par):
    
    # Calculation of the interaction matrix at Tk
    Et = T_correction_mtx(Tk,par)
    aeffRT = par['aeff'] / (par['RSI'] * Tk)   
    Aint = np.exp(aeffRT*Et)
    
    #-------------------------------
    # Number of compounds
    NC = len(z)
    pot_sigma_pures = np.zeros((NC,61)) 
    
    #-------------------------------
    # sigma-potential of the mixture at T [=] kcal/mol
    profile_mix = np.dot(np.transpose(z),profiles_pures)
    p_sigma_mix = np.transpose(profile_mix)
    pot_sigma_mix = sigma_potential(Tk, p_sigma_mix, Aint, pot_mix_init.reshape(-1, 1), par)
    
    #-------------------------------
    lngam = np.zeros((NC,1)) 
    for i in range(NC):
        
        # sigma-potential of pure compounds at T [=] kcal/mol
        p_sigma_i = profiles_pures[i,:].reshape(-1, 1) 
        pot_sigma_init = pot_pure_init[i,:].reshape(-1, 1) 
        pot_sigma_i = sigma_potential(Tk, p_sigma_i, Aint, pot_sigma_init, par)
        pot_sigma_pures[i,:] = pot_sigma_i
        
        # activity coeffcients of the solutes in the mixture
        lngam[i] = (np.dot((pot_sigma_mix),p_sigma_i) - np.dot((pot_sigma_i),p_sigma_i))/(par['RSI']*Tk)
        
    return lngam, pot_sigma_pures, pot_sigma_mix

#************************************
#************************************
#         PR EoS + COSMO-RS
#************************************
#************************************

"""
#==============================================================================  
# Function: Vm_corr, FUG = thermo(Tk, Pbar, zi, molecules, pot_pures_init, pot_mix_init, par)  
#==============================================================================  
# This function calculates the molar volume of a mixture and the fugacity  
# coefficients of solutes based on an equation of state (EoS) approach.  
#  
# INPUTS:  
# -- Tk        : Temperature [K]  
# -- Pbar      : Pressure [bar]  
# -- zi        : Mole fraction of each compound in the mixture (array of size NC)  
# -- molecules : List of molecule objects containing thermodynamic properties:  
#                - Tc : Critical temperature [K]  
#                - Pc : Critical pressure [bar]  
#                - w  : Acentric factor [-]  
#                - L, M, N, c : Twu parameters (if applicable)  
#                - psigma : Sigma profile data    
# -- pot_pure_init, pot_mix_init : these inputs are initialization for the calculations of
#                                  sigma-potentials with COSMO-RS
# -- par       : Dictionary containing model parameters:  
#                - 'ceosGC' : Parameters for group contribution EoS  
#                - 'q1', 's' : Mixing rule parameters
#  
# OUTPUTS:  
# -- Vm_corr : Corrected molar volume of the mixture [L/mol]  
# -- FUG     : Fugacity coefficients of solutes in the mixture (array of size NC) 
"""
def thermo(Tk, Pbar, zi, molecules, pot_pures_init, pot_mix_init, par):
    
    NC = len(molecules)
    a = np.zeros((NC,1))
    b = np.zeros((NC,1))
    c = 8888.*np.ones((NC,1))
    profiles_pures = np.zeros((NC,61))  
    
    for i in range(NC):
        # constants
        Tc = molecules[i].Tc
        Pc = molecules[i].Pc
        w = molecules[i].w
        m = molecules[i].m
        L = molecules[i].L
        M = molecules[i].M
        N = molecules[i].N
        c0 = molecules[i].c
        # sigma-profiles
        psigma = molecules[i].psigma        
        profiles_pures[i,:] = np.transpose(psigma[:,1])
        # pure compound EoS parameters at  and Tc
        if molecules[i].idx == 8888 or molecules[i].smiles == "[HH]" or molecules[i].smiles == "[He]":
            # It means that the molecule was not found in the database
            # For such a molecule, the group contribution methods were applied
            # ps: in this case, we use SOAVE 1972 as the alpha function 
            a[i],b[i],c[i] = par_soave72(Tk,Tc,Pc,m)
        else:
            # Otherwise, we use TWU 1991
            a[i],b[i],c[i] = par_twu91(Tk,Tc,Pc,w,L,M,N,c0)
    
    # Activity coefficients of solutes in the mixture
    lnGam, pot_pures, pot_mix = lngamcosmo(Tk,zi,profiles_pures, pot_pures_init, pot_mix_init, par)

    # EoS Mixture parameters
    a_mix, b_mix, c_mix, abRT_mix, sum_zb, der_abRT_mix = mix_rule(Tk,zi,lnGam,a,b,c,par['q1'],par['s'])

    # Molar volume and fugacities
    phase = 0 # more stable solition of the cubic EoS
    Z, Vm = molar_volume(Pbar,Tk,a_mix,b_mix,phase)
    FUG = fugacity(Pbar, Tk, Vm, b_mix, sum_zb, der_abRT_mix, c_mix) 
    
    # Correction of the molar volume using the volume translation constant
    Vm_corr = Vm - c_mix
    
    return Vm_corr, FUG, pot_pures, pot_mix
