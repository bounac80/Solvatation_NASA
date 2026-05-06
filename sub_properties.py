# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:34:26 2026

@author: Paes1
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sub_model import thermo, molar_volume, entropy, lngamcosmo, mix_rule
from sub_model import par_soave72, Tder_soave
from sub_model import par_twu91, Tder_twu
from sub_phase_equilibria import psat_pure

"""
#==============================================================================  
# Function: DGsolv, NASAcoeff, r2 = fit_nasa(Tk, Pbar, zi, molecules, params)  
#==============================================================================  
# This function fits the solvation free energy data to a NASA polynomial form  
# using linear regression.  
#  
# INPUTS:  
# -- Tk        : Temperature array [K] (nT x 1)  
# -- Pbar      : Pressure [bar]  
# -- zi        : Mole fraction of each compound in the mixture (nM x 1)  
# -- molecules : List of molecule objects containing thermodynamic properties:  
#                - Tc : Critical temperature [K]  
#                - Pc : Critical pressure [bar]  
#                - w  : Acentric factor [-]  
#                - L, M, N, c0 : Twu parameters (if applicable)  
# -- params    : Dictionary of model parameters  
#  
# OUTPUTS:  
# -- DGsolv   : Solvation free energy [kcal/mol] for each compound (nM x nT)  
# -- NASAcoeff: NASA polynomial coefficients (nM x 7)  
# -- r2       : R-squared value for the regression fit (nM x 1) 
"""
def fit_nasa(Tk, Pbar, zi, molecules, params):
    
    def progress_bar(iteration, total, length=50):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = '#' * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r|{bar}| {percent}%')
        sys.stdout.flush()
    
    nT = len(Tk)
    nM = len(zi)
    DGsolv = np.zeros((nM,nT))
    DSsolv = np.zeros((nM,nT))
    DHsolv = np.zeros((nM,nT))
    DCPsolv = np.zeros((nM,nT))
    
    #--------------------------------------------
    # size initialization for pure compound arrays
    NC = len(molecules)
    a = np.zeros((NC,1))
    b = np.zeros((NC,1))
    c = 8888.*np.ones((NC,1))
    Radius = 8888.*np.ones((NC,1))
    # solvent data
    Psat_pure = 8888*np.ones((NC,1))
    Psat_mix = 8888*np.ones((nT,1))
    Mu_mix = 8888*np.ones((nT,1))
    profiles_pures = np.zeros((NC,61)) 
    all_smiles = []
    
    # initialization of sigma potentials 
    pot_pures_init = np.zeros((NC,61)) 
    pot_mix_init = np.zeros((1,61)) 
    
    #-------------------------------------------
    for i in range(NC):
        psigma = molecules[i].psigma        
        profiles_pures[i,:] = np.transpose(psigma[:,1])
        Vcosmo = molecules[i].Vcosmo
        all_smiles.append(molecules[i].smiles)
        # Estimating the radius of the molecule from COSMO cavity
        Radius[i] = ((3*Vcosmo/4*np.pi)**(1/3)) # estimated radius in m
    
    idx = 0
    for T in Tk:
        T = T.item()
        for i in range(NC):
            # Calculation of pure compound EoS parameters
            Tc = molecules[i].Tc
            if i == 1:
                Tc_solvent = Tc
            Pc = molecules[i].Pc
            w = molecules[i].w
            m = molecules[i].m
            L = molecules[i].L
            M = molecules[i].M
            N = molecules[i].N
            c0 = molecules[i].c
            # pure compound parameters
            if molecules[i].idx == 8888 or molecules[i].smiles == "[HH]" or molecules[i].smiles == "[He]":
                # It means that the molecule was not found in the database
                # For such a molecule, the group contribution methods were applied
                # ps: in this case, we use SOAVE 1972 as the alpha function 
                a[i],b[i],c[i] = par_soave72(T,Tc,Pc,m)
            else:
                # Otherwise, we use TWU 1991
                a[i],b[i],c[i] = par_twu91(T,Tc,Pc,w,L,M,N,c0)
            
            # Calculation of pure compound vapor pressure
            if T < molecules[i].Tc and zi[i] > 0:
                # Vapor pressure (pure compound)
                # -- correlation for initial guess
                P0 = Pc * 10 ** (7/3 * (w + 1) * (1 - Tc/T))
                # -- full calculation with the PR-EoS
                Psat_pure[i] = psat_pure(T,a[i],b[i],c[i],P0)

        #--------------------------------------------
        # Vapor pressure of the bulk phase
        count_positive = sum(1 for z in zi if z > 0)
        # (if mixture of solvents)
        if count_positive  >= 2:
            lnGam, pot_pures, pot_mix = lngamcosmo(T,zi,profiles_pures, pot_pures_init, pot_mix_init, params)
            a_mix, b_mix, c_mix, abRT_mix, sum_zb, der_abRT_mix = mix_rule(T,zi,lnGam,a,b,c,params['q1'],params['s'])
            xi_lngam = zi * np.exp(lnGam)
            P0 = np.matmul(np.transpose(Psat_pure),xi_lngam).item()
            Psat = psat_pure(T,a_mix,b_mix,c_mix,P0)
            Psat_mix[idx,0] = Psat
            pot_pures_init = pot_pures
            pot_mix_init = pot_mix
        # (if pure solvent)
        else:
            Psat = Psat_pure[0, 0]   # scalar, not shape-(1,) slice
            Psat_mix[idx,0] = Psat
            
        #--------------------------------------------
        # Solvation free energies
        eps = 1e-5
        P = np.maximum(Psat,Pbar) * 1.01
        DG, DS, DH, DCP, pot_pures, pot_mix = solvation_quantities(T, P, zi, molecules, pot_pures_init, pot_mix_init, params, eps)
        DGsolv[:,idx] = DG[:,0]/(8.31446261815324/4184. * T)
        DSsolv[:,idx] = DS[:,0]/(8.31446261815324/4184)
        DHsolv[:,idx] = DH[:,0]/(8.31446261815324/4184 * T)
        DCPsolv[:,idx] = DCP[:,0]/(8.31446261815324/4184)
        
        pot_pures_init = pot_pures
        pot_mix_init = pot_mix

        #--------------------------------------------
        # Viscosity of the liquid phase
        Mu_mix[idx,0] = transport(T, P, zi, molecules, params)
        
        idx += 1
        progress_bar(idx, len(Tk))
    print()
        
    #--------------------------------------------
    # Solvation NASA polynomials
    # -- Gibbs free energy
    Xg = np.zeros((nM,7))
    Xg1 = 1. - np.log(Tk)
    Xg2 = -1/2*Tk
    Xg3 = -1/6*(Tk**2)
    Xg4 = -1/12*(Tk**3)
    Xg5 = -1/20*(Tk**4)
    Xg6 = 1/Tk
    Xg7 = -Tk/Tk
    Xg = np.hstack((Xg1,Xg2,Xg3,Xg4,Xg5,Xg6,Xg7))
    # -- Entropy
    Xs = np.zeros((nM,7))
    Xs1 = np.log(Tk)
    Xs2 = Tk
    Xs3 = 1/2*(Tk**2)
    Xs4 = 1/3*(Tk**3)
    Xs5 = 1/4*(Tk**4)
    Xs6 = 0*Tk
    Xs7 = Tk/Tk
    Xs = np.hstack((Xs1,Xs2,Xs3,Xs4,Xs5,Xs6,Xs7))
    # -- Enthalpy
    Xh = np.zeros((nM,7))
    Xh1 = Tk/Tk
    Xh2 = 1/2*Tk
    Xh3 = 1/3*(Tk**2)
    Xh4 = 1/4*(Tk**3)
    Xh5 = 1/5*(Tk**4)
    Xh6 = 1/Tk
    Xh7 = 0*Tk
    Xh = np.hstack((Xh1,Xh2,Xh3,Xh4,Xh5,Xh6,Xh7))
    # -- Heat capacity
    # Tk_fit_cp = Tk[Tk < 0.9*Tc_solvent]
    Xc1 = Tk/Tk
    Xc2 = Tk
    Xc3 = Tk**2
    Xc4 = Tk**3
    Xc5 = Tk**4
    Xc6 = 0*Tk
    Xc7 = 0*Tk
    Xc = np.hstack((Xc1,Xc2,Xc3,Xc4,Xc5,Xc6,Xc7))
    
    # all data
    X = np.vstack((Xg,Xs,Xh,Xc))
    
    # Create a linear regression model
    # coeff = np.zeros((nM,7))
    NASAcoeff = np.zeros((nM,7))
    r2_nasa = np.zeros((nM,1))
    
    for i in range(0,nM):
        Yg = np.transpose(DGsolv[i,:]).reshape(-1, 1)
        Ys = np.transpose(DSsolv[i,:]).reshape(-1, 1)
        Yh = np.transpose(DHsolv[i,:]).reshape(-1, 1)
        Yc = np.transpose(DCPsolv[i,:]).reshape(-1, 1)
        Y = np.vstack((Yg,Ys,Yh,Yc))
        model = LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        r2_nasa[i] = model.score(X, Y)
        NASAcoeff[i, :] = model.coef_
    
         
    #--------------------------------------------
    # Results
    thermochem = {
        "smiles": all_smiles,
        "T vector": Tk,
        "Tmin": np.min(Tk),
        "Tmax": np.max(Tk),
        "Vapor pressures in bar": Psat_mix,
        "Viscosities in Pa.s": Mu_mix,
        "DGsolv/RT": DGsolv,
        "DSsolv/R": DSsolv,
        "DHsolv/RT": DHsolv,
        "DCPsolv/RT": DCPsolv,
        "NASA_coefficients": NASAcoeff,
        "R2_nasa": r2_nasa,
        }
        
    return thermochem

"""
#==============================================================================  
# Function: DGsolv, DSsolv, DHsolv, DCPsolv = solvation_quantities(Tk, Pbar, zi, molecules, pot_pures_init, pot_mix_init, params, eps)  
#==============================================================================  
# This function calculates solvation thermodynamic properties, including Gibbs free energy,  
# entropy, enthalpy, and heat capacity, using finite difference approximations.  
#  
# INPUTS:  
# -- Tk        : Temperature [K]  
# -- Pbar      : Pressure [bar]  
# -- zi        : Mole fraction of each component in the mixture (nM x 1)  
# -- molecules : List of molecule objects containing thermodynamic properties
# -- pot_pure_init, pot_mix_init : these inputs are initialization for the calculations of
#                                  sigma-potentials with COSMO-RS  
# -- params    : Dictionary of model parameters  
# -- eps       : Small perturbation factor for numerical derivatives  
#  
# OUTPUTS:  
# -- DGsolv  : Solvation Gibbs free energy at Tk [kcal/mol]  
# -- DSsolv  : Solvation entropy [kcal/mol/K]  
# -- DHsolv  : Solvation enthalpy [kcal/mol]  
# -- DCPsolv : Solvation heat capacity [kcal/mol/K]  
"""
def solvation_quantities(Tk, Pbar, zi, molecules, pot_pures_init, pot_mix_init, params, eps):
    
    # Function to calculated solvation free energies
    def solvation_free_energy(P, T, V_res, FUG):
        # Gas constant [=] bar.L/(mol K)
        R = 8.31446261815324e-2
        expFUG = np.exp(FUG)
        DGsolv = (100 * R * T * np.log(P * V_res * expFUG / (R * T))) / 4184.
        return DGsolv
    
    #----------------------------
    
    # DGsolv at TK [kcal/mol]
    Vm, FUG, pot_pures, pot_mix = thermo(Tk, Pbar, zi, molecules, pot_pures_init, pot_mix_init, params)
    DGsolv = solvation_free_energy(Pbar, Tk, Vm, FUG)
    
    # DGsolv at T = Tk*(1-eps)
    T = Tk*(1-eps)
    Vm, FUG, pot_pures0, pot_mix0 = thermo(T, Pbar, zi, molecules, pot_pures_init, pot_mix_init, params)
    DG0 = solvation_free_energy(Pbar, T, Vm, FUG)
    
    # DGsolv at T = Tk*(1+eps)
    T = Tk*(1+eps)
    Vm, FUG, pot_pures0, pot_mix0 = thermo(T, Pbar, zi, molecules, pot_pures_init, pot_mix_init, params)
    DG1 = solvation_free_energy(Pbar, T, Vm, FUG)
    
    #---------------------------
    
    # SOLVATION ENTROPY IN [kcal/mol/K]
    deltaT = Tk*eps
    DSsolv = -(DG1 - DG0)/(2*deltaT)
    
    # SOLVATION ENTHALPY [kcal/mol]
    DHsolv = DGsolv + Tk*DSsolv
    
    # SOLVATION HEAT CAPACITY [kcal/mol/K]
    DCPsolv = - T*(DG1 - 2*DGsolv + DG0)/deltaT**2
    
    return DGsolv, DSsolv, DHsolv, DCPsolv, pot_pures, pot_mix


"""
#==============================================================================
# Function: Eta = viscosity(T,v,da_dT,b):
#==============================================================================
# Function that calculates viscosities from entropy scaling
# -- T = temperature in K
# -- v = molar volume in L/mol
# -- S_tv = dimensionless residual entropy at T and vc
# -- S_tv = dimensionless residual entropy at Tc and Vc
# -- a1, a2, b1, b2, c1, d1 = entropy scaling constants
# -- Mw = molecular weight
# OUTPUTS
# -- Eta = dynamic viscosity in Pa.s
"""
def viscosity(T,v,S_tv,Sc_tv,a1,a2,b1,b2,c1,d1,Mw):
    # Avogadro constant
    NA = 6.02214076e23
    # Boltzmann constant in m2 kg s-2 K-1
    kb = 1.380649e-23
    # molecular mass (mass of one molecule)
    Mw = Mw/1000 # from g/mol to kg/mol
    m0 = Mw / NA; # from kg / mol to kg / molecule
    # molecular density (molecules per unit of volume)
    v = v / 1000; # from L/mol to m3/mol
    ro = (1/v) * NA; # molecules / m3
    # reference eta
    Eta_ref = ro**(2/3) * np.sqrt(m0 * kb * T)
    # Scaling X variable
    Xs = -(S_tv/Sc_tv) - np.log((S_tv/Sc_tv));
    # Scaling Y variable
    Ys = ((a1 + a2*S_tv)/(1 + np.exp(c1*Xs)) + (b1 + b2*S_tv)/(1 + np.exp(-c1*Xs))) * Xs + d1/Sc_tv;
    # Dynamic viscosity
    Eta = np.exp(Ys) * Eta_ref;
    return Eta


"""
#==============================================================================  
# Function: Mu_mix = transport(Tk, Pbar, zi, molecules, par)  
#==============================================================================  
# This function calculates the dynamic viscosity of a mixture using entropy  
# scaling methods based on an equation of state (EoS) approach. It computes  
# pure component viscosities and combines them using a logarithmic mixing rule.  
#  
# INPUTS:  
# -- Tk        : Temperature [K]  
# -- Pbar      : Pressure [bar]  
# -- zi        : Mole fraction of each compound in the mixture (array of size NC)  
# -- molecules : List of molecule objects containing thermodynamic properties:  
#                - Tc : Critical temperature [K]  
#                - Pc : Critical pressure [bar]  
#                - w  : Acentric factor [-]  
#                - L, M, N, c : TWU parameters  
#                - m  : Soave parameter (for fallback EoS)  
#                - a1, a2, b1, b2, c1, d1 : Entropy scaling parameters  
#                - MW : Molecular weight [g/mol]  
#                - idx, smiles : Used for detecting database availability  
# -- par       : Dictionary containing model parameters (may include alpha function type etc.)  
#  
# OUTPUTS:  
# -- Mu_mix : Dynamic viscosity of the mixture [Pa.s]  
"""

def transport(Tk, Pbar, zi, molecules, par):

    NC = len(molecules)
    Mu_pure = 8888*np.ones((NC,1))
    for i in range(NC):
        
        if zi[i] > 0: # only for solvents
            # constants
            Tc = molecules[i].Tc
            Pc = molecules[i].Pc
            w = molecules[i].w
            m = molecules[i].m
            L = molecules[i].L
            M = molecules[i].M
            N = molecules[i].N
            c0 = molecules[i].c
            # -- component-specific parameters for entropy scaling
            a1 = molecules[i].a1
            a2 = molecules[i].a2
            b1 = molecules[i].b1
            b2 = molecules[i].b2
            c1 = molecules[i].c1
            d1 = molecules[i].d1
            Mw = molecules[i].MW # g/mol
            # pure compound EoS parameters at Pc and Tc
            if molecules[i].idx == 8888 or molecules[i].smiles == "[HH]" or molecules[i].smiles == "[He]":
                # It means that the molecule was not found in the database
                # For such a molecule, the group contribution methods were applied
                # ps: in this case, we use SOAVE 1972 as the alpha function 
                a, b, c = par_soave72(Tk,Tc,Pc,m)
                ac, bc, cc = par_soave72(Tk,Tc,Pc,m)
                dac_dT = Tder_soave(Tk,Tc,Pc,m)
                da_dT = Tder_soave(Tk,Tc,Pc,m)
            else:
                # Otherwise, we use TWU 1991
                a, b, c = par_twu91(Tk,Tc,Pc,w,L,M,N,c0)
                da_dT = Tder_twu(Tk,Tc,Pc,w,L,M,N)
                ac, bc, cc = par_twu91(Tk,Tc,Pc,w,L,M,N,c0)
                dac_dT = Tder_twu(Tc,Tc,Pc,w,L,M,N)
            
            if a1 == 8888: 
                # -- in this case we use universal parameters
                a1 = -0.35918064
                a2 = 0.00488280
                b1 = 0.59431770
                b2 = -0.05315258
                c1 = 0.44624897
                d1 = 0.41245087
                
            # -- residual entropies at T and Vm
            Z, Vm = molar_volume(Pbar,Tk,a,b,0)
            S_tv = entropy(Tk,Vm,da_dT,b)
            
            # -- residual entropies at Tc and Vc
            Zc, Vc = molar_volume(molecules[i].Pc,molecules[i].Tc,ac,bc,0)
            Sc_tv = entropy(molecules[i].Tc,Vc,dac_dT,bc)
            
            # -- entropy scaling to viscosity (in Pa.s)
            Mu_pure[i] = viscosity(Tk,Vm-cc,S_tv,Sc_tv,a1,a2,b1,b2,c1,d1,Mw)
            
    # Viscosity of the mixture
    lnMU_pure = np.log(Mu_pure)
    lnMU_mix = np.matmul(np.transpose(zi), lnMU_pure).item()
    Mu_mix = np.exp(lnMU_mix)

    return Mu_mix

