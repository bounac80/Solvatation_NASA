# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:42:33 2026

@author: Paes1
"""

from rdkit import Chem # install rdkit in an env
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
from sub_fragmenter import bestfragmentation
import numpy as np
import pandas as pd

#************************************
# Classes
#************************************
class Molecule:
    def __init__(self,idx,cas,name,MW,Tc,Pc,w,m,L,M,N,c,a1,a2,b1,b2,c1,d1,Vcosmo,psigma,smiles,groups,occ):
        self.idx = idx
        self.cas = cas
        self.name = name
        self.MW = MW
        self.Tc = Tc
        self.Pc = Pc
        self.w = w
        self.m = m
        self.L = L
        self.M = M
        self.N = N
        self.c = c
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.d1 = d1
        self.Vcosmo = Vcosmo
        self.psigma = psigma
        self.smiles = smiles
        self.groups = groups
        self.occ = occ
 
        
# In[par_soaveGC]:
"""
#==============================================================================
# Function: Tc,Pc,w,c = par_soaveGC(T,occ,par_gc)
#==============================================================================
# Calculation of pure compounds data through group contribution
# INPUTS
# -- T = temperature in K: real[1,1]
# -- occ = occurances matrix of UNIFAC groups: real[NC,Ngr]
# -- par_gc = group contributions real[Ngr,1]
#             par_gc.ac = contributions for the atractive parameter at T = Tc: real[Ngr,1]
#             par_gc.b = contributions for the co-volume: real[Ngr,1]
#             par_gc.a0 = contributions for the atractive parameter at T = 0K: real[Ngr,1]
# OUTPUTS
# -- Tc = critical temperature in K: real[NC,1]
# -- Pc = critical pressure in bar: real[NC,1]
# -- w = acentric factor: real[NC,1]
# -- c = avolume translation = 0 L/mol: real[NC,1]
"""
def GC_calc(occ,ceosGC):
    # gas constant [=] bar.L/(mol K)
    R = 8.31446261815324e-2;
    # CEoS constants
    omega_a = 0.45724
    omega_b = 0.0778
    OmhR_a = omega_a*R**2;
    OmhR_b = omega_b*R;
    # expoents
    expb = 4/5
    expac = 2/3
    expa0 = 0.48
    ctes_b = ceosGC[:,0]
    ctes_b = ctes_b.reshape((len(ctes_b), 1))
    ctes_ac = ceosGC[:,1]
    ctes_ac = ctes_ac.reshape((len(ctes_ac), 1))
    ctes_a0 = ceosGC[:,2]
    ctes_a0 = ctes_a0.reshape((len(ctes_a0), 1))
    # GC calculations for b, ac, and a0
    b  = (OmhR_b * (np.matmul(np.transpose(occ),ctes_b))**(1/expb)).item()
    ac = (OmhR_a * (np.matmul(np.transpose(occ),ctes_ac))**(1/expac)).item()
    a0 = (OmhR_a * (np.matmul(np.transpose(occ),ctes_a0))**(1/expa0)).item()
    # critical constants
    Tc = float((omega_b/omega_a*1/R)*(ac/b))
    Pc = float((omega_b**2/omega_a)*(ac/(b**2)))
    # soave alpha function parameter
    m = float(np.sqrt(((a0))/(ac)) - 1.)
    # acentric factor
    def solve_w(m):
        coefficients = [-0.26992, 1.54226, 0.37464 - m]
        roots = np.roots(coefficients)
        valid_roots = [w for w in roots if -1 <= w <= 3]
        if not valid_roots:
            return None
        return min(valid_roots, key=abs)  # Choose the one closest to zero
    w = solve_w(m)
    # volume translation = 0
    c = 0.
    # COSMO cavity
    ctes_v = ceosGC[:,3]
    Vcosmo  = (np.matmul(np.transpose(occ),ctes_v)).item()
    return Tc, Pc, w, m, c, Vcosmo

# In[calcpsigmaGC]:
    
"""
#==============================================================================  
# Function: p_sigma = calcpsigmaGC(occ, psigmaGC, sigma)  
#==============================================================================  
# This function calculates the sigma profile using group contribution (GC).
#  
# INPUTS:  
# -- occ       : Occurrence matrix of functional groups (nGroups x 1)  
# -- psigmaGC  : Sigma profile contributions of each functional group (nGroups x nSigma)  
# -- sigma     : Discretized sigma values [e/nm²] (nSigma x 1)  
#  
# OUTPUTS:  
# -- p_sigma   : Computed sigma profile (nSigma x 2)  
#                Column 1: Sigma values  
#                Column 2: Sigma profile density
"""
def calcpsigmaGC(occ, psigmaGC, sigma):
    # Calculate sigma-profile
    p_sigma = np.zeros((len(sigma), 2))
    p_sigma[:, 0] = sigma
    p_sigma[:, 1] = np.matmul(np.transpose(occ),psigmaGC)
    p_sigma[:, 1] = [(i > 1e-4) * i for i in p_sigma[:, 1]]
    return p_sigma

# In[psigmaBDD]:
    
"""            
#==============================================================================
# Function: p_sigma = psigmaBDD(molecule,level):
#==============================================================================
# Function that gets the sigma profile from the database
# -- molecule = name of the file .dat: str[1,1]
# -- level = "BP-TZVPD-FINE": str[1,1]
# OUTPUTS
# -- p_sigma = sigma profile: real[61,2]
"""
def psigmaBDD(molecule_id,level):
    # Finding sigma-profile
    switch = {
        'BP-TZVPD-FINE': 'database/psigma/BP-TZVPD-FINE/' + molecule_id + '.dat',
        'BP-TZVP': 'database/psigma/BP-TZVP/' + molecule_id + '.dat',
        'PB-DMOL3': 'sigma-profiles/PB-DMOL3/' + molecule_id + '.dat',
    }
    filename = switch.get(level, '')
    if filename:
        with open(filename , 'r') as f:
            line = f.readline()
            i = 0
            p_sigma = np.zeros((61,2))
            for line in f:
                new_line = line.split()
                p_sigma[i,0] = float(new_line[0])
                p_sigma[i,1] = float(new_line[1])
                i = i + 1
    else:
        p_sigma = np.zeros((61, 2))
    return p_sigma

# In[LoadBDD]:
"""
#==============================================================================
# Function: BDD = LoadBDD(filename):
#==============================================================================
# Function to load the database of pure compounds
# -- filename = file containing the pure compound data: str[1,1]
# OUTPUTS
# -- BDD = list of molecules in the database: list[Nmol]
#          (Note that each element of the list belongs to the "Molecule" class
#           containing idx, cas, name, Pc, w, L, M, N, c, psigma, smiles).
#          (Nmol = numbe of molecules in the database)
"""
def LoadBDD(filename):
    
    BDD_df = pd.read_csv(filename, delimiter=';')
    BDD = []

    for row in BDD_df.itertuples(index=False): 
        name = str(row[0])
        idx = int(row[1])  
        cas = str(row[2])
        MW = float(row[3])
        Tc = float(row[4])
        Pc = float(row[5])
        w = float(row[6])
        m = 0.37464+1.5422*w-0.26992*w**2
        L = float(row[7])
        M = float(row[8])
        N = float(row[9])
        c = float(row[10])/1000.
        a1 = float(row[11])
        a2 = float(row[12])
        b1 = float(row[13])
        b2 = float(row[14])
        c1 = float(row[15])
        d1 = float(row[16])
        Vcosmo = float(row[17])
        
        psigma = psigmaBDD(cas,'BP-TZVPD-FINE')
        smiles = str(row[18])
        smiles = Chem.CanonSmiles(smiles)
        groups = "[database]"  # Default value
        occ = 0
        BDD.append(Molecule(idx,cas,name,MW,Tc,Pc,w,m,L,M,N,c,a1,a2,b1,b2,c1,d1,Vcosmo,psigma,smiles,groups,occ))
        
    return BDD

# In[loadmolecules]:

"""
#==============================================================================
# Function: p_sigma = sigma_profile_bdd(molecule,level):
#==============================================================================
# Function that gets the sigma profile from the database
# -- molecule = name of the file .dat: str[1,1]
# -- level = "BP-TZVPD-FINE": str[1,1]
# OUTPUTS
# -- p_sigma = sigma profile: real[61,2]
""" 

def loadmolecules(file_solvents,file_solutes,file_conditions,BDD,params):
    
    ceosGC = params['ceosGC']
    psigmaGC = params['psigmaGC']
    sigma_max = params['sigma_max']
    delta_sigma = params['delta_sigma']
    sigma = np.arange(-sigma_max, sigma_max + delta_sigma, delta_sigma)

    # -- Step 1: Load and canonicalize all SMILES
    zi = []
    smiles_input = []

    with open(file_solvents) as f:
        next(f)  # Skip header
        for line in f:
            tokens = line.split()
            cansmi = Chem.CanonSmiles(tokens[0])
            smiles_input.append(cansmi)
            zi.append(float(tokens[1]))

    with open(file_solutes) as g:
        next(g)
        for line in g:
            tokens = line.split()
            cansmi = Chem.CanonSmiles(tokens[0])
            smiles_input.append(cansmi)
            zi.append(0.0)

    zi = np.array(zi).reshape((-1, 1))

    # -- Step 2: Create a lookup dictionary from BDD (canonical SMILES as key)
    bdd_dict = {
        Chem.CanonSmiles(mol.smiles): mol for mol in BDD
    }

    # -- Step 3: Process each molecule
    Molecules = []
    for smi in smiles_input:
        if smi in bdd_dict:
            Molecules.append(bdd_dict[smi])
        else:
            # If not in DB, generate a placeholder molecule and estimate
            idx = 8888
            cas = name = "8888"
            MW = Tc = Pc = w = L = M = N = c = 8888.0
            a1 = 8888
            a2 = 8888
            b1 = 8888
            b2 = 8888
            c1 = 8888
            d1 = 8888
            
            # group fragmentation
            groups, occ, frags = bestfragmentation(smi)
            Tc, Pc, w, m, c, Vcosmo = GC_calc(occ, ceosGC)
            psigma = calcpsigmaGC(occ, psigmaGC, sigma)
            Molecules.append(Molecule(idx, cas, name, MW, Tc, Pc, w, m,
                    L, M, N, c, a1 ,a2, b1, b2, c1, d1, Vcosmo, psigma, smi, groups, occ))

    # -- Step 4: Load pressure and temperature
    with open(file_conditions) as h:
        lines = h.readlines()
        Pbar = float(lines[1].split()[2])
        Tmin = float(lines[2].split()[2])
        Tmax = float(lines[3].split()[2])
        nT = int(lines[4].split()[2])

    Tk = np.linspace(Tmin, Tmax, nT).reshape((-1, 1))

    return Molecules, zi, Tk, Pbar

