#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import mol_data_generation.utils.file_utils as fu
import numpy as np
import mol_data_generation.utils.xtb_fcts as xtb
import scipy.spatial as scsp
from scipy.spatial.distance import euclidean 
import matplotlib.pyplot as plt
import random

# constants
e=1.6e-19 # C
m_to_A=1e-10
NA = 6.022e23 # g/mol
c = 2.998e8 # m/s
c_cm = 2.998e10
elementmasses={"C":12.01115, "H":1.00797, "O":15.99940,"N":14.007, "S":32.065}
#k_ = 8.6173303e-5 # eV/K
k = 1.380649e-16 # erg/K



## get number_samples/conformer ##

# num_samples_conf = int(self.prep_sample.num_samples/len(elements_conf))
def get_num_samples_conf(num_samples, num_conf):
    
    num_samples_conf_i = int(num_samples/num_conf)
    
    num_samples_i = num_samples_conf_i*num_conf
    if num_samples_i == num_samples:
        num_samples_conf_list = [num_samples_conf_i]*num_conf
        return num_samples_conf_list
    else:
        rest = num_samples-num_samples_i
        j = random.randrange(0,num_conf)
        num_samples_conf_list = []
        for i in range(num_conf):
            if i == j:
                num_samples_conf_list.append(num_samples_conf_i+rest)
            else:
                num_samples_conf_list.append(num_samples_conf_i)
        return num_samples_conf_list
        
## get vib parameters conf ##
def get_vib_parameters(moldir, num_atoms):
    wavenumbers =  fu.read_frequencies(moldir)
    red_masses = fu.read_red_masses(moldir)
    normal_coords = fu.read_normal_coordinates(moldir, num_atoms) #array
    
    f = c*wavenumbers*100 # 1/s
    reducedmasses_g= red_masses/NA # g
    force_constants = (2*np.pi*f)**2 * reducedmasses_g

    nf = len(red_masses)

    return normal_coords, force_constants, nf

def random_ci(nf):
    c_i = np.random.random(nf)
    rand_scale = np.random.random(1)
    c_sum = np.sum(c_i)
    factor = c_sum/rand_scale
    c_n = c_i/factor
    return c_n

def random_ci_01(nf, ci_min, ci_max):
    c_i = np.random.random(nf)
    rand_scale = np.random.uniform(ci_min, ci_max) #0.005,0.1
    c_sum = np.sum(c_i)
    factor = c_sum/rand_scale
    c_n = c_i/factor
    return c_n

def pos_or_neg():
    return 1 if np.random.random() < 0.5 else -1 

# get relative bond distances
def get_relative_bond_lengths(coords, bond_idx_list, bond_length_list):
    relative_bond_lengths = []
        
    for i, bond_pair in enumerate(bond_idx_list):

        distance_bond_i = euclidean(coords[bond_pair[0]-1], coords[bond_pair[1]-1])
        
        rel_bond_dist_i = distance_bond_i/bond_length_list[i]
        
        relative_bond_lengths.append(rel_bond_dist_i)
    return relative_bond_lengths
   
def do_sampling(coords_init, elements_init, normal_coords, force_constants,n_atoms, nf, num_samples,energy_init, force_init, energy_0, delta_E_max, temperature, ci_range, bond_idx_list_conf, bond_lengths_list_conf, chrg = 0, solve = None):
    coords_sampled_conf = [coords_init]
    elements_sampled_conf = [elements_init]
    energies_sampled_conf = [energy_init]
    forces_sampled_conf = [force_init]
    #print('eneries_sampled_conf init', energies_sampled_conf)
    delta_energies_init = abs(energy_0 - energy_init)
    delta_energies_sampled_conf = [delta_energies_init]
    # for parameter testing
    c_sum = []
    c_i_all = []
    r_i_all = []    
    relative_bonds = []
    for idx in range(num_samples-1):
        clash = True
        energy_max = True
        while clash and energy_max:
            c_tot = np.copy(coords_init)
            c_i = random_ci_01(nf, ci_range[0], ci_range[1])  ## change c_i scaling
    
            ci_sum = sum(c_i) # check whether sum ci scaling is too big, if passed geos only sum ci << ci sum -> need to adjust ci or T
            
            r_i_cm = np.sqrt((3.0*c_i*n_atoms*k*temperature)/(force_constants))
            r_i_A = r_i_cm*10e8
            
            for i in range(len(r_i_A)):
                a = pos_or_neg()
                
                c_tot += a*r_i_A[i] * normal_coords[i]
                
            ds = np.sort(scsp.distance.cdist(c_tot,c_tot).flatten())[n_atoms:]
            
            relative_bond_lengths = get_relative_bond_lengths(c_tot, bond_idx_list_conf, bond_lengths_list_conf)
            
            # check relative bond lengths - atom type specific
            # how to determine max bond lengths per H-X type?
            
            if min(ds) > 0.5 and max(relative_bond_lengths) <= 1.25:
                energy_sampled = xtb.single_point_energy(c_tot, elements_init, charge = chrg, solvent = solve) # add charge here and as fct variable
                force_sampled = xtb.single_force(c_tot, elements_init, charge = chrg, solvent = solve)
                '''
                try:
                    print(abs(energy_sampled-energy_init))
                except TypeError:
                    continue
                '''
                if energy_sampled != None and force_sampled.all() != None:
                    delta_E_sampled = abs(energy_sampled-energy_0)
                    
                    if delta_E_sampled <= delta_E_max:
                        clash = False
                        energy_max = False
                        
        coords_sampled_conf.append(c_tot)
        elements_sampled_conf.append(elements_init)
        
        energies_sampled_conf.append(energy_sampled)
        delta_energies_sampled_conf.append(abs(energy_sampled-energy_0))
        forces_sampled_conf.append(force_sampled)
        c_sum.append(ci_sum)
        c_i_all.append(c_i)
        r_i_all.append(r_i_A)
        
        relative_bonds.append(relative_bond_lengths)
    #print('relative bond within do sampling', relative_bonds)
        
    return coords_sampled_conf, elements_sampled_conf, energies_sampled_conf, delta_energies_sampled_conf, forces_sampled_conf, c_sum,c_i_all, r_i_all, relative_bonds
            
