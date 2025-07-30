#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import math
import mol_data_generation.utils.xtb_fcts as xtb
from scipy.spatial.distance import euclidean

import numpy as np



def calculate_center_distance(vector1, vector2):
    center = [(vector1[0] + vector2[0]) / 2, (vector1[1] + vector2[1]) / 2, (vector1[2] + vector2[2]) / 2]
    distance = euclidean(vector1, center)
    return center, distance


def random_point_on_sphere_surface_center(radius, coords_center):
    # Generate uniformly distributed values of u and v
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    
    # Convert u and v to spherical coordinates
    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = coords_center[0] + radius * math.sin(phi) * math.cos(theta)
    y = coords_center[1] + radius * math.sin(phi) * math.sin(theta)
    z = coords_center[2] + radius * math.cos(phi)
    
    return [x, y, z]


def check_dist(coord1, coords_system, skip_ind = []):
    distances = []
    for j, coord2 in enumerate(coords_system):
        if skip_ind != []:
            if j in skip_ind:
                continue
        distance = euclidean(coord1, coord2)
        distances.append(distance)
    return distances

def get_new_h1_position(r_max, center0, coords_system,elements_system, idx_h1, idx_r0, idx_r2, hat_dist, chrg = 0, unp = 1, solve= None, check_bonds = False):
    
    clash = True
    # add energy and force check to condition
    while clash:
    
        radius_sph = random.uniform(0,0.5*r_max)
        #print(radius_sph)
        h1_new = random_point_on_sphere_surface_center(radius_sph, center0)
        #print(h1_new)
        
        coords_shift = np.copy(coords_system)
        
        coords_shift[idx_h1] = h1_new
    
        ## add clash checks
        # check H1 distance to all other coords

        distances = check_dist(coords_shift[idx_h1], coords_shift, skip_ind = [idx_h1])

        if min(distances) >= hat_dist:
            
            # check h1 bound?? 
            # if num_bonds == 1: unp_e = 1
            # if num_bonds == 0: unp_e = 3
            # input: coords_shift[idx_h1] [idx_rad0] [idx_rad_2] elements 
            
            if check_bonds:
                coords_check = [coords_shift[idx_h1], coords_shift[idx_r0], coords_shift[idx_r2]]
                elements_check = [elements_system[idx_h1], elements_system[idx_r0], elements_system[idx_r2]]
                bond_num = func_check_bonds(coords_check, elements_check)
                if bond_num == 0:
                    unp = 3
                if bond_num == 1:
                    unp = 1
                if bond_num >= 2 : #!= 0 and bond_num != 1 == 3??
                    unp = 0
                    #print('bond num >= 2, check system')
            else:
                unp = 1
            # calcualte xtb energy and force
            energy_sampled = xtb.single_point_energy(coords_shift, elements_system, charge = chrg, unp_e = unp, solvent = solve)
            force_sampled = xtb.single_force(coords_shift, elements_system, charge = chrg, unp_e = unp, solvent = solve)
            
            try:
                if force_sampled.all() != None:
                    if len(force_sampled) == len(coords_shift):
                        forces = True
            except:
                if force_sampled == None:
                    #print('forces none')
                    forces = False
                
            if energy_sampled != None and forces:
                clash = False
            #print('no clashes')
    
    return coords_shift, energy_sampled, force_sampled


def get_final_state(h1_idx, coords, elements,coords_h2, chrg, solve):

    coords_final = coords.copy()
    elements_final = elements.copy()
    
    coords_final[h1_idx] = coords_h2
 


    unp = 1
    energy_final = xtb.single_point_energy(coords_final, elements_final, charge = chrg, unp_e = unp, solvent = solve)
    force_final = xtb.single_force(coords_final, elements_final, charge = chrg, unp_e = unp, solvent = solve)
    
    
    try:
        if force_final.all() != None:
            if len(force_final) == len(coords_final):
                forces = True
    except:
        if force_final == None:
            #print('forces none')
            forces = False
        
    if energy_final != None and forces:
        final_state = True
    else:
        final_state = False
    #print('no clashes')
    
    return coords_final, elements_final, energy_final, force_final, final_state

def get_atms_outside_sphere(coords, idx_h1, r_cutoff):
    freeze_list = []
    for j, coord in enumerate(coords):
        if j == idx_h1:
            #continue # if dont want to freeze H0
            # freeze also h0 in middle
            freeze_list.append(j)
        else:
            #print(coords, idx_h1, coord)
            distance = euclidean(coords[idx_h1], coord) #s[j]
            
            if distance >= r_cutoff:
                freeze_list.append(j)
    return freeze_list


def select_steps(coords_all, elements_all, energies_all, x, y, z):
    coords_selected = coords_all[:y]  # Include the first 'y' elements
    elements_selected = elements_all[:y]
    energies_selected = energies_all[:y]
    
    # Append every 'x'th element in between the first 'y' and last 'z' elements
    for i in range(y, len(coords_all) - z, x):
        coords_selected.append(coords_all[i])
        elements_selected.append(elements_all[i])
        energies_selected.append(energies_all[i])
        
    coords_selected.extend(coords_all[-z:])  # Include the last 'z' elements
    elements_selected.extend(elements_all[-z:])
    energies_selected.extend(energies_all[-z:])
    
    return coords_selected, elements_selected, energies_selected


def get_h0_r1_r2_distances(coords_all, idx_h0, idx_r1, idx_r2):
    
    h0_r1_dist = euclidean(coords_all[idx_h0], coords_all[idx_r1])
    h0_r2_dist = euclidean(coords_all[idx_h0], coords_all[idx_r2])
    r1_r2_dist = euclidean(coords_all[idx_r1], coords_all[idx_r2])
    return h0_r1_dist, h0_r2_dist, r1_r2_dist



def func_check_bonds(coords, elements, force_bonds=False, forced_bonds=[]):

    # covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
    # values for metals decreased by 10% according to Robert Paton's Sterimol implementation
    rcov = {
        "H": 0.34, "He": 0.46, "Li": 1.2, "Be": 0.94, "B": 0.77, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, "Na": 1.4, "Mg": 1.25, "Al": 1.13, "Si": 1.04, "P": 1.1, "S": 1.02, "Cl": 0.99, "Ar": 0.96, "K": 1.76, "Ca": 1.54, "Sc": 1.33, "Ti": 1.22, "V": 1.21, "Cr": 1.1, "Mn": 1.07, "Fe": 1.04, "Co": 1.0, "Ni": 0.99, "Cu": 1.01, "Zn": 1.09, "Ga": 1.12, "Ge": 1.09, "As": 1.15, "Se": 1.1, "Br": 1.14, "Kr": 1.17, "Rb": 1.89, "Sr": 1.67, "Y": 1.47, "Zr": 1.39, "Nb": 1.32, "Mo": 1.24, "Tc": 1.15, "Ru": 1.13, "Rh": 1.13, "Pd": 1.19, "Ag": 1.15, "Cd": 1.23, "In": 1.28, "Sn": 1.26, "Sb": 1.26, "Te": 1.23, "I": 1.32, "Xe": 1.31, "Cs": 2.09, "Ba": 1.76, "La": 1.62, "Ce": 1.47, "Pr": 1.58, "Nd": 1.57, "Pm": 1.56, "Sm": 1.55, "Eu": 1.51, "Gd": 1.52, "Tb": 1.51, "Dy": 1.5, "Ho": 1.49, "Er": 1.49, "Tm": 1.48, "Yb": 1.53, "Lu": 1.46, "Hf": 1.37, "Ta": 1.31, "W": 1.23, "Re": 1.18, "Os": 1.16, "Ir": 1.11, "Pt": 1.12, "Au": 1.13, "Hg": 1.32, "Tl": 1.3, "Pb": 1.3, "Bi": 1.36, "Po": 1.31, "At": 1.38, "Rn": 1.42, "Fr": 2.01, "Ra": 1.81, "Ac": 1.67, "Th": 1.58, "Pa": 1.52, "U": 1.53, "Np": 1.54, "Pu": 1.55
    }

    # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    natom = len(coords)
    #max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom, natom))
    bonds = []
    for i in range(0, natom):
        if elements[i] not in rcov.keys():
            continue
        for iat in range(0, natom):
            if elements[iat] not in rcov.keys():
                continue
            if iat != i:
                dx = coords[iat][0] - coords[i][0]
                dy = coords[iat][1] - coords[i][1]
                dz = coords[iat][2] - coords[i][2]
                r = np.linalg.norm([dx, dy, dz])
                rco = rcov[elements[i]]+rcov[elements[iat]]
                rco = rco*k2
                rr = rco/r
                damp = 1.0/(1.0+np.math.exp(-k1*(rr-1.0)))
                if damp > 0.85:  # check if threshold is good enough for general purpose
                    conmat[i, iat], conmat[iat, i] = 1, 1
                    pair = [min(i, iat), max(i, iat)]
                    if pair not in bonds:

                        # add some empirical rules here:
                        is_bond = True
                        elements_bond = [elements[pair[0]], elements[pair[1]]]
                        if "Pd" in elements_bond:
                            if not ("As" in elements_bond or "Cl" in elements_bond or "P" in elements_bond):
                                is_bond = False
                        elif "Ni" in elements_bond:
                            if not ("C" in elements_bond or "P" in elements_bond):
                                is_bond = False
                        elif "As" in elements_bond:
                            if not ("Pd" in elements_bond or "F" in elements_bond):
                                is_bond = False
                        if is_bond:
                            bonds.append(pair)

    # remove bonds in certain cases
    bonds_to_remove = []
    # P has too many bonds incl one P-Cl bond which is probably to the spacer
    P_bond_indeces = []
    P_bonds_elements = []
    for bondidx, bond in enumerate(bonds):
        elements_bond = [elements[bond[0]], elements[bond[1]]]
        if "P" in elements_bond:
            P_bond_indeces.append(bondidx)
            P_bonds_elements.append(elements_bond)
    if len(P_bond_indeces) > 4:
        print("WARNING: found a P with more than 4 bonds. try to remove one")
        if ["P", "Cl"] in P_bonds_elements:
            bonds_to_remove.append(
                P_bond_indeces[P_bonds_elements.index(["P", "Cl"])])
        elif ["Cl", "O"] in P_bonds_elements:
            bonds_to_remove.append(
                P_bond_indeces[P_bonds_elements.index(["Cl", "P"])])

    # Cl-Cl bonds
    for bondidx, bond in enumerate(bonds):
        elements_bond = [elements[bond[0]], elements[bond[1]]]
        if ["Cl", "Cl"] == elements_bond:
            bonds_to_remove.append(bondidx)

    bonds_new = []
    for bondidx, bond in enumerate(bonds):
        if bondidx not in bonds_to_remove:
            bonds_new.append(bond)
    bonds = bonds_new

    # very special case where the C atoms of Ni(CO)3 make additional bonds to lone pairs of N
    # get the indeces of the Ni(CO)3 C bonds
    c_atom_indeces = []
    for bondidx, bond in enumerate(bonds):
        elements_bond = [elements[bond[0]], elements[bond[1]]]
        if "Ni" == elements_bond[0] and "C" == elements_bond[1]:
            # check if this C has a bond to O
            for bondidx2, bond2 in enumerate(bonds):
                elements_bond2 = [elements[bond2[0]], elements[bond2[1]]]
                if bond[1] in bond2 and "O" in elements_bond2:
                    c_atom_indeces.append(bond[1])
                    break
        elif "Ni" == elements_bond[1] and "C" == elements_bond[0]:
            for bondidx2, bond2 in enumerate(bonds):
                elements_bond2 = [elements[bond2[0]], elements[bond2[1]]]
                if bond[0] in bond2 and "O" in elements_bond2:
                    c_atom_indeces.append(bond[0])
                    break

    if len(c_atom_indeces) > 0:
        bonds_to_remove = []
        for c_atom_idx in c_atom_indeces:
            for bondidx, bond in enumerate(bonds):
                elements_bond = [elements[bond[0]], elements[bond[1]]]
                if c_atom_idx in bond and "N" in elements_bond:
                    bonds_to_remove.append(bondidx)
        bonds_new = []
        for bondidx, bond in enumerate(bonds):
            if bondidx not in bonds_to_remove:
                bonds_new.append(bond)
        bonds = bonds_new

    '''
    # add forced bonds
    if forced_bonds:
        for b in forced_bonds:
            b_to_add = [min(b), max(b)]
            if b_to_add not in bonds:
                print("WARNING: was forced to add a %s-%s bond that was not detected automatically." %
                      (elements[b_to_add[0]], elements[b_to_add[1]]))
                bonds.append(b_to_add)

    # add bonds for atoms that are floating around
    indeces_used = []
    for b in bonds:
        indeces_used.append(b[0])
        indeces_used.append(b[1])
    indeces_used = list(set(indeces_used))
    if len(indeces_used) < len(coords):
        for i in range(len(coords)):
            if i not in indeces_used:
                e = elements[i]
                c = coords[i]
                distances = scsp.distance.cdist([c], coords)[0]
                next_atom_indeces = np.argsort(distances)[1:]
                for next_atom_idx in next_atom_indeces:
                    b_to_add = [min([i, next_atom_idx]),
                                max([i, next_atom_idx])]
                    elements_bond = [
                        elements[b_to_add[0]], elements[b_to_add[1]]]
                    if elements_bond not in [["Cl", "H"], ["H", "Cl"], ["Cl", "F"], ["F", "Cl"], ["F", "H"], ["H", "F"], ["Pd", "F"], ["F", "Pd"], ["H", "H"], ["F", "F"], ["Cl", "Cl"]]:
                        print("WARNING: had to add a %s-%s bond that was not detected automatically." %
                              (elements[b_to_add[0]], elements[b_to_add[1]]))
                        bonds.append(b_to_add)
                        break
                    else:
                        pass
    '''
    #print(len(bonds), bonds)
    return(len(bonds))


