#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import uuid
import os
import random
from sys import exit
import numpy as np
import scipy.spatial as scsp

import mol_data_generation.utils.file_utils as fu
import mol_data_generation.utils.xtb_fcts as xtb
from rdkit import Chem
from rdkit.Chem import AllChem



def draw_two_random_mols(AA_names, smiles_list, num_systems): 
    dict_in = {}
    init_dict_smiles = dict(zip(AA_names, smiles_list))

    for i in range(num_systems):
        dipeptide_choice = np.random.choice(AA_names, 2, replace=True)
        dipeptide_string = '{}_{}'.format(dipeptide_choice[0],dipeptide_choice[1])                
        
        dict_in[dipeptide_string] = [init_dict_smiles[dipeptide_choice[0]], init_dict_smiles[dipeptide_choice[1]]]
    return dict_in

def draw_two_mols(AA_names, smiles_list, AA_systems):
    dict_in = {}
    init_dict_smiles = dict(zip(AA_names, smiles_list))
    
    for system in AA_systems:
        dipeptide_string = '{}_{}'.format(system[0],system[1]) 
        dict_in[dipeptide_string] = [init_dict_smiles[system[0]], init_dict_smiles[system[1]]]
    return dict_in

def draw_three_random_mols(AA_names,smiles_list, num_systems):
    dict_in = {}
    init_dict_smiles = dict(zip(AA_names, smiles_list))
    
    for i in range(num_systems):
        tripeptide_choice = np.random.choice(AA_names, 3, replace=True)
        tripeptide_string = '{}_{}_{}'.format(tripeptide_choice[0],tripeptide_choice[1],tripeptide_choice[2])
        
        dict_in[tripeptide_string] = [init_dict_smiles[tripeptide_choice[0]], init_dict_smiles[tripeptide_choice[1]], init_dict_smiles[tripeptide_choice[2]]]
    return dict_in

def draw_three_mols(AA_names,smiles_list, AA_systems):
    dict_in = {}
    init_dict_smiles = dict(zip(AA_names, smiles_list))
    
    for system in AA_systems:
        tripeptide_string = '{}_{}_{}'.format(system[0],system[1],system[2])
        
        dict_in[tripeptide_string] = [init_dict_smiles[system[0]], init_dict_smiles[system[1]], init_dict_smiles[system[2]]]
    return dict_in

def draw_two_mols_from_dict(initial_dict, new_systems):
    dict_in = {}
    
    for system in new_systems:
        
        choice_string = '{}_{}'.format(system[0],system[1])
        
        dict_in[choice_string] = [initial_dict[system[0]], initial_dict[system[1]]]
    
    return dict_in

def draw_three_mols_from_dict(initial_dict, new_systems):
    dict_in = {}
    
    for system in new_systems:
        
        choice_string = '{}_{}_{}'.format(system[0],system[1], system[2])
        
        dict_in[choice_string] = [initial_dict[system[0]], initial_dict[system[1]], initial_dict[system[2]]]
    
    return dict_in

def draw_two_random_mols_from_dict(initial_dict, num_systems):
    dict_in = {}
    
    for i in range(num_systems):
        choice = random.choices(list(initial_dict), k= 2)
        choice_string = '{}_{}'.format(choice[0],choice[1])
        
        dict_in[choice_string] = [initial_dict[choice[0]], initial_dict[choice[1]]]
    
    return dict_in
    
def draw_three_random_mols_from_dict(initial_dict, num_systems):
    dict_in = {}
    
    for i in range(num_systems):
        choice = random.choices(list(initial_dict), k= 3)
        choice_string = '{}_{}_{}'.format(choice[0],choice[1], choice[2])
        
        dict_in[choice_string] = [initial_dict[choice[0]], initial_dict[choice[1]], initial_dict[choice[2]]]
    
    return dict_in

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def get_coords_from_smiles(smiles, suffix, conversion_method):
    if conversion_method=="any":
        to_try = ["rdkit", "molconvert", "obabel"]
    elif conversion_method=="rdkit":
        to_try = ["rdkit"]
    elif conversion_method=="molconvert":
        to_try = ["molconvert"]
    elif conversion_method=="obabel":
        to_try = ["obabel"]

    error=""
    for m in to_try:
        #print("   ---   try to convert %s to 3D using %s (of %s)"%(smiles, m, str(to_try)))
        if m=="molconvert":

            if which("molconvert") != None:

                coords, elements = get_coords_from_smiles_marvin(smiles, suffix)
                if coords is None or elements is None:
                    error+=" molconvert_failed "
                    pass
                else:
                    if abs(np.max(coords.T[2])-np.min(coords.T[2]))>0.01:
                        print("   ---   conversion done with molconvert")
                        return(coords, elements)
                    else:
                        error+=" molconvert_mol_flat "
                        pass
                        #print("WARNING: molconvert produced a flat molecule. proceed with other methods (obabel or rdkit)")
            else:
                error+=" molconvert_not_available "

        if m=="obabel":
            if which("obabel") != None:
                #print("use obabel")
                coords, elements = get_coords_from_smiles_obabel(smiles, suffix)
                if coords is None or elements is None:
                    error+=" obabel_failed "
                    pass
                else:
                    if abs(np.max(coords.T[2])-np.min(coords.T[2]))>0.01:
                        print("   ---   conversion done with obabel")
                        return(coords, elements)
                    else:
                        error+=" obabel_failed "
                        pass

            else:
                error+=" obabel_not_available "

        if m=="rdkit":
            #print("use rdkit")
            coords, elements = get_coords_from_smiles_rdkit(smiles, suffix)
            if coords is None or elements is None:
                error+=" rdkit_failed "
                pass
            else:
                if abs(np.max(coords.T[2])-np.min(coords.T[2]))>0.01:
                    #print("   ---   conversion done with rdkit")
                    return(coords, elements)
                else:
                    error+=" rdkit_failed "
                    pass

    exit("ERROR: NO 3D conversion worked: %s"%(error))

def get_coords_from_smiles_obabel(smiles, suffix):
    name=uuid.uuid4()

    if not os.path.exists("input_structures%s"%(suffix)):
        try:
            os.makedirs("input_structures%s"%(suffix))
        except:
            pass

    filename="input_structures%s/%s.xyz"%(suffix, name)
    os.system("obabel -:\"%s\" --gen3D -oxyz > %s"%(smiles, filename))
    if not os.path.exists(filename):
        return(None, None)
        #print("ERROR: could not convert %s to 3D using obabel. Exit!"%(smiles))
        #exit()

    coords, elements = fu.readXYZ(filename)
    if len(coords)==0:
        return(None, None)
        #print("ERROR: could not convert %s to 3D using obabel. Exit!"%(smiles))
        #exit()
    os.system("rm %s"%(filename))
    return(coords, elements)


def get_coords_from_smiles_rdkit(smiles, suffix):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        return(None, None)
        #print("could not convert %s to rdkit molecule. Exit!"%(smiles))
        #exit()
    try:
        m = Chem.AddHs(m)
    except:
        return(None, None)
        #print("ERROR: could not add hydrogen to rdkit molecule of %s. Exit!"%(smiles))
        #exit()
    try:
        conformer_id = AllChem.EmbedMolecule(m, useRandomCoords=True)
        if conformer_id < 0:
            print(f'Could not embed molecule {smiles}.')
            raise ValueError(f'Could not embed molecule {smiles}.')
    except:
        return(None, None)
        #print("ERROR: could not calculate 3D coordinates from rdkit molecule %s. Exit!"%(smiles))
        #exit()
    try:
        block=Chem.MolToMolBlock(m)
        blocklines=block.split("\n")
        coords=[]
        elements=[]
        for line in blocklines[4:]:
            if len(line.split())==4:
                break
            elements.append(line.split()[3])
            coords.append([float(line.split()[0]),float(line.split()[1]),float(line.split()[2])])
        coords=np.array(coords)
        mean = np.mean(coords, axis=0)
        distances = scsp.distance.cdist([mean],coords)[0]
        if np.max(distances)<0.1:
            return(None, None)
            #print("ERROR: something is wrong with rdkit molecule %s. Exit!"%(smiles))
            #print("%i\n"%(len(coords)))
            #for atomidx, atom in enumerate(coords):
            #    print("%s %f %f %f"%(elements[atomidx], atom[0], atom[1], atom[2]))
            #exit()
            
    except:
        return(None, None)
        #print("ERROR: could not read xyz coordinates from rdkit molecule %s. Exit!"%(smiles))
        #exit()
    return(coords, elements)




def get_coords_from_smiles_marvin(smiles, suffix):

    name=uuid.uuid4()

    if not os.path.exists("tempfiles%s"%(suffix)):
        try:
            os.makedirs("tempfiles%s"%(suffix))
        except:
            pass
    if not os.path.exists("input_structures%s"%(suffix)):
        try:
            os.makedirs("input_structures%s"%(suffix))
        except:
            pass

    outfile=open("tempfiles%s/%s.smi"%(suffix, name),"w")
    outfile.write("%s\n"%(smiles))
    outfile.close()

    path_here=os.getcwd()
    os.system("molconvert -2 mrv:+H %s/tempfiles%s/%s.smi > tempfiles%s/%s.mrv"%(path_here,suffix, name, suffix, name))
    filename="tempfiles%s/%s.mrv"%(suffix, name)
    if not os.path.exists(filename):
        os.system("rm tempfiles%s/%s.smi"%(suffix, name))
        return(None, None)
        #print("ERROR: could not convert %s to 2D (mrv) using marvin. Exit!"%(smiles))
        #exit()

    os.system("molconvert -3 xyz %s/tempfiles%s/%s.mrv > input_structures%s/%s.xyz"%(path_here, suffix, name, suffix, name))
    filename="input_structures%s/%s.xyz"%(suffix, name)
    if not os.path.exists(filename):
        os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv"%(suffix, name, suffix, name))
        return(None, None)
        #print("ERROR: could not convert %s to 3D (xyz) using marvin. Exit!"%(smiles))
        #exit()

    coords, elements = fu.readXYZ(filename)
    if len(coords)==0:
        os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv input_structures%s/%s.xyz"%(suffix, name, suffix, name, suffix, name))
        print("ERROR: could not convert %s to 3D (coords in empty) using marvin. Exit!"%(smiles))
        #return(None, None)
        #exit()
    os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv input_structures%s/%s.xyz"%(suffix, name, suffix, name, suffix, name))
    return(coords, elements)

### conformer ###
def choose_conformers(coords_all, elements_all, energies_all, number,outdir, conf_types = 'low_random'):
    # also save coords in to list of chosen conf and energy - energy also for eq?
    coords_out, elements_out = [], []
    #energy_in = xtb.single_point_energy(coords_in, elements_in)
    energy_out = []
    if conf_types == 'low_random':
        # choose 1/2 lowest and other half random conformers
        
        if len(elements_all) > number:       
            coords_conf,  elements_conf, energy_conf = get_conformers(coords_all, elements_all, energies_all, number)
            for i in range(len(coords_conf)):                
                coords_out.append(coords_conf[i])
                elements_out.append(elements_conf[i])
                energy_out.append(energy_conf[i])
        else:
            print('Number of available conformers smaller than given conf number.')
            for i in range(len(elements_all)):
                coords_out.append(coords_all[i])
                elements_out.append(elements_all[i])
                energy_out.append(energies_all[i])
                
    if conf_types == 'lowest':
        # choose number lowest conformers       
        if len(elements_all) > number:
            for i in range(number):
                coords_out.append(coords_all[i])
                elements_out.append(elements_all[i])
                energy_out.append(energies_all[i])
        else:
            print('Number of available conformers smaller than given conf number.')
            for i in range(len(elements_all)):
                coords_out.append(coords_all[i])
                elements_out.append(elements_all[i])
                energy_out.append(energies_all[i])
    file_out = '{}coords_conformers_selected.xyz'.format(outdir)
    fu.exportXYZs(coords_out, elements_out, file_out)    
    np.save('{}energy_conformers_selected.npy'.format(outdir), np.array(energy_out), allow_pickle=True)
    
    return energy_out

# chose X/2 conformers with lowest energy and X/2 others randomly
def get_conformers(coords_all,elements_all,energies_all, num_confs):
    coords_elem_conf_all = list(zip(coords_all,elements_all, energies_all))
    coords_elem_conf = []

    for i in range(0,int(num_confs/2)):
        coords_elem_conf.append(coords_elem_conf_all[i])
    
    unzip_conf = list(zip(*coords_elem_conf))
    coords_conf_low = list(unzip_conf[0])
    elements_conf_low = list(unzip_conf[1])
    energies_conf_low = list(unzip_conf[2])
    
    coords_elem_conf_rest = coords_elem_conf_all[int(num_confs/2):]
    conf_random_list = random.sample(coords_elem_conf_rest,int(num_confs/2))
    unzip_conf_r = list(zip(*conf_random_list))
    coords_conf_r = list(unzip_conf_r[0])
    elements_conf_r = list(unzip_conf_r[1])
    energies_conf_r = list(unzip_conf_r[2])
    
    coords_conf = coords_conf_low + coords_conf_r
    elements_conf = elements_conf_low + elements_conf_r
    energies_conf = energies_conf_low + energies_conf_r
    return coords_conf, elements_conf, energies_conf