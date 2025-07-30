#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# goals

# initial coord generation from smiles with or without reaction template 

# write config file with file structure/ names of generated molecules

# conformer generation (choose method) with option to choose number and specific conformers 

import os
import json
import glob
import numpy as np
from rdkit import Chem

import mol_data_generation.mol_construction.construction_fcts as construct
import mol_data_generation.utils.file_utils as fu
import mol_data_generation.utils.xtb_fcts as xtb
from pathlib import Path
import matplotlib.pyplot as plt

from rdkit.Chem.AllChem import ReactionFromSmarts as smt2rxn
from rdkit.Chem import MolToSmiles as mol2smi



class ConstructInitData():
    
    
    
    def __init__(self, outdir, dict_smiles = None, solvent = None, num_threads = 1):
        
        self.dict_smiles = self.check_dict(dict_smiles)
        self.outdir = self.check_dir(outdir)
        self.solvent = solvent
        self.num_threads = num_threads
        
        self.system_type = None
        self.reaction_type = None
        self.reaction_smart = None
        
        self.init_smart = None
        self.init_systems = None
        
        #self.list_dict_in_steps = []
        
        

    
    def generate_smiles_dict(self, react_type, settings):
        
        if settings['reaction_steps'] == 1:
            
            # naming?? settings['naming'] one -> add naming convention 
            if settings['input_type'] == 'one' and settings['systems'] != 'none': 
                dict_in = {}
                init_dict_smiles = dict(zip(settings['init_names'], settings['init_smiles']))
                if type(settings['systems']) == int:
                    mol_names = np.random.choice(settings['init_names'], size= settings['systems'], replace= False)
                    for name in mol_names:
                        dict_in[name+'{}'.format(settings['naming'])] = init_dict_smiles[name]
                    
                if type(settings['systems']) == list:
                    for name in settings['systems']:
                        dict_in[name+'{}'.format(settings['naming'])] = init_dict_smiles[name]
                    
                #print(dict_in) ####
                self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'single')
               
            
            if settings['input_type'] == 'one' and settings['systems'] == 'none':
                print('need to provide init systems and number or list')
            
            
            if settings['input_type'] == 'two':
                
                
                if type(settings['systems']) == int:
                    
                    dict_in = construct.draw_two_random_mols(settings['init_names'], settings['init_smiles'], settings['systems'])
                    
                    self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'two')
                    # do reaction and add to dict_smiles
                    
                if type(settings['systems']) == list:
                    
                    dict_in = construct.draw_two_mols(settings['init_names'], settings['init_smiles'], settings['systems'])
                    
                    self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'two')
            
    
            if settings['input_type'] == 'three':
                
                
                if type(settings['systems']) == int:
                    
                    dict_in = construct.draw_three_random_mols(settings['init_names'], settings['init_smiles'], settings['systems'])
                    
                    self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'three')
                    
                    
                if type(settings['systems']) == list:
                    
                    dict_in = construct.draw_three_mols(settings['init_names'], settings['init_smiles'], settings['systems'])
                    
                    self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'three')            
        
                
        if settings['reaction_steps'] > 1: 
            list_dict_in_steps = []
            #dict_in = {}
            for i in range(settings['reaction_steps']):
                settings_step = settings['step_{}'.format(i+1)]
                
                # need to add all results from reactions i into one dict as input for smiles_generation(dict_in)
                
                if i == 0: # erster Schritt start from system types/ list , following steps from dictionary until final step
                    # check number reactions
                    if settings_step['number_reactions'] == 1:
                        # dict_in from int/list , out: dict_interm with names: smiles wo react type
                        
                        if settings_step['input_type'] == 'one' and settings_step['systems'] != 'none': 
                            dict_in = {}
                            init_dict_smiles = dict(zip(settings_step['init_names'], settings_step['init_smiles']))
                            if type(settings_step['systems']) == int:
                                mol_names = np.random.choice(settings_step['init_names'], size= settings_step['systems'], replace= False)
                                for name in mol_names:
                                    dict_in[name+'{}'.format(settings_step['naming'])] = init_dict_smiles[name]
                                
                            if type(settings_step['systems']) == list:
                                for name in settings_step['systems']:
                                    dict_in[name+'{}'.format(settings_step['naming'])] = init_dict_smiles[name]
                                
                            # change out smiles generation to step = intermediate -> in: dict_in - out: dict_out self.dict_interm_{}.format(i)
                            dict_reaction = self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'single', step = 'intermediate', step_number = 1)
                            
                        
                        if settings_step['input_type'] == 'one' and settings_step['systems'] == 'none':
                            print('need to provide init systems and number or list')
                        
                        
                        if settings_step['input_type'] == 'two':
                            
                            
                            if type(settings_step['systems']) == int:
                                
                                dict_in = construct.draw_two_random_mols(settings_step['init_names'], settings_step['init_smiles'], settings_step['systems'])
                                
                                #self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'two')
                                # do reaction and add to dict_smiles
                                
                            if type(settings_step['systems']) == list:
                                
                                dict_in = construct.draw_two_mols(settings_step['init_names'], settings_step['init_smiles'], settings_step['systems'])
                            
                            # change out smiles generation to step = intermediate -> in: dict_in - out: dict_out self.dict_interm_{}.format(i)
                            dict_reaction = self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'two', step = 'intermediate', step_number = 1)
                            
                
                        if settings_step['input_type'] == 'three':
                            
                            
                            if type(settings_step['systems']) == int:
                                
                                dict_in = construct.draw_three_random_mols(settings_step['init_names'], settings_step['init_smiles'], settings_step['systems'])
                                
                                #self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'three')
                                
                                
                            if type(settings_step['systems']) == list:
                                
                                dict_in = construct.draw_three_mols(settings_step['init_names'], settings_step['init_smiles'], settings_step['systems'])
                            
                            # change out smiles generation to step = intermediate -> in: dict_in - out: dict_out self.dict_interm_{}.format(i)
                            dict_reaction = self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'three', step = 'intermediate', step_number = 1)
                        
                        list_dict_in_steps.append(dict_reaction)
                        
                     
                    else:
                        # iterate through given keywords if number_reactions > 1 settings_step['reaction_keywords']
                        # keywords needed or iterate through keys in settings_step --> need keywords! because have also other dict entry on this
                        # level
                        # for each:
                        # step = intermediate/ adapted smiles gener fct, dict_in from int/list out: dict_intermediate but only names and smiles,  
                        # not react type
                        dict_reaction = {}
                        for reaction in settings_step['reaction_keywords']:
                            dict_step_reaction = {}
                            settings_step_reaction = settings_step[reaction]
                            if settings_step_reaction['input_type'] == 'none':
                                
                                # choose and just add dictionary to dictionary for next step
                                init_dict_smiles = dict(zip(settings_step_reaction['init_names'], settings_step_reaction['init_smiles']))
                                if type(settings_step_reaction['systems']) == int:
                                    mol_names = np.random.choice(settings_step_reaction['init_names'], size= settings_step_reaction['systems'], replace= False)
                                    for name in mol_names:
                                        dict_step_reaction[name] = init_dict_smiles[name]
                                if type(settings_step_reaction['systems']) == list:
                                    for name in settings_step_reaction['systems']:
                                        dict_step_reaction[name] = init_dict_smiles[name]
                                
                                for key in dict_step_reaction:
                                    dict_reaction[key] = dict_step_reaction[key]
                                
                            if settings_step_reaction['input_type'] == 'two':
                                
                                if type(settings_step_reaction['systems']) == int:
                                    
                                    dict_in = construct.draw_two_random_mols(settings_step_reaction['init_names'], settings_step_reaction['init_smiles'], settings_step_reaction['systems'])
                                    
                                    #self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'two')
                                    # do reaction and add to dict_smiles
                                    
                                if type(settings_step_reaction['systems']) == list:
                                    
                                    dict_in = construct.draw_two_mols(settings_step_reaction['init_names'], settings_step_reaction['init_smiles'], settings_step_reaction['systems'])
                                
                                # change out smiles generation to step = intermediate -> in: dict_in - out: dict_out self.dict_interm_{}.format(i)
                                dict_step_reaction = self.smiles_generation(react_type, settings_step_reaction['smart_reaction'], dict_in, mode = 'two', step = 'intermediate', step_number = 1)
                                
                                for key in dict_step_reaction:
                                    dict_reaction[key] = dict_step_reaction[key]

                            if settings_step_reaction['input_type'] == 'three':
                                
                                if type(settings_step_reaction['systems']) == int:
                                    
                                    dict_in = construct.draw_three_random_mols(settings_step_reaction['init_names'], settings_step_reaction['init_smiles'], settings_step_reaction['systems'])
                                    
                                    #self.smiles_generation(react_type, settings['smart_reaction'], dict_in, mode = 'two')
                                    # do reaction and add to dict_smiles
                                    
                                if type(settings_step_reaction['systems']) == list:
                                    
                                    dict_in = construct.draw_three_mols(settings_step_reaction['init_names'], settings_step_reaction['init_smiles'], settings_step_reaction['systems'])
                                
                                # change out smiles generation to step = intermediate -> in: dict_in - out: dict_out self.dict_interm_{}.format(i)
                                dict_step_reaction = self.smiles_generation(react_type, settings_step_reaction['smart_reaction'], dict_in, mode = 'three', step = 'intermediate', step_number = 1)
                                
                                for key in dict_step_reaction:
                                    dict_reaction[key] = dict_step_reaction[key]                        
                        
                        
                        self.list_dict_in_steps.append(dict_reaction)
                    
                if i > 0 and i < settings['reaction_steps']-1:
                    # intermediate 
                    # reaction number == 1
                    # in: dict_interm_{}.format(i-1) out: dict_interm_new create self.dict_interm_{}.format(i)
                    # dict_in = self.list_dict_in_steps[i-1]
                    
                    if settings_step['number_reactions'] == 1:
                        # dict_in from int/list , out: dict_interm with names: smiles wo react type
                        
                        if settings_step['systems'] == 'generated':
                            # reaction input type one?? or make extra function to also randomly make pairs from earlier systems dict?
                            if settings_step['input_type'] == 'one': #reaction input type                                
                                initial_dict = list_dict_in_steps[i-1]
                                ## naming! change dict in key names
                                dict_in = {}
                                for key in initial_dict:
                                    dict_in[key+'{}'.format(settings_step['naming'])] = initial_dict[key]                                     
                                
                                dict_step_reaction = self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'single', step = 'intermediate', step_number = i)
                                
                                
                            if settings_step['input_type'] == 'two':
                                init_dict = list_dict_in_steps[i-1]
                                # naming connect _ two chosen systems from dict
                                if type(settings_step['new_init_systems']) == int:
                                    
                                    dict_in = construct.draw_two_random_mols_from_dict(init_dict, settings_step['new_init_systems'])
                                if type(settings_step['new_init_systems']) == list:
                                    
                                    dict_in = construct.draw_two_mols_from_dict(init_dict, settings_step['new_init_systems'])                                
                                
                                dict_step_reaction = self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'two', step = 'intermediate', step_number = i)

                            if settings_step['input_type'] == 'three':
                                init_dict = list_dict_in_steps[i-1]
                                # naming connect _ two chosen systems from dict
                                if type(settings_step['new_init_systems']) == int:
                                    
                                    dict_in = construct.draw_three_random_mols_from_dict(init_dict, settings_step['new_init_systems'])
                                if type(settings_step['new_init_systems']) == list:
                                    
                                    dict_in = construct.draw_three_mols_from_dict(init_dict, settings_step['new_init_systems'])                                
                                
                                dict_step_reaction = self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'three', step = 'intermediate', step_number = i)                                
                            
                            
                            list_dict_in_steps.append(dict_step_reaction)
                        
                        else:
                            print('currently intermediate reaction step, need generated systems from earlier step')
                    
                        
                    else:
                        print('currently intermediate reaction step, can only do one reaction')
                    
                    
                    

### final                
                if i == settings['reaction_steps']-1: # step == final
                    # in: dict_interm_{}.format(i-1) out: dict_smiles updated with dict and react type    
                    
                    if settings_step['number_reactions'] == 1:
                        # dict_in from int/list , out: dict_interm with names: smiles wo react type
                        
                        if settings_step['systems'] == 'generated':
                            # reaction input type one?? or make extra function to also randomly make pairs from earlier systems dict?
                            if settings_step['input_type'] == 'one': #reaction input type                                
                                initial_dict = list_dict_in_steps[i-1]
                                
                                ## naming! change dict in key names
                                dict_in = {}
                                for key in initial_dict:
                                    dict_in[key+'{}'.format(settings_step['naming'])] = initial_dict[key]                                     
                                
                                
                                self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'single', step = 'final', step_number = i)
                                
                                
                            if settings_step['input_type'] == 'two':
                                init_dict = list_dict_in_steps[i-1]
                                # naming connect _ two chosen systems from dict
                                if type(settings_step['new_init_systems']) == int:
                                    
                                    dict_in = construct.draw_two_random_mols_from_dict(init_dict, settings_step['new_init_systems'])
                                if type(settings_step['new_init_systems']) == list:
                                    
                                    dict_in = construct.draw_two_mols_from_dict(init_dict, settings_step['new_init_systems'])                                
                                
                                self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'two', step = 'final', step_number = i)

                            if settings_step['input_type'] == 'three':
                                init_dict = list_dict_in_steps[i-1]
                                # naming connect _ two chosen systems from dict
                                if type(settings_step['new_init_systems']) == int:
                                    
                                    dict_in = construct.draw_three_random_mols_from_dict(init_dict, settings_step['new_init_systems'])
                                if type(settings_step['new_init_systems']) == list:
                                    
                                    dict_in = construct.draw_three_mols_from_dict(init_dict, settings_step['new_init_systems'])                                
                                
                                self.smiles_generation(react_type, settings_step['smart_reaction'], dict_in, mode = 'three', step = 'final', step_number = i)                                
                            
                            
                          
                        
                        else:
                            print('currently final reaction step, need generated systems from earlier step')
                    
                        
                    else:
                        print('currently final reaction step, can only do one reaction')
                    

        
    
    def smiles_generation(self, react_type, reaction_smarts, dict_in, mode = 'single', step = 'final', step_number = 1):
            
        # input: reaction_smarts, dict in {'alanine_glycine':[smiles1,smiles2], 'name2':[]... }
        if step == 'final':
            dict_reaction = {}
            if mode == 'two':
                for system_name_key in dict_in:
                
                    smiles_mol0 = dict_in[system_name_key][0]
                    smiles_mol1 = dict_in[system_name_key][1]
                
                    mol0 = Chem.MolFromSmiles(smiles_mol0)
                    mol1 = Chem.MolFromSmiles(smiles_mol1)
                    
                    reactants = (mol0, mol1)
                    
                    reaction_rdkit = smt2rxn(reaction_smarts)
                    products_react = reaction_rdkit.RunReactants(reactants) 
                    result_react = products_react[0][0]
                    smiles_result = mol2smi(result_react)
                    
                    dict_reaction[system_name_key] = smiles_result
                
                
                '''
                if self.dict_smiles == {}: # oder dict_smiles[react_type] = None ?
                    self.dict_smiles[react_type] = dict_reaction
                else:
                    for key in dict_reaction:
                        self.dict[react_type][key] = dict_reaction[key]
                '''
                self.dict_smiles[react_type] = dict_reaction
                
                #return dict_reaction
            
            if mode == 'three':
                for system_name_key in dict_in:
                
                    smiles_mol0 = dict_in[system_name_key][0]
                    smiles_mol1 = dict_in[system_name_key][1]
                    smiles_mol2 = dict_in[system_name_key][2]
                
                    mol0 = Chem.MolFromSmiles(smiles_mol0)
                    mol1 = Chem.MolFromSmiles(smiles_mol1)
                    mol2 = Chem.MolFromSmiles(smiles_mol2)
                    
                    reactants = (mol0, mol1, mol2)
                    
                    reaction_rdkit = smt2rxn(reaction_smarts)
                    products_react = reaction_rdkit.RunReactants(reactants) 
                    result_react = products_react[0][0]
                    smiles_result = mol2smi(result_react)
                    
                    dict_reaction[system_name_key] = smiles_result
    
                self.dict_smiles[react_type] = dict_reaction
    
            if mode == 'single':
                for system_name_key in dict_in:
                
                    smiles_mol0 = dict_in[system_name_key]
                    #print('smiles', smiles_mol0)
                    mol0 = Chem.MolFromSmiles(smiles_mol0)
                    #print('mol', mol0)
                    reactants = (mol0, )
                    #print(system_name_key, reaction_smarts)
                    #print(smiles_mol0)
                    reaction_rdkit = smt2rxn(reaction_smarts)
                    products_react = reaction_rdkit.RunReactants(reactants) 
                    result_react = products_react[0][0]
                    #print('result_react', result_react)
                    smiles_result = mol2smi(result_react)
                    #print('smiles_result', smiles_result)
                    dict_reaction[system_name_key] = smiles_result
    
                self.dict_smiles[react_type] = dict_reaction
                
            
            #smiles_res = construct.rdkit_reaction(reaction_smarts, reactants)
            
            # make into/ add to dictionary
            # dictionary {type1: {name: smiles}, type2: {name:smiles, name:smiles...} ...}
            
            # add type and names to smiles_dict
        
        
        # change out smiles generation to step = intermediate -> in: dict_in - out: dict_out self.dict_interm_{}.format(i)
        if step == 'intermediate': # or else
            dict_reaction = {}
            if mode == 'two':
                for system_name_key in dict_in:
                
                    smiles_mol0 = dict_in[system_name_key][0]
                    smiles_mol1 = dict_in[system_name_key][1]
                
                    mol0 = Chem.MolFromSmiles(smiles_mol0)
                    mol1 = Chem.MolFromSmiles(smiles_mol1)
                    
                    reactants = (mol0, mol1)
                    
                    reaction_rdkit = smt2rxn(reaction_smarts)
                    products_react = reaction_rdkit.RunReactants(reactants) 
                    result_react = products_react[0][0]
                    smiles_result = mol2smi(result_react)
                    
                    dict_reaction[system_name_key] = smiles_result
                
                '''
                if self.dict_smiles == {}: # oder dict_smiles[react_type] = None ?
                    self.dict_smiles[react_type] = dict_reaction
                else:
                    for key in dict_reaction:
                        self.dict[react_type][key] = dict_reaction[key]
                '''
                #self.dict_smiles[react_type] = dict_reaction
                
                #self.dict_in_step_{}.format(i) = 
                return dict_reaction
                
            
            if mode == 'three':
                for system_name_key in dict_in:
                
                    smiles_mol0 = dict_in[system_name_key][0]
                    smiles_mol1 = dict_in[system_name_key][1]
                    smiles_mol2 = dict_in[system_name_key][2]
                
                    mol0 = Chem.MolFromSmiles(smiles_mol0)
                    mol1 = Chem.MolFromSmiles(smiles_mol1)
                    mol2 = Chem.MolFromSmiles(smiles_mol2)
                    
                    reactants = (mol0, mol1, mol2)
                    
                    reaction_rdkit = smt2rxn(reaction_smarts)
                    products_react = reaction_rdkit.RunReactants(reactants) 
                    result_react = products_react[0][0]
                    smiles_result = mol2smi(result_react)
                    
                    dict_reaction[system_name_key] = smiles_result
    
                return dict_reaction
    
            if mode == 'single':
                for system_name_key in dict_in:
                
                    smiles_mol0 = dict_in[system_name_key] #[0]
                
                    mol0 = Chem.MolFromSmiles(smiles_mol0)
                    
                    reactants = (mol0, )
                    
                    reaction_rdkit = smt2rxn(reaction_smarts)
                    products_react = reaction_rdkit.RunReactants(reactants) 
                    result_react = products_react[0][0]
                    smiles_result = mol2smi(result_react)
                    
                    dict_reaction[system_name_key] = smiles_result
    
                return dict_reaction
                
            
            #smiles_res = construct.rdkit_reaction(reaction_smarts, reactants)
            
            # make into/ add to dictionary
            # dictionary {type1: {name: smiles}, type2: {name:smiles, name:smiles...} ...}
            
            # add type and names to smiles_dict
        
            else:
                print('specify reaction mode')
    
    
    def save_smiles_dict(self, filepath):
        json.dump( self.dict_smiles, open( "{}.json".format(filepath), 'w' ) )
        
        print('saved smiles dictionary.')

    def load_smiles_dict(self, filepath):
        
        data = json.load( open( "{}.json".format(filepath) ) )
        
        self.dict_smiles = data
    
    
    def coord_generation(self, overwrite = False, optimize = True):
        # from smiles -> xyz coord 
        # input: dictionary {type1: {name: smiles}, type2: {name:smiles, name:smiles...} ...}
        # coords, elements = construct.get_coords_from_smiles(smiles, 1, 'rdkit')
        #
        # out: init coords für alle molecules in dictionary (check older code dipeptide generation)
        # out: write init coords in subfolder in outdir
        # call write names of molecules in file (maybe in construct fcts)
        
        # self.dict_smiles 
        # check dict
        
        #for key in a_dict:
            # print(key, '->', a_dict[key])
            
        # create folder 'type/name/...xyz'
            
        
        for key in self.dict_smiles:
            for molecule in self.dict_smiles[key]:
                
                mol_dir = '{}/{}/{}'.format(self.outdir, key, molecule)
                mol_out = '{}/{}_init_coords.xyz'.format(mol_dir, molecule)
                
                if os.path.exists(mol_out) and not overwrite:
                    print('init coords exist and overwrite = False')
                
                else:
                    os.makedirs(mol_dir)
                    coords, elements = construct.get_coords_from_smiles(self.dict_smiles[key][molecule], 1, 'rdkit')
                    if optimize:
                        # do xtb optimization
                        # add more charge states
                        
                        #pos_chr = self.dict_smiles[key][molecule].count('+')
                        #neg_chr = -1*(self.dict_smiles[key][molecule].count('-'))
                        #print(molecule)
                        #print(self.dict_smiles[key][molecule], pos_chr, neg_chr)
                        #print(molecule)
                        pos_chr = molecule.count('+')
                        neg_chr = -1*molecule.count('-')
                        
                        chrg = pos_chr+neg_chr
                        #print('chrg', chrg)
                        
                        coords_opt, elements_opt = xtb.optimize_geometry(coords, elements, mol_dir, charge = chrg, solvent = self.solvent)
                        fu.exportXYZ(coords_opt,elements_opt,mol_out)                        
                    else:
                        fu.exportXYZ(coords,elements,mol_out)
                
            print('generated initial coordinates for {}'.format(key))
                
        
    
    
    def conf_generation(self, settings_conf, indir_path = None):
        
        # infile_path not none if initial data already given and not generated beforehand
        # files must be called: *_init_coords.xyz
        if indir_path == None:
            # need path from smiles
            indir_path = self.outdir

        # need glob access to all files within dir
        #pathlist_coords_init = Path(indir_path).glob('*/**/*_init_coords.xyz')
        list_dir_files = glob.glob("{}/*/**/*_init_coords.xyz".format(indir_path))
        list_dir = glob.glob("{}/*/**/".format(indir_path))
        
        for idx, path_dir in enumerate(list_dir):
            
            file_name = list_dir_files[idx].split("/")[-1]
            
            mol_name = file_name.replace('_init_coords.xyz','')
            
            
            coords_in, elements_in = fu.readXYZ(list_dir_files[idx])
            
            # adapt it to handle more charges
            # if '+' in name: charge = +1, if '-' in name: charge = -1 if '+' not in... '-' not in ..charge = 0
            
            pos_chr = file_name.count('+')
            neg_chr = -1*(file_name.count('-'))

            charge = pos_chr+neg_chr
            
            # for each file/ mol generate conf according to settings
            coords_out, elements_out, energies_out = xtb.run_crest(coords_in, elements_in, path_dir, file_name, settings_conf['crest_fast'], charge, settings_conf['crest_method'], solvent = self.solvent, num_threads = self.num_threads)
            
            # save all energies of conf to npy
            # unit energies??
            # check dir ? self.solvent
            np.save('{}conformer_all_energies.npy'.format(path_dir), np.array(energies_out), allow_pickle=True)
            print('Generated {} conformers for {} using Crest.'.format(len(energies_out), mol_name))
            

            
            # then choose conformers according to settings['conf_types'] and number
            energy_out_conf = construct.choose_conformers(coords_out, elements_out, energies_out, settings_conf['number'],path_dir, conf_types = settings_conf['conf_types'])
            #print('Choose {} conformers for {}.'.format(settings_conf['number'], file_name))
            
            # plot energies conf in simple plot 
            # add: highlight chosen conformers in different color -> modify choose_conformer function
            self.plot_conformer_energy(mol_name, energies_out, energy_out_conf, path_dir)

    def check_dir(self, outdir):
        # check if directory exists, if not make
        # return 'outdir'
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        return outdir
    
    def check_dict(self, dict_smiles):
        # add if None
        if dict_smiles == {}:
            print('smiles dictionary empty, need to generate smiles code')
        
        return dict_smiles
        
    def plot_conformer_energy(self,mol_name, energies_conf, energy_out_conf, outdir):
        
        fig, ax = plt.subplots()
        common_label = 'Other Conformers'
        legend_entries = []
        red_points_label = 'Conformers Chosen'
        
        for x, y in enumerate(energies_conf):
            if y in energy_out_conf:
                scatter = ax.scatter(x, y, c='darkorange')
                if common_label not in legend_entries:
                    legend_entries.append(common_label)
            else:
                scatter = ax.scatter(x, y, c='royalblue')
                if common_label not in legend_entries:
                    legend_entries.append(common_label)
        
        # Add a single label for all red points
        if red_points_label not in legend_entries:
            legend_entries.append(red_points_label)
        
        ax.set_xlabel('Conformer i')
        ax.set_ylabel(r'$ E [eV]$')
        
        # Create the legend manually
        ax.legend(legend_entries)
        
        ax.set_title('CREST Conformer Search {}: Energy Distribution of {} Total Conformers'.format(mol_name , len(energies_conf)))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}E_conformer_distribution_{}.png'.format(outdir, mol_name), dpi = 300)
        #plt.show()
        
