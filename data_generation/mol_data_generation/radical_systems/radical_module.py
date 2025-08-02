#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import random
import os
import itertools
import numpy as np
import mol_data_generation.utils.file_utils as fu
import mol_data_generation.radical_systems.radical_functions as rf
import mol_data_generation.utils.xtb_fcts as xtb
import mol_data_generation.utils.scaler as scaler
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial.distance import euclidean

class RadicalSampling():
    """
    Class for generating and exporting radical systems by removing hydrogens to create radicals
    from molecules and sampling both intra- and inter-molecular radical structures.
    """

    def __init__(self,insystem_list, indir_list,indir_info,init_indir_list, solvent, t_nms, delE_nms, protonated=False, rad_dist = 2.0, infile_structure = None):
        
        """
        Initialize the RadicalSampling workflow and store parameters.

        Parameters
        ----------
        insystem_list : list
            List of systems (strings) used in sampling.
        indir_list : list
            List of directories containing input data for NMS samples.
        indir_info : str
            Directory containing molecular info and logs.
        init_indir_list : list
            List of directories with unperturbed initial structures.
        solvent : str
            Solvent model used in XTB.
        t_nms : int
            NMS sampling temperature (K).
        delE_nms : float
            Max allowed NMS sampling energy window (eV).
        protonated : bool
            Whether to sample protonated systems (affects H selection).
        rad_dist : float
            Minimum allowed distance between radical and H atoms in combined system.
        infile_structure : str
            Input directory structure ('cluster' or flat).
        """

        self.indir_list = indir_list
        self.indir_info = indir_info
        self.insystem_list = insystem_list
        self.rad_dist = rad_dist
        self.init_indir_list = init_indir_list
        self.infile_structure = infile_structure 
        #self.outdir = outdir
        
        self.solvent = solvent
        
        #self.nms_num = num_nms
        self.nms_delE = delE_nms
        self.nms_t = t_nms
        
        self.nms_sample_name = 'T{}_Emax{}'.format(self.nms_t, int(self.nms_delE))
        
        self.sample_parameters_inter = {}
        self.sample_parameters_intra = {}
        # sample name must include ph7 if protonated
        self.protonated = protonated
        
        #self.coords_infile_path_list = self.get_infile_path_list(self.indir_list)
        
        self.energies_all_list = None
        self.energies_all_list_intra = None
        
    def do_intra_rad_sampling(self,sample_name, num_systems, num_config,outdir, mode = 'H_radical', radical_opt = False):

       """
            Perform intra-molecular radical sampling: for each molecule,
            generate radicals by removing an H atom, possibly optimize, and export coords, energies, and forces.

            Parameters
            ----------
            sample_name : str
                Name for exported files.
            num_systems : int
                Number of systems to randomly select (0=all in list).
            num_config : int
                Number of radical configs per system.
            outdir : str
                Output directory.
            mode : str
                Radical mode ('H_radical' = remove hydrogen to create radical).
            radical_opt : bool
                If True, optimize radical geometry after H removal.
        """ 

       # Gather input XYZ file paths for intra sampling
       coords_dir_list_mols = self.get_infile_path_list_intra(self.indir_list, xyz_pattern="*nms_samples_*.xyz")
       

       coords_sampled_all, elements_sampled_all = [], []
       energies_sampled_all, forces_sampled_all = [], []
       elements_F_sampled_all = []
       h_rad_distances = []
       system_names = []
       #radical_names, donor_names = [],[]
       #num_atms_rad, num_atms_don = [],[]
       bonds_systems = []
       idx_radicals, idx_h0s = [],[]
       idx_r2 = []
       coords_h2 = []
       
       if num_systems == 0:
           #Loop through all systems in coords_dir_list_mols
           for coords_dir in coords_dir_list_mols:

               
               path_to_coords_1 = coords_dir

               coords_all_1, elements_all_1 = fu.readXYZs(path_to_coords_1)
               print(len(coords_all_1), 'coords in file', path_to_coords_1)

               
               if self.infile_structure == 'cluster':
                   path_to_log_1, mol_name_1 = self.get_mol_paths_cluster_intra(path_to_coords_1)
                   
               else:
                   path_to_log_1, mol_name_1 = self.get_mol_paths_intra(path_to_coords_1)
               
               
               pos_chr_0 = mol_name_1.count('+')
               neg_chr_0 = -1*mol_name_1.count('-')
               tot_chr = pos_chr_0+neg_chr_0

               
               # # Identify bonds and hydrogens
               # identify bonds and H
               if mode == 'H_radical':
                   
                   # get bond info here
                   bond_idx_list_1,bond_atm_list_1, bond_lengths_list_1 = rf.read_out_bonds(path_to_log_1)     
                   #print('bond idx list 1', bond_idx_list_1)    

                   
                   # all possible h tuples
                   atms_idx_h_tuples_1 = rf.get_aa_h(mol_name_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1, protonated = self.protonated) # protonated = 

                   # dictionary with correct length/ formula for dipeptides, if length not ok, then remove
                   
                   for config_i in range(num_config):
                                 
                      no_system = True
                       
                      while no_system:
                          # choose nms coords from coords of the chosen molecule
                          rand_int_1 = random.randrange(0,len(coords_all_1))
                          coords_1 = coords_all_1[rand_int_1]
                          elements_1 = elements_all_1[rand_int_1]
                           
                          #print('INIT LEN', len(coords_1), len(elements_1))
                          # choose H
                           
                          
                           
                          dist_H_idx, dist_er = rf.get_H_dist_intra(coords_1, atms_idx_h_tuples_1)
                          #print(dist_er)
                          if dist_er == True:
                              print('encountered dist error in intra rad')
                              
                              
                           
                          max_H_dist_intra = 2.65
                           
                          h1, h2, r1, r2, tuple_er = rf.get_H_tuple_intra(dist_H_idx, max_H_dist_intra)
                          
                          if tuple_er == True:
                              print('encountered tuple distance error')
                              
                              continue
                           
                          # Remove chosen H from molecule to create radical
                           
                          rad_idx_new, h1_idx_new, r2_idx_new, coords_init, elements_init, bond_idx_list_new, bond_atm_list_new, bond_lenghts_list_new, bond_idx_list_before, bond_atm_list_before, bond_lengths_list_before, coords_before_h1,  rad_idx_new_2, h1_idx_new_2, r2_idx_new_2, coords_before_1, elements_before_1, bond_idx_list_new_2, bond_atm_list_new_2, bond_lenghts_list_new_2, coords_before, elements_before, coords_before_h2 = rf.rmv_h_from_mol_intra(
                               h2, r1, h1, r2, coords_1, elements_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1)
        
        
                           # optimize radical 
                           
                           # state 1
                          if radical_opt:
                              print('optimize radical') # rm outdir?
                              num_atoms = len(coords_init)
                              freeze_list = rf.get_neighbors(rad_idx_new, bond_idx_list_new, bond_atm_list_new, num_atoms)
                              coords_final_1, elements_final_1 = xtb.optimize_geometry(coords_init, elements_init,charge = tot_chr, unp_e = 1, freeze_atms = freeze_list, solvent = self.solvent)
                              if coords_final_1 == [] or len(coords_final_1) < num_atoms:
                                  continue
                            #print(coords_2)
                          else:
                               coords_final_1 = coords_init
                               elements_final_1 = elements_init

    
                                   
                            
                               
                                   
                            # calculate E and F
     
                          e_system_1 = xtb.single_point_energy(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                          f_system_1 = xtb.single_force(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                           
                          
                          try:
                            if f_system_1.all() != None:                              
                                if len(f_system_1) == len(coords_final_1):
                                    forces = True
                          except:
                              try:
                                if f_system_1 == None:
                                    print('forces none')
                                    forces = False
                              except:
                                  pass

                                  
                          if e_system_1 != None and forces and dist_er == False:
                              no_system = False

                         
                      coords_sampled_all.append(coords_final_1)

                      #coords_sampled_all.append(coords_final_2)

                       
                      elements_sampled_all.append(elements_final_1)
                      #elements_sampled_all.append(elements_final_2)
                       
                      energies_sampled_all.append(e_system_1)
                      #energies_sampled_all.append(e_system_2)
                       
                      forces_sampled_all.append(f_system_1)
                      #forces_sampled_all.append(f_system_2)
                      
                      elements_F_sampled_all.append(elements_final_1)
                      #elements_F_sampled_all.append(elements_final_2)
                       
                      h_rad_dist = euclidean(coords_final_1[h1_idx_new], coords_final_1[rad_idx_new])
                      #print('h rad dist', h_rad_dist)
                       
                      h_rad_distances.append(h_rad_dist)
                       
                      system_names.append(mol_name_1)
                      #donor_names.append(mol_name_1)
                      #radical_names.append(mol_name_2)
                      #num_atms_don.append(len(coords_1_new))
                      #num_atms_rad.append(len(coords_2_new))
                      bonds_systems.append(bond_idx_list_new)
                      idx_radicals.append(rad_idx_new)
                      idx_h0s.append(h1_idx_new)                   
                      idx_r2.append(r2_idx_new)
                      coords_h2.append(coords_before_h1)
                       

                       
                       
                        
               else:
                   print('Need to implement reaction mode in radical functions.')
           
           
           
           
       else: 
           #  Randomly sample num_systems from input list
           for system_i in range(num_systems):
               
               # choose randomly 1 molecule from file list
               # draw random infile, load nms coords
               mols_paths_chosen = np.random.choice(coords_dir_list_mols, 1, replace=True)
               #
               #print('chosen 1', mols_paths_chosen[0], 'chosen 2', mols_paths_chosen[1])
               # designate [0] as mol1 with H1 and [1] radical
               
               ## need to adapt to new structure! 
               
               path_to_coords_1 = mols_paths_chosen[0]
               #glob.glob('{}*_nms_samples_{}.xyz'.format(mols_chosen[0],nms_name))  ##adjust
               #print(path_to_coords_1)
               
               #print(path_to_coords_1)
               coords_all_1, elements_all_1 = fu.readXYZs(path_to_coords_1)
               
               
               ## need to change to get mol names from chosen and vib info from indir
               #print('path_to_coords_1', path_to_coords_1)
    
               
               if self.infile_structure == 'cluster':
                   path_to_log_1, mol_name_1 = self.get_mol_paths_cluster_intra(mols_paths_chosen[0])
                   
               else:
                   path_to_log_1, mol_name_1 = self.get_mol_paths_intra(mols_paths_chosen[0])
               
               
               
               pos_chr_0 = mol_name_1.count('+')
               neg_chr_0 = -1*mol_name_1.count('-')
               tot_chr = pos_chr_0+neg_chr_0
    
               
    
               # identify bonds and H
               if mode == 'H_radical':
                   
                   # get bond info here
                   bond_idx_list_1,bond_atm_list_1, bond_lengths_list_1 = rf.read_out_bonds(path_to_log_1)     
                   #print('bond idx list 1', bond_idx_list_1)    
    
                   
                   # all possible h tuples
                   atms_idx_h_tuples_1 = rf.get_aa_h(mol_name_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1, protonated = self.protonated) # protonated = 
    
                   # dictionary with correct length/ formula for dipeptides, if length not ok, then remove
                   # oder durch nms gehen und consistent length checken!!
                   
                   for config_i in range(num_config):
                                 
                      no_system = True
                       
                      while no_system:
                          # choose nms coords from coords of the chosen molecule
                          rand_int_1 = random.randrange(0,len(coords_all_1))
                          coords_1 = coords_all_1[rand_int_1]
                          elements_1 = elements_all_1[rand_int_1]
                           
                          #print('INIT LEN', len(coords_1), len(elements_1))
                          # choose H
                           
                          ### need to change this
                           
                          dist_H_idx, dist_er = rf.get_H_dist_intra(coords_1, atms_idx_h_tuples_1)
                          #print(dist_er)
                          if dist_er == True:
                              print('encountered dist error in intra rad')
                              
                              
                           
                          max_H_dist_intra = 2.65
                           
                          h1, h2, r1, r2, tuple_er = rf.get_H_tuple_intra(dist_H_idx, max_H_dist_intra)
                          
        
                           
                           
                          #remove atom from molecule 2 and update 
                           
        #                 rad_idx,h1_idx_new,r2_idx_new, coords_init, elements_init, bond_idx_list, bond_atm_list, bond_lengths_list, coords_before, elements_before, bond_idx_list_before, bond_atm_list_before, bond_lengths_list_before, coords_before_h2 = rf.rmv_h_from_mol_intra(
        #                 h2, r1,h1,r2, coords_1, elements_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1)
                           
                          rad_idx_new, h1_idx_new, r2_idx_new, coords_init, elements_init, bond_idx_list_new, bond_atm_list_new, bond_lenghts_list_new, bond_idx_list_before, bond_atm_list_before, bond_lengths_list_before, coords_before_h1,  rad_idx_new_2, h1_idx_new_2, r2_idx_new_2, coords_before_1, elements_before_1, bond_idx_list_new_2, bond_atm_list_new_2, bond_lenghts_list_new_2, coords_before, elements_before, coords_before_h2 = rf.rmv_h_from_mol_intra(
                               h2, r1, h1, r2, coords_1, elements_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1)
        
        
                           # optimize radical 
                           
                           # state 1
                          if radical_opt:
                              print('optimize radical') # rm outdir?
                              num_atoms = len(coords_init)
                              freeze_list = rf.get_neighbors(rad_idx_new, bond_idx_list_new, bond_atm_list_new, num_atoms)
                              coords_final_1, elements_final_1 = xtb.optimize_geometry(coords_init, elements_init,charge = tot_chr, unp_e = 1, freeze_atms = freeze_list, solvent = self.solvent)
                              if coords_final_1 == [] or len(coords_final_1) < num_atoms:
                                  continue
                            #print(coords_2)
                          else:
                               coords_final_1 = coords_init
                               elements_final_1 = elements_init
                               #print(len(coords_final_1))
                               #print(len(elements_final_1))
  
                                   
                               
                                   
                            # calculate E and F
                            
                            # state 1
                          e_system_1 = xtb.single_point_energy(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                          f_system_1 = xtb.single_force(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                           

                          try:
                              if f_system_1.all() != None:                              
                                  if len(f_system_1) == len(coords_final_1):
                                      forces = True
                          except:
                              if f_system_1 == None:
                                  print('forces none')
                                  forces = False
                        
                          if e_system_1 != None and forces and dist_er == False:
                              #print('no system False', e_system_1, e_system_2, len(f_system_1), len(f_system_2))
                              no_system = False
                                
                         
                      coords_sampled_all.append(coords_final_1)
    
                      #coords_sampled_all.append(coords_final_2)
    
                       
                      elements_sampled_all.append(elements_final_1)
                      #elements_sampled_all.append(elements_final_2)
                       
                      energies_sampled_all.append(e_system_1)
                      #energies_sampled_all.append(e_system_2)
                       
                      forces_sampled_all.append(f_system_1)
                      #forces_sampled_all.append(f_system_2)
                      
                      elements_F_sampled_all.append(elements_final_1)
                      #elements_F_sampled_all.append(elements_final_2)
                       
                      h_rad_dist = euclidean(coords_final_1[h1_idx_new], coords_final_1[rad_idx_new])
                      #print('h rad dist', h_rad_dist)
                       
                      h_rad_distances.append(h_rad_dist)
                       
                      system_names.append(mol_name_1)
                      #donor_names.append(mol_name_1)
                      #radical_names.append(mol_name_2)
                      #num_atms_don.append(len(coords_1_new))
                      #num_atms_rad.append(len(coords_2_new))
                      bonds_systems.append(bond_idx_list_new)
                      idx_radicals.append(rad_idx_new)
                      idx_h0s.append(h1_idx_new)                   
                      idx_r2.append(r2_idx_new)
                      coords_h2.append(coords_before_h1)
                       
                      #print(mol_name_1)
                        
                      #print('system 1', len(coords_final_1), len(elements_final_1), len(f_system_1), type(coords_final_1), type(f_system_1))      
                       
                      #print('system 2', len(coords_final_2), len(elements_final_2), len(f_system_2),  type(coords_final_2), type(f_system_2))
        
                      # also save bond, rad idx, H1 idx information! delta E! add E information!
    
                       
                       
                        
               else:
                   print('Need to implement reaction mode in radical functions.')
           
            
           

       
       # Store sampled data in class and export to files
       self.sample_parameters_intra['system_names'] = system_names
       self.sample_parameters_intra['bonds_systems'] = bonds_systems
       self.sample_parameters_intra['idx_radicals'] = idx_radicals
       self.sample_parameters_intra['idx_h0s'] = idx_h0s
       self.sample_parameters_intra['h_rad_distances'] = h_rad_distances
       
       self.sample_parameters_intra['idx_rad2'] = idx_r2
       self.sample_parameters_intra['coords_h2_before'] = coords_h2
       
       self.energies_all_list_intra = energies_sampled_all
       ## export
       
       if not os.path.exists(outdir):
           os.makedirs(outdir)

       
       
       print('len coords final', len(coords_sampled_all))
       print('len forces final', len(forces_sampled_all))
       print('len elements final', len(elements_sampled_all))
       fu.exportXYZs(coords_sampled_all, elements_sampled_all, '{}/{}_rad_intra_systems_coords.xyz'.format(outdir,sample_name) )
       #fu.exportXYZs(forces_sampled_all, elements_sampled_all, '{}/{}_rad_intra_systems_forces.xyz'.format(outdir, sample_name) ) # add info, change fct
       print('exported all coords')
       fu.export_csv_rad_systems_intra(outdir,sample_name, system_names, h_rad_distances, bonds_systems, idx_radicals, idx_h0s, idx_r2, coords_h2)
       print('exported csv')
       fu.exportXYZs(forces_sampled_all, elements_F_sampled_all, '{}/{}_rad_intra_systems_forces.xyz'.format(outdir, sample_name) ) # add info, change fct

       
       np.save('{}/{}_rad_intra_systems_energies.npy'.format(outdir, sample_name), energies_sampled_all, allow_pickle=True) 
       np.save('{}/{}_rad_intra_systems_radii.npy'.format(outdir, sample_name,), h_rad_distances, allow_pickle=True) 
    


    def do_inter_rad_sampling_V2(self,sample_name, num_systems, num_config, outdir, rad_radius = [1.0, 4.2], mode = 'H_radical', radical_opt = False, system_list = []):
       

       """
        Perform inter-molecular radical sampling: For each system,
        generate two-molecule radical complexes by removing H from each and
        combining as an intermolecular system, with geometric and distance constraints.

        Parameters
        ----------
        sample_name : str
            Name for exported files.
        num_systems : int
            Number of unique pairs to sample (0 = all unique pairs).
        num_config : int
            Number of sampled configs per system pair.
        outdir : str
            Output directory for exports.
        mode : str
            Radical mode (default: 'H_radical').
        rad_radius : list or str
            Sampling radius range for radical–radical separation ([min,max] or 'chi2' for chi2 distribution).
        reaction : bool
            Whether to perform reaction mode (not yet implemented).
        """
       
    
       print("indir_list", self.indir_list)

       if system_list == []:
           coords_dir_list_mols = self.get_infile_path_list_inter(self.indir_list)
           print("NMS sample files found:", coords_dir_list_mols)
           
       else:
           coords_dir_list_mols_1 = self.get_infile_path_list_inter(system_list[0], partner=1)
           coords_dir_list_mols_2 = self.get_infile_path_list_inter(system_list[1], partner=2)

           print("coords_dir_list_1:", coords_dir_list_mols_1)
           print("coords_dir_list_2:", coords_dir_list_mols_2)
           

       coords_sampled_all, elements_sampled_all = [], []
       energies_sampled_all, forces_sampled_all = [], []
       h_rad_distances = []
       system_names = []

       bonds_systems = []
       idx_radicals, idx_h0s = [],[]
       idx_r2 = []
       coords_h2 = []
       
       radical_names, donor_names = [],[]
       num_atms_rad, num_atms_don = [],[]
       
       if num_systems == 0:
           #print(type(coords_dir_list_mols), len(coords_dir_list_mols))
           
           if system_list ==[]:
               all_combinations = self.all_list_combinations(coords_dir_list_mols)
               print('generated {} combinations'.format(len(all_combinations)))
               
           else:
               all_combinations = self.all_combinations_two_lists([coords_dir_list_mols_1, coords_dir_list_mols_2])
               print('generated {} combinations'.format(len(all_combinations)))
            
           for combination_i in all_combinations:
                
               
               path_to_coords_1 = combination_i[0]
               #glob.glob('{}*_nms_samples_{}.xyz'.format(mols_chosen[0],nms_name))  ##adjust
               #print(path_to_coords_1)
               
               #print(path_to_coords_1)
               coords_all_1, elements_all_1 = fu.readXYZs(path_to_coords_1)
               
               path_to_coords_2 = combination_i[1]
               #print(path_to_coords_1)
               #print(path_to_coords_2)
               coords_all_2, elements_all_2 = fu.readXYZs(path_to_coords_2)
               
               ## need to change to get mol names from chosen and vib info from indir
               #print('path_to_coords_1', path_to_coords_1)
               #print('path_to_coords_2', path_to_coords_2)
               
               if self.infile_structure == 'cluster':
                   path_to_log_1, mol_name_1, path_to_log_2, mol_name_2 = self.get_mol_paths_cluster(combination_i[0], combination_i[1])
                   #print(path_to_log_1, mol_name_1, path_to_log_2, mol_name_2)
               else:
                   path_to_log_1, mol_name_1, path_to_log_2, mol_name_2 = self.get_mol_paths(combination_i[0], combination_i[1])
               
               
               pos_chr_0 = mol_name_1.count('+')
               neg_chr_0 = -1*mol_name_1.count('-')
               tot_chr_0 = pos_chr_0+neg_chr_0
               
               pos_chr_1 = mol_name_2.count('+')
               neg_chr_1 = -1*mol_name_2.count('-')
               tot_chr_1 = pos_chr_1+neg_chr_1
               
               tot_chr = tot_chr_0+tot_chr_1
               
               if mode == 'H_radical':
                   
                   # get bond info here
                   bond_idx_list_1,bond_atm_list_1, bond_lengths_list_1 = rf.read_out_bonds(path_to_log_1)     
                   #print('bond idx list 1', bond_idx_list_1)    
                   bond_idx_list_2,bond_atm_list_2, bond_lengths_list_2 = rf.read_out_bonds(path_to_log_2)  
                   # differentiate cases amide, caps, aa
                   
                   mol_name_1 = os.path.splitext(os.path.basename(path_to_coords_1))[0]
                   mol_name_2 = os.path.splitext(os.path.basename(path_to_coords_2))[0]
                   
                   # all possible h tuples
                   atms_idx_h_tuples_1 = rf.get_aa_h(mol_name_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1, protonated = self.protonated) # protonated = 
                   atms_idx_h_tuples_2 = rf.get_aa_h(mol_name_2, bond_idx_list_2, bond_atm_list_2, bond_lengths_list_2, protonated = self.protonated)
                   
                   
                   
                   for config_i in range(num_config):
                       
                       no_system = True
                       
                       while no_system:
                           # choose nms coords from coords of the two chosen molecules
                           rand_int_1 = random.randrange(0,len(coords_all_1))
                           coords_1 = coords_all_1[rand_int_1]
                           elements_1 = elements_all_1[rand_int_1]
                           
                           rand_int_2 = random.randrange(0,len(coords_all_2))
                           coords_2 = coords_all_2[rand_int_2]
                           elements_2 = elements_all_2[rand_int_2]
                           
                           
                           
                           # choose H
                           
                           
                           h_idx_1, h_bond_1 = rf.get_h_idx(atms_idx_h_tuples_1)
        
                           h_idx_rm, rad_pos_idx = rf.get_h_idx(atms_idx_h_tuples_2)
                           
                           
                           
                           #remove atom from molecule 2 and update 

                           
                           # translate H1 and H2 to (0,0)
                           coords_1_new = rf.translate_to_center(coords_1, h_idx_1)
                           coords_2_new = rf.translate_to_center(coords_2, h_idx_rm)               
                           
        
                           
        
                           # find system H1 - H2
                           
                           coords_system,elements_system, bonds_system, radius,rad_idx_new , h2_idx_new,  e_system, f_system,found_system = rf.find_system_inter_V2(coords_1_new, coords_2_new, h_idx_1, h_bond_1, rad_pos_idx, h_idx_rm, elements_1, elements_2, bond_idx_list_1, bond_idx_list_2, rad_radius,tot_chr, self.solvent, self.rad_dist)
                           
                           if found_system == True:
                               #print('found sys')
                               # remove Hs to get both states
                               rad_idx_new,h1_idx_new, r2_idx_new, coords_final_1, elements_final_1, bond_idx_list_new,  bond_idx_list_before,  coords_before_h2,  rad_idx_new_2,h1_idx_new_2, r2_idx_new_2, coords_final_2, elements_final_2, bond_idx_list_new_2, coords_before, elements_before, coords_before_h1 = rf.rmv_h_from_mol_inter_V2(h2_idx_new, rad_idx_new, h_idx_1,h_bond_1, coords_system,elements_system, bonds_system)
                               
                               # state 1
                               
                               # e, f calc
                               
                               e_system_1 = xtb.single_point_energy(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                               f_system_1 = xtb.single_force(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                               
                               try:
                                   if f_system_1.all() != None:                              
                                       if len(f_system_1) == len(coords_final_1):
                                           forces = True
                               except:
                                   if f_system_1 == None:
                                       print('forces none')
                                       forces = False
                             
                               if e_system_1 != None and forces:
                                   #print('no system False', e_system_1,  len(f_system_1))
                                   no_system = False
                           
                       

                           

                       # save 
                       
                       coords_sampled_all.append(coords_final_1)
                       elements_sampled_all.append(elements_final_1)

                       h_rad_dist = euclidean(coords_final_1[h1_idx_new], coords_final_1[rad_idx_new])
                       h_rad_distances.append(h_rad_dist)
                       
                       system_name = '{}_and_{}'.format(mol_name_1, mol_name_2)
                       
                       system_names.append(system_name)
                       bonds_systems.append(bond_idx_list_new)
                       idx_radicals.append(rad_idx_new)
                       idx_h0s.append(h1_idx_new)                   
                       idx_r2.append(r2_idx_new)
                       coords_h2.append(coords_before_h2)
                       
                       energies_sampled_all.append(e_system_1)
                       forces_sampled_all.append(f_system_1)
                       
                       donor_names.append(mol_name_1)
                       radical_names.append(mol_name_2)
                       num_atms_don.append(len(coords_1_new))
                       num_atms_rad.append(len(coords_2_new))
                       
                    

               else:
                   print('Need to implement reaction mode in radical functions.')
           
           
       
       else:
       
           for system_i in range(num_systems):
               
               
               if system_list == []:
                   # choose randomly 2 molecules from file list
                   # draw random infile, load nms coords
                   mols_paths_chosen = np.random.choice(coords_dir_list_mols, 2, replace=True)
                   
                   path_to_coords_1 = mols_paths_chosen[0]
                   #glob.glob('{}*_nms_samples_{}.xyz'.format(mols_chosen[0],nms_name))  ##adjust
                   #print(path_to_coords_1)
                   
                   #print(path_to_coords_1)
                   coords_all_1, elements_all_1 = fu.readXYZs(path_to_coords_1)
                   
                   path_to_coords_2 = mols_paths_chosen[1]
                   
                   coords_all_2, elements_all_2 = fu.readXYZs(path_to_coords_2)
                   
               else:
                   mols_paths_chosen_1 = np.random.choice(coords_dir_list_mols_1, 1, replace=True)
                   mols_paths_chosen_2 = np.random.choice(coords_dir_list_mols_2, 1, replace=True)
                    
                   path_to_coords_1 = mols_paths_chosen_1[0]
                   coords_all_1, elements_all_1 = fu.readXYZs(path_to_coords_1)
                   
                   path_to_coords_2 = mols_paths_chosen_2[0]
                   coords_all_2, elements_all_2 = fu.readXYZs(path_to_coords_2)
                   
               
             
               
               if self.infile_structure == 'cluster':
                   path_to_log_1, mol_name_1, path_to_log_2, mol_name_2 = self.get_mol_paths_cluster(path_to_coords_1, path_to_coords_2)
               else:
                   path_to_log_1, mol_name_1, path_to_log_2, mol_name_2 = self.get_mol_paths(path_to_coords_1, path_to_coords_2)
               
               
               
               ## need to add more charge combinations!!
               
               pos_chr_0 = mol_name_1.count('+')
               neg_chr_0 = -1*mol_name_1.count('-')
               tot_chr_0 = pos_chr_0+neg_chr_0
               
               pos_chr_1 = mol_name_2.count('+')
               neg_chr_1 = -1*mol_name_2.count('-')
               tot_chr_1 = pos_chr_1+neg_chr_1
               
               tot_chr = tot_chr_0+tot_chr_1

               if mode == 'H_radical':
                   
                   # get bond info here
                   bond_idx_list_1,bond_atm_list_1, bond_lengths_list_1 = rf.read_out_bonds(path_to_log_1)     
                   #print('bond idx list 1', bond_idx_list_1)    
                   bond_idx_list_2,bond_atm_list_2, bond_lengths_list_2 = rf.read_out_bonds(path_to_log_2)  
                   # differentiate cases amide, caps, aa
                   
                   mol_name_1 = os.path.splitext(os.path.basename(path_to_coords_1))[0]
                   mol_name_2 = os.path.splitext(os.path.basename(path_to_coords_2))[0]
                   
                   
                   # all possible h tuples
                   atms_idx_h_tuples_1 = rf.get_aa_h(mol_name_1, bond_idx_list_1, bond_atm_list_1, bond_lengths_list_1, protonated = self.protonated) # protonated = 
                   atms_idx_h_tuples_2 = rf.get_aa_h(mol_name_2, bond_idx_list_2, bond_atm_list_2, bond_lengths_list_2, protonated = self.protonated)
                   
                   
                   
                   for config_i in range(num_config):
                       
                       no_system = True
                       
                       while no_system:
                           # choose nms coords from coords of the two chosen molecules
                           rand_int_1 = random.randrange(0,len(coords_all_1))
                           coords_1 = coords_all_1[rand_int_1]
                           elements_1 = elements_all_1[rand_int_1]
                           
                           rand_int_2 = random.randrange(0,len(coords_all_2))
                           coords_2 = coords_all_2[rand_int_2]
                           elements_2 = elements_all_2[rand_int_2]
                           
                           
                           
                           # choose H
                           
                           
                           h_idx_1, h_bond_1 = rf.get_h_idx(atms_idx_h_tuples_1)
        
                           h_idx_rm, rad_pos_idx = rf.get_h_idx(atms_idx_h_tuples_2)
                           
                           
                           
                           # translate H1 and H2 to (0,0)
                           coords_1_new = rf.translate_to_center(coords_1, h_idx_1)
                           coords_2_new = rf.translate_to_center(coords_2, h_idx_rm)               
                           
        
                           
        
                           # find system H1 - H2
                           
                           coords_system,elements_system, bonds_system, radius,rad_idx_new , h2_idx_new,  e_system, f_system,found_system = rf.find_system_inter_V2(coords_1_new, coords_2_new, h_idx_1, h_bond_1, rad_pos_idx, h_idx_rm, elements_1, elements_2, bond_idx_list_1, bond_idx_list_2, rad_radius,tot_chr, self.solvent, self.rad_dist)
                           
                           
                           if found_system == True:
                               #print('found sys')
                               # remove Hs to get both states
                               rad_idx_new,h1_idx_new, r2_idx_new, coords_final_1, elements_final_1, bond_idx_list_new,  bond_idx_list_before,  coords_before_h2,  rad_idx_new_2,h1_idx_new_2, r2_idx_new_2, coords_final_2, elements_final_2, bond_idx_list_new_2, coords_before, elements_before, coords_before_h1 = rf.rmv_h_from_mol_inter_V2(h2_idx_new, rad_idx_new, h_idx_1,h_bond_1, coords_system,elements_system, bonds_system)
                               
                               # state 1
                               
                               # e, f calc
                               
                               e_system_1 = xtb.single_point_energy(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                               f_system_1 = xtb.single_force(coords_final_1, elements_final_1, charge = tot_chr, unp_e = 1, solvent = self.solvent)
                               
                               try:
                                   if f_system_1.all() != None:                              
                                       if len(f_system_1) == len(coords_final_1):
                                           forces = True
                               except:
                                   if f_system_1 == None:
                                       print('forces none')
                                       forces = False
                             
                               if e_system_1 != None and forces:
                                   #print('no system False', e_system_1, len(f_system_1))
                                   no_system = False

                       
                       coords_sampled_all.append(coords_final_1)
                       elements_sampled_all.append(elements_final_1)
    
                       h_rad_dist = euclidean(coords_final_1[h1_idx_new], coords_final_1[rad_idx_new])
                       h_rad_distances.append(h_rad_dist)
                       
                       system_name = '{}_and_{}'.format(mol_name_1, mol_name_2)
                       
                       system_names.append(system_name)
                       bonds_systems.append(bond_idx_list_new)
                       idx_radicals.append(rad_idx_new)
                       idx_h0s.append(h1_idx_new)                   
                       idx_r2.append(r2_idx_new)
                       coords_h2.append(coords_before_h2)
                       
                       energies_sampled_all.append(e_system_1)
                       forces_sampled_all.append(f_system_1)
                       
                       donor_names.append(mol_name_1)
                       radical_names.append(mol_name_2)
                       num_atms_don.append(len(coords_1_new))
                       num_atms_rad.append(len(coords_2_new))
                       
                       
               else:
                   print('Need to implement reaction mode in radical functions.')
           
            

       
       # add to class sample parameter dictionary (?)
       self.sample_parameters_inter['system_names'] = system_names
       self.sample_parameters_inter['donor_names'] = donor_names
       self.sample_parameters_inter['radical_names'] = radical_names
       self.sample_parameters_inter['num_atms_don'] = num_atms_don
       self.sample_parameters_inter['num_atms_rad'] = num_atms_rad
       self.sample_parameters_inter['bonds_systems'] = bonds_systems
       self.sample_parameters_inter['idx_radicals'] = idx_radicals
       self.sample_parameters_inter['idx_h0s'] = idx_h0s
       self.sample_parameters_inter['h_rad_distances'] = h_rad_distances
       
       self.sample_parameters_inter['idx_rad2'] = idx_r2
       
       self.sample_parameters_inter['coords_h2_before'] = coords_h2
       
       self.energies_all_list = energies_sampled_all
       ## export
       
       if not os.path.exists(outdir):
           os.makedirs(outdir)
       
       fu.exportXYZs(coords_sampled_all, elements_sampled_all, '{}/{}_rad_inter_systems_coords.xyz'.format(outdir,sample_name) )
       fu.exportXYZs(forces_sampled_all, elements_sampled_all, '{}/{}_rad_inter_systems_forces.xyz'.format(outdir, sample_name) ) # add info, change fct
       fu.export_csv_rad_systems_inter_V2(outdir,sample_name, system_names, h_rad_distances, bonds_systems, idx_radicals, idx_h0s, idx_r2, coords_h2, donor_names, radical_names,num_atms_don, num_atms_rad)
           
       
       np.save('{}/{}_rad_inter_systems_energies.npy'.format(outdir, sample_name), energies_sampled_all, allow_pickle=True) 
       np.save('{}/{}_rad_inter_systems_radii.npy'.format(outdir, sample_name,), h_rad_distances, allow_pickle=True)          



    def plot_inter_rad_distances(self, outdir, sample_name):
        
        
        distances = self.sample_parameters_inter['h_rad_distances']
        
        #distances_flat = [element for innerList in distances for element in innerList]
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(distances, 15, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'Distance $[\AA]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title(r'Inter H Sampling: Radical - H0 Distances, num_samples={}'.format(len(distances))) ## change
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/{}_radical_h0_distances_inter_initial_hist_{}.png'.format(outdir,sample_name, len(distances)), dpi = 300)
        #plt.show()

    def plot_intra_rad_distances(self, outdir, sample_name):
        
        
        distances = self.sample_parameters_intra['h_rad_distances']
        
        #distances_flat = [element for innerList in distances for element in innerList]
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(distances, 15, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'Distance $[\AA]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title(r'Intra H Sampling: Radical - H0 Distances, num_samples={}'.format(len(distances))) ## change
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/{}_radical_h0_distances_intra_initial_hist_{}.png'.format(outdir,sample_name, len(distances)), dpi = 300)
        #plt.show()
    
    def plot_rad_system_energies(self, outdir,sample_name, system_type = 'inter'):
        
            
        # energies

        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(self.energies_all_list, 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$ E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title('Initial {} Radical Systems: E distribution num_samples={}'.format(system_type, len(self.energies_all_list)))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/{}_energies_initial_{}_radical_systems_hist.png'.format(outdir,sample_name, system_type), dpi = 300)
        #plt.show()  
        
    def plot_rad_system_energies_intra(self, outdir,sample_name, system_type = 'intra'):
        
            
        # energies

        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(self.energies_all_list_intra, 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$ E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title('Initial {} Radical Systems: E distribution num_samples={}'.format(system_type, len(self.energies_all_list_intra)))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/{}_energies_initial_{}_radical_systems_hist.png'.format(outdir,sample_name, system_type), dpi = 300)
        #plt.show()       
        

        
           
    def check_protonated(self):
        
        if 'ph7' in self.sample_name:
            return True
        
        else:
            return False
        
    def get_infile_path_list_inter(self, indir_list, partner=1):
        """
        Recursively collect all NMS .xyz files in the provided indir_list.
        Ignores partner argument (used for interface compatibility).

        Parameters
        ----------
        indir_list : list of str
            List of top-level directories to search for NMS sample files.
        partner : int
            Unused; for compatibility.

        Returns
        -------
        xyz_files : list of str
            List of full paths to NMS .xyz files found in all dirs.
        """
        xyz_files = []
        for indir in indir_list:
            for dirpath, dirnames, filenames in os.walk(indir):
                for filename in filenames:
                    # If you want only NMS samples, check for a unique pattern:
                    # e.g., 'nms_samples_' in filename
                    if filename.endswith('.xyz') and 'nms_samples_' in filename:
                        xyz_files.append(os.path.join(dirpath, filename))
        return xyz_files
 
    def get_infile_path_list(self, indir_list):
        
        coords_infile_path_list = []
        
        nms_indir_list = []
        
        for indir in indir_list:
            #print(indir)    
            list_dirs_mols = glob.glob("{}/*_T{}_Emax{}/".format(indir,self.nms_t, int(self.nms_delE)))
            
            for mol_dir in list_dirs_mols:
                
                nms_indir_list.append(mol_dir)
                
        for dir_nms in nms_indir_list:
            #print('nms dir', dir_nms)
            path_to_coords = glob.glob('{}*nms_samples_*_T{}_Emax{}.xyz'.format(dir_nms, self.nms_t, int(self.nms_delE)))
                
            for path in path_to_coords:
                coords_infile_path_list.append(path)
                
        return coords_infile_path_list

    def get_infile_path_list_intra(self, indir_list, xyz_pattern="*.xyz", exclude_names=None):
        
        coords_infile_path_list = []
        for indir in indir_list:
            # Recursively find all xyz files matching pattern
            found = glob.glob(os.path.join(indir, "**", xyz_pattern), recursive=True)
            for path in found:
                # Optionally exclude files containing certain names
                if exclude_names and any(ex in path for ex in exclude_names):
                    continue
                coords_infile_path_list.append(path)
        return coords_infile_path_list



    def get_mol_paths(self, path_to_coords_1, path_to_coords_2):
        """
        Robustly find the vib_analysis_conf_0 folders and molecule names for two .xyz files.
        """
        # For molecule 1
        parent_dir_1 = os.path.dirname(path_to_coords_1)
        mol_name_1 = os.path.splitext(os.path.basename(path_to_coords_1))[0]
        path_to_log_1 = os.path.join(parent_dir_1, "vib_analysis_conf_0")
        if not os.path.isdir(path_to_log_1):
            raise FileNotFoundError(f"No vib_analysis_conf_0 for: {path_to_coords_1}")
        
        # For molecule 2
        parent_dir_2 = os.path.dirname(path_to_coords_2)
        mol_name_2 = os.path.splitext(os.path.basename(path_to_coords_2))[0]
        path_to_log_2 = os.path.join(parent_dir_2, "vib_analysis_conf_0")
        if not os.path.isdir(path_to_log_2):
            raise FileNotFoundError(f"No vib_analysis_conf_0 for: {path_to_coords_2}")

        return path_to_log_1, mol_name_1, path_to_log_2, mol_name_2


    
    def get_mol_paths_cluster(self, coord_path_mol_1, coord_path_mol_2): # 
        
        # get mol name
        coord_file_1 = coord_path_mol_1.split('/')[-1]
        coord_file_2 = coord_path_mol_2.split('/')[-1]
        
        pattern_to_rm = '_nms_samples_num*_T{}_Emax{}.xyz'.format(self.nms_t, int(self.nms_delE))
        
        mol_name_1 = rf.remove_variable_substring(coord_file_1, pattern_to_rm)
        
        mol_name_2 = rf.remove_variable_substring(coord_file_2, pattern_to_rm)
        #print('mol_name_1', mol_name_1)
        #print('mol_name_2', mol_name_2)
        
        system_file_1 = coord_path_mol_1.split('/')[1]        
        system_name_1 = system_file_1.replace('nms_', '')
        #print('system_name_1', system_name_1)
        system_file_2 = coord_path_mol_2.split('/')[1]        
        system_name_2 = system_file_2.replace('nms_', '')
        
        #print('system_file_1', system_file_1)
        #print('system_name_2', system_name_2)
        #print('system_file_2', system_file_2)
        list_all_paths_init = []
        for dir_i in self.init_indir_list:
            name_dirs = dir_i
                #print('name_dirs', name_dirs)
            paths_to_dirs = glob.glob('{}/{}/*/*/'.format(self.indir_info,name_dirs))
            
            for i in range(len(paths_to_dirs)):
                list_all_paths_init.append(paths_to_dirs[i])
        
        #print('list_all_paths_1', len(list_all_paths_init), list_all_paths_init[0] )
        
        # get to path
        
        for path_i in list_all_paths_init:
            #print('path_i', path_i)
            if mol_name_1 in path_i:
                path_to_log_1 = '{}/vib_analysis_conf_0'.format(path_i)
                
            if mol_name_2 in path_i:
                path_to_log_2 = '{}/vib_analysis_conf_0'.format(path_i)
            
            
        
        #path_to_log_1 = glob.glob('{}/{}/*/{}/vib_analysis_conf_0'.format(self.indir_info, system_name_1, mol_name_1))[0]
        
        #path_to_log_2 = glob.glob('{}/{}/*/{}/vib_analysis_conf_0'.format(self.indir_info, system_name_2, mol_name_2))[0]
        
        return path_to_log_1, mol_name_1, path_to_log_2, mol_name_2

    def get_mol_paths_cluster_intra(self, coord_path_mol_1): # 
        
        # get mol name
        coord_file_1 = coord_path_mol_1.split('/')[-1]
        
        
        pattern_to_rm = '_nms_samples_num*_T{}_Emax{}.xyz'.format(self.nms_t, int(self.nms_delE))
        
        mol_name_1 = rf.remove_variable_substring(coord_file_1, pattern_to_rm)
        
        
        #print('mol_name_1', mol_name_1)
        
        
        system_file_1 = coord_path_mol_1.split('/')[1]        
        system_name_1 = system_file_1.replace('nms_', '')
        #print('system_name_1', system_name_1)

        
        #print('system_file_1', system_file_1)

        list_all_paths_init = []
        for dir_i in self.init_indir_list:
            name_dirs = dir_i
            #if 'dip' in dir_i:
            #    name_dirs = '{}*'.format(dir_i)
            #    #print('name_dirs', name_dirs)
            #else:
            #    name_dirs = dir_i
                #print('name_dirs', name_dirs)
            paths_to_dirs = glob.glob('{}/{}/*/*/'.format(self.indir_info,name_dirs))
            
            for i in range(len(paths_to_dirs)):
                list_all_paths_init.append(paths_to_dirs[i])
        
        #print('list_all_paths_1', len(list_all_paths_init), list_all_paths_init[0] )
        
        # get to path
        
        for path_i in list_all_paths_init:
            #print('path_i', path_i) 
            if mol_name_1 in path_i:
                path_to_log_1 = '{}/vib_analysis_conf_0'.format(path_i)
                

            
            
        
        #path_to_log_1 = glob.glob('{}/{}/*/{}/vib_analysis_conf_0'.format(self.indir_info, system_name_1, mol_name_1))[0]
        
        #path_to_log_2 = glob.glob('{}/{}/*/{}/vib_analysis_conf_0'.format(self.indir_info, system_name_2, mol_name_2))[0]
        
        return path_to_log_1, mol_name_1

    def get_mol_paths_intra(self, coord_path_mol_1):
        xyz_file = os.path.basename(coord_path_mol_1)            # Alanine_nms_samples_num20_T50_Emax5.xyz
        parent_dir = os.path.dirname(coord_path_mol_1)            # samples/aa_test_nms/aa/Alanine
        mol_name = os.path.splitext(xyz_file)[0]                  # Alanine_nms_samples_num20_T50_Emax5

        # Look for vib_analysis_conf_0 inside the parent dir
        vib_dir = os.path.join(parent_dir, 'vib_analysis_conf_0')
        if not os.path.exists(vib_dir):
            raise FileNotFoundError(f"vib_analysis_conf_0 not found in {parent_dir}")
        
        
        return vib_dir, mol_name


    def all_list_combinations(self, input_list):
        return [list(combination) for combination in itertools.combinations_with_replacement(input_list, 2)]

    
    def all_combinations_two_lists(self, input_list):
        return [list(p) for p in itertools.product(*input_list)]
        


    def scale_and_plot_energies_radicals(self, outdir, sample_name, system_type='inter'):
        
        if system_type == 'inter':
            energies_all = np.load('{}/{}_rad_inter_systems_energies.npy'.format(outdir, sample_name), allow_pickle=True)
            forces_all, elements_all = fu.readXYZs('{}/{}_rad_inter_systems_forces.xyz'.format(outdir, sample_name))
        if system_type == 'intra':
            energies_all = np.load('{}/{}_rad_intra_systems_energies.npy'.format(outdir, sample_name), allow_pickle=True)
            forces_all, elements_all = fu.readXYZs('{}/{}_rad_intra_systems_forces.xyz'.format(outdir, sample_name))
            
        scaler_en = scaler.ExtensiveEnergyForceScaler()
        
        y = [energies_all, forces_all]
        
        atomic_numbers = scaler_en.convert_elements_to_atn(elements_all)
        #print(len(atomic_numbers))
        scaler_en.fit(atomic_numbers, y)

        #print(scaler.get_weights())
        #print(scaler.get_config())

        scaler_en._plot_predict(atomic_numbers, y, outdir, sample_name)

        x_res, out_e, grads_out_all = scaler_en.transform(atomic_numbers, y)
        
        #self.scaled_energies_all_list.append(out_e[0])
        
        # plot hist scaled energy
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(out_e[0], 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$ E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title('Scaled E distribution num_samples={}'.format(len(energies_all)))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/Scaled_E_rad_systems_hist_{}.png'.format(outdir, sample_name), dpi = 300)
        #plt.show()  

        num_atoms = []
        for i in range(len(elements_all)):
            num_atoms.append(len(elements_all[i]))

        
        fig, ax = plt.subplots()
        hist = ax.hist2d(num_atoms, out_e[0], bins = 20, cmap = 'plasma',norm =  colors.LogNorm()) #norm=colors.LogNorm() cm2 = 
        fig.colorbar(hist[3], ax=ax) #fig.colorbar() #cm2[3], ax=ax2
        ax.set_xlabel('Number of atoms/system')
        ax.set_ylabel(r'$ E [eV]$')
        ax.set_title('Number of atoms per molecule - Scaled Energies')
        fig.tight_layout()
        plt.savefig('{}/rad_systems_{}_2d_E_{}.png'.format( outdir, sample_name, len(out_e[0])), dpi = 300)


        # E-Hdistances 2d
        if system_type == 'inter':
            distances = self.sample_parameters_inter['h_rad_distances']
        
        if system_type == 'intra':
            distances = self.sample_parameters_intra['h_rad_distances']
        
        fig, ax = plt.subplots()
        hist = ax.hist2d(distances, out_e[0], bins = 20, cmap = 'plasma',norm =  colors.LogNorm()) #norm=colors.LogNorm() cm2 = 
        fig.colorbar(hist[3], ax=ax) #cm2[3], ax=ax2
        ax.set_xlabel(r'Distance $[\AA]$')
        ax.set_ylabel(r'$ E [eV]$')
        ax.set_title('H-Radical Distances - Scaled Energies')
        fig.tight_layout()
        plt.savefig('{}/rad_systems_{}_2d_E_dist_{}.png'.format( outdir, sample_name, len(out_e[0])), dpi = 300)


'''
        num_atoms_E = []
        for mol in param_dict['num_atoms']:
            num_atoms_E.append([param_dict['num_atoms'][mol]]* param_dict['num_samples_per_mol'][mol])

        num_atoms_E_flat = [element for innerList in num_atoms_E for element in innerList]      

    def scale_and_plot_energies(self, outdir = None, indir_forces = None, indir_energies = None):
        if outdir == None:
            outdir = self.indir
        
        if indir_forces == None:
            path_forces = '{}/nms_forces_all_{}.xyz'.format(self.indir, self.sample_name)
        else:
            path_forces = '{}/nms_forces_all_{}.xyz'.format(indir_forces, self.sample_name)
        
        if indir_energies == None:
            path_energies = '{}/nms_energies_all_{}.npy'.format(self.indir, self.sample_name)
        else:
            path_energies = '{}/nms_energies_all_{}.npy'.format(indir_energies, self.sample_name)
        
        #print('path force',path_forces)
        #print('path energy', path_energies)
        
        energies_all = np.load(path_energies, allow_pickle=True)
        #print('len en', len(energies_all))
        #print(energies_all.shape)
        forces_all, elements_all = fu.readXYZs(path_forces)
        #print('len force', len(forces_all))
        
        # scale energy
        scaler_en = scaler.ExtensiveEnergyForceScaler()
        
        y = [energies_all, forces_all]
        
        atomic_numbers = scaler_en.convert_elements_to_atn(elements_all)
        #print(len(atomic_numbers))
        scaler_en.fit(atomic_numbers, y)

        #print(scaler.get_weights())
        #print(scaler.get_config())

        scaler_en._plot_predict(atomic_numbers, y, outdir, self.sample_name)

        x_res, out_e, grads_out_all = scaler_en.transform(atomic_numbers, y)
        
        self.scaled_energies_all_list.append(out_e[0])
        
        # plot hist scaled energy
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(out_e[0], 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$ E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title('NMS: Scaled E distribution num_samples={}, $\Delta Emax$ = {} [eV]'.format(len(self.energies_all_list), self.delta_E_max))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/Scaled_E_nms_hist_{}.png'.format(outdir, self.sample_name), dpi = 300)
        #plt.show()  



        cm2 = ax2.hist2d(num_atoms_E_flat, scaled_en_all, bins = 20, cmap = 'plasma',norm =  colors.LogNorm()) #norm=colors.LogNorm()
        fig.colorbar(cm2[3], ax=ax2)
        ax2.set_xlabel('Number of atoms/molecule')
        ax2.set_ylabel(r'$ E [eV]$')
        ax2.set_title('Scaled Energies')


        self.indir_list = indir_list
        self.indir_info = indir_info
        self.insystem_list = insystem_list

path_to_log_1 = '{}vib_analysis_conf_0'.format(mols_chosen[0])
path_to_log_2 = '{}vib_analysis_conf_0'.format(mols_chosen[1])

mol_name_1 = mols_chosen[0].split('/')[-2]

mol_name_2 = mols_chosen[1].split('/')[-2]


Alanine_nms_samples_num100_T100_Emax5
for idx, path_dir in enumerate(list_dirs):
indir = 'samples/cluster_latest_samples'
       # prepare dicts_i
       #dict_ki = {}
           # 
           
           #list_dir_files_energies =  glob.glob("{}/*/**/energy_conformers_selected.npy".format(path_to_files))
           list_dirs = glob.glob("{}/*/**/".format(path_to_files))
       
sample_name = '_num100_T100_Emax5'

num_systems = 10
num_configurations_systems = 10 # one system several times or more systems once?

# function to choose system pairs

list_mol_dirs = glob.glob("{}/*/*/*/".format(indir))
num_mol = 1
mols_choice = np.random.choice(list_mol_dirs, num_mol, replace=True)

# designate one as mol1 with H1 and one radical

# remove atoms, translate, optimize etc

# then search system

# write loop to do this for num_system times

# think about where to save/ how and what to save for systems

'''