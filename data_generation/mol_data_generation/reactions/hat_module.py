#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import os
import pandas as pd


import mol_data_generation.radical_systems.radical_functions as rf
import mol_data_generation.utils.file_utils as fu
import mol_data_generation.utils.xtb_fcts as xtb
import mol_data_generation.reactions.hat_functions as hat
import mol_data_generation.utils.scaler as scaler
from matplotlib import colors
import matplotlib.pyplot as plt

class HATSampling():
    
    def __init__(self, indir, outdir, sample_name, solvent, hat_dist = 0.8):
        
        self.indir = indir
        self.outdir = outdir
        self.sample_name = sample_name
        self.solvent = solvent
        self.hat_dist = hat_dist
        self.sample_parameters_hat = {}
        
        
        self.coords_hat_all = []
        self.elements_hat_all = []
        self.energies_hat_all = []
        self.forces_hat_all = []
        
        # intra 
        self.sample_parameters_hat_intra = {}
        self.coords_hat_all_intra = []
        self.elements_hat_all_intra = []
        self.energies_hat_all_intra = []
        self.forces_hat_all_intra = []
        
    def do_hat_sampling(self, csv_name, xyz_name,xyz_force_name,energy_npy_name, num_init_hat = 1,x = 5, y = 4, z=5, r_cutoff = 3.5, opt_mode = 'rand_on_sphere', check_hat_bonds = False):
        
        ## load initial radical systems ##
        
        # indir = samples/radical_system_tests 
        
        # naming convention
        # csv_file_name = 'csv_init_radical_systems_info.csv'
        # rad_inter_systems_coords.xyz rad_inter_systems_energies.npy rad_inter_systems_forces.xyz 
        
        df = pd.read_csv('{}/{}'.format(self.indir, csv_name))
        # csv_init_radical_systems_info.csv
        id_init = df['ID'].tolist()
        system_names = df['system_name'].tolist()
        donor_names = df['donor_name'].tolist()
        radical_names = df['radical_name'].tolist()
        h_rad_dist_init = df['h_rad_dist_init'].tolist()
        num_atms_donor = df['num_atms_donor'].tolist()
        num_atms_radical = df['num_atms_radical'].tolist()
        idx_h0s = df['idx_h0'].tolist()
        idx_rads = df['idx_rad'].tolist()
        bond_orders = df['bond_order'].tolist()
        idx_rads_2 = df['idx_rad2'].tolist()
        
        coords_all, elements_all = fu.readXYZs('{}/{}'.format(self.indir, xyz_name)) #rad_inter_systems_coords.xyz
        #print('len coords init', len(coords_all))
        forces_all, elements_f_all = fu.readXYZs('{}/{}'.format(self.indir, xyz_force_name))
        energies_init = np.load('{}/{}'.format(self.indir, energy_npy_name), allow_pickle = True)
        ## move H1 randomly in between molecules

        
        
        id_init_all = []
        system_names_all = []
        donor_names_all = []
        radical_names_all = []
        h_rad_dist_init_all = []
        num_atms_donor_all = []
        num_atms_radical_all = []
        idx_h0s_all = []
        idx_rads_all = []
        idx_rads_2_all = []
        bond_orders_all = []
        h0_r1_dist_all, h0_r2_dist_all, r1_r2_dist_all = [],[],[]
        
        
        
        for idx, system in enumerate(system_names):
            
            self.coords_hat_all.append(coords_all[idx])
            self.elements_hat_all.append(elements_all[idx])
            
            
            #print('initial coords append', coords_all[idx])
            self.energies_hat_all.append(energies_init[idx])
            
            #forces_hat_all.append(f_system)
            self.forces_hat_all.append(forces_all[idx])
            

            
            center0, r_max = hat.calculate_center_distance(coords_all[idx][idx_h0s[idx]], coords_all[idx][idx_rads[idx]])
            
            h0_r1_dist_init, h0_r2_dist_init, r1_r2_dist_init = hat.get_h0_r1_r2_distances(coords_all[idx], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])

            # initial config

            
            # add h0-r0 distance and h0-r1 distance
            
            id_init_all.append(id_init[idx])
            system_names_all.append(system_names[idx])
            donor_names_all.append(donor_names[idx])
            radical_names_all.append(radical_names[idx])
            h_rad_dist_init_all.append(h_rad_dist_init[idx])
            num_atms_donor_all.append(num_atms_donor[idx])
            num_atms_radical_all.append(num_atms_radical[idx])
            idx_h0s_all.append(idx_h0s[idx])
            idx_rads_all.append(idx_rads[idx])
            bond_orders_all.append(bond_orders[idx])
            idx_rads_2_all.append(idx_rads_2[idx])
            
            h0_r1_dist_all.append(h0_r1_dist_init)
            h0_r2_dist_all.append(h0_r2_dist_init)
            r1_r2_dist_all.append(r1_r2_dist_init)            

            pos_chr = system.count('+')
            neg_chr = -1*system.count('-')
            
            tot_chr = pos_chr+neg_chr
            
            for n_init in range(num_init_hat):
                #try:
                coords_system_shifted, energy_system, f_system = hat.get_new_h1_position(r_max, center0, coords_all[idx], elements_all[idx], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx],self.hat_dist, chrg = tot_chr, solve = self.solvent, check_bonds = check_hat_bonds)
                
                #print(coords_system_shifted)
                #coords_shifted_init.append(coords_system_shifted[:])
                
                #elements_shifted_init.append(elements_all[idx])
                
                # could add different mode here
                
                if opt_mode == 'rand_on_sphere':
                    #print('Sampling mode', opt_mode)
                    

                    
                    # shifted H
                    
                    self.coords_hat_all.append(coords_system_shifted)
                    #print('shifted coords append',coords_system_shifted)
                    self.elements_hat_all.append(elements_all[idx])
                    
                    h0_r1_dist, h0_r2_dist, r1_r2_dist = hat.get_h0_r1_r2_distances(coords_system_shifted, idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                    
                    #energy_system = xtb.single_point_energy(coords_system_shifted, elements_all[idx], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                    
                    self.energies_hat_all.append(energy_system)
                    
                    
                    #f_system = xtb.single_force(coords_system_shifted, elements_all[idx], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                    #forces_hat_all.append(f_system)
                    self.forces_hat_all.append(f_system)
                    
    
                    # add h0-r0 distance and h0-r1 distance
                    
                    id_init_all.append(id_init[idx])
                    system_names_all.append(system_names[idx])
                    donor_names_all.append(donor_names[idx])
                    radical_names_all.append(radical_names[idx])
                    h_rad_dist_init_all.append(h_rad_dist_init[idx])
                    num_atms_donor_all.append(num_atms_donor[idx])
                    num_atms_radical_all.append(num_atms_radical[idx])
                    idx_h0s_all.append(idx_h0s[idx])
                    idx_rads_all.append(idx_rads[idx])
                    bond_orders_all.append(bond_orders[idx])
                    idx_rads_2_all.append(idx_rads_2[idx])
                    
                    h0_r1_dist_all.append(h0_r1_dist)
                    h0_r2_dist_all.append(h0_r2_dist)
                    r1_r2_dist_all.append(r1_r2_dist)
                
                if opt_mode == 'optimize_freeze':
                    print('Sampling mode', opt_mode)
                    freeze_list = hat.get_atms_outside_sphere(coords_system_shifted, idx_h0s[idx], r_cutoff)
    
                    mol_dir = '{}/xtb_opt_hat'.format(self.indir) # überschreiben lassen?
    
                    coords_opt, elements_opt = xtb.optimize_geometry(coords_system_shifted, elements_all[idx], mol_dir, charge= tot_chr, unp_e = 1, freeze_atms = freeze_list, solvent = self.solvent)
            
                    coords_opt_all,elements_opt_all, energies_opt_all = fu.read_xtboptlog(mol_dir)
                    
                    # select
                    
                    coords_selected, elements_selected, energies_selected = hat.select_steps(coords_opt_all,elements_opt_all, energies_opt_all, x, y, z)
                    
                    # append single coords selected 
                    for i in range(len(coords_selected)):
                        #coords_hat_all.append(coords_selected[i])
                        #elements_hat_all.append(elements_selected[i])
                        #energies_hat_all.append(energies_selected[i])
                        self.coords_hat_all.append(coords_selected[i])
                        self.elements_hat_all.append(elements_selected[i])
                        
                        self.energies_hat_all.append(energies_selected[i])
                        
                        f_system = xtb.single_force(coords_selected[i], elements_selected[i], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                        #forces_hat_all.append(f_system)
                        self.forces_hat_all.append(f_system)
                        
                        h0_r1_dist, h0_r2_dist,r1_r2_dist = hat.get_h0_r1_r2_distances(coords_selected[i], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                        
                        
                        
                        id_init_all.append(id_init[idx])
                        system_names_all.append(system_names[idx])
                        donor_names_all.append(donor_names[idx])
                        radical_names_all.append(radical_names[idx])
                        h_rad_dist_init_all.append(h_rad_dist_init[idx])
                        num_atms_donor_all.append(num_atms_donor[idx])
                        num_atms_radical_all.append(num_atms_radical[idx])
                        idx_h0s_all.append(idx_h0s[idx])
                        idx_rads_all.append(idx_rads[idx])
                        bond_orders_all.append(bond_orders[idx])
                        idx_rads_2_all.append(idx_rads_2[idx])
                        
                        h0_r1_dist_all.append(h0_r1_dist)
                        h0_r2_dist_all.append(h0_r2_dist)
                        r1_r2_dist_all.append(r1_r2_dist)
    
             
                
                   
        self.sample_parameters_hat['id_init'] = id_init_all
        self.sample_parameters_hat['system_names'] = system_names_all
        self.sample_parameters_hat['donor_names'] = donor_names_all
        self.sample_parameters_hat['radical_names'] = radical_names_all
        self.sample_parameters_hat['num_atms_don'] = num_atms_donor_all
        self.sample_parameters_hat['num_atms_rad'] = num_atms_radical_all
        self.sample_parameters_hat['bonds_order'] = bond_orders_all
        self.sample_parameters_hat['idx_radicals'] = idx_rads_all
        self.sample_parameters_hat['idx_h0s'] = idx_h0s_all
        self.sample_parameters_hat['idx_rad2'] = idx_rads_2_all
        self.sample_parameters_hat['h_rad_distances_init'] = h_rad_dist_init_all
        
        self.sample_parameters_hat['h0_r1_dist'] = h0_r1_dist_all
        self.sample_parameters_hat['h0_r2_dist'] = h0_r2_dist_all
        self.sample_parameters_hat['r1_r2_dist'] = r1_r2_dist_all          
               
                    
                #except:
                    #print('encountered exception (xtb opt)')
                    #continue
        print('Finished HAT sampling')
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        # export init and all selected 
        
        #fu.exportXYZs(coords_hat_all, elements_hat_all, '{}/hat_coords_sampled.xyz'.format(self.outdir))
        fu.exportXYZs(self.coords_hat_all, self.elements_hat_all, '{}/{}_hat_coords_sampled.xyz'.format(self.outdir, self.sample_name))
        
        #np.save('{}/hat_energies_sampled.npy'.format(self.outdir), energies_hat_all, allow_pickle = True)
        np.save('{}/{}_hat_energies_sampled.npy'.format(self.outdir,self.sample_name), self.energies_hat_all, allow_pickle = True)
        
        #fu.exportXYZs(forces_hat_all, elements_hat_all, '{}/hat_forces_sampled.xyz'.format(self.outdir))
        fu.exportXYZs(self.forces_hat_all, self.elements_hat_all, '{}/{}_hat_forces_sampled.xyz'.format(self.outdir, self.sample_name))
        
        # export csv
        print('export csv hat')
        # need to adapt for intra HAT!
        #fu.export_csv_hat(self.outdir, id_init_all, system_names_all, donor_names_all, radical_names_all, h_rad_dist_init_all, num_atms_donor_all, num_atms_radical_all, bond_orders_all, idx_rads_all, idx_h0s_all)
        fu.export_csv_hat(self.outdir, self.sample_name, self.sample_parameters_hat['id_init'], self.sample_parameters_hat['system_names'], self.sample_parameters_hat['donor_names'], self.sample_parameters_hat['radical_names'], self.sample_parameters_hat['h_rad_distances_init'], self.sample_parameters_hat['num_atms_don'], self.sample_parameters_hat['num_atms_rad'], self.sample_parameters_hat['bonds_order'], self.sample_parameters_hat['idx_radicals'], self.sample_parameters_hat['idx_h0s'], self.sample_parameters_hat['idx_rad2'],self.sample_parameters_hat['h0_r1_dist'],self.sample_parameters_hat['h0_r2_dist'], self.sample_parameters_hat['r1_r2_dist'])



    def do_hat_sampling_inter_V2(self, csv_name, xyz_name,xyz_force_name,energy_npy_name, num_init_hat = 1,x = 5, y = 4, z=5, r_cutoff = 3.5, opt_mode = 'rand_on_sphere', check_hat_bonds = False):
        
        ## load initial radical systems ##
        
        # indir = samples/radical_system_tests 
        
        # naming convention
        # csv_file_name = 'csv_init_radical_systems_info.csv'
        # rad_inter_systems_coords.xyz rad_inter_systems_energies.npy rad_inter_systems_forces.xyz 
        
        df = pd.read_csv('{}/{}'.format(self.indir, csv_name))
        # csv_init_radical_systems_info.csv
        id_init = df['ID'].tolist()
        system_names = df['system_name'].tolist()
        donor_names = df['donor_name'].tolist()
        radical_names = df['radical_name'].tolist()
        h_rad_dist_init = df['h_rad_dist_init'].tolist()
        num_atms_donor = df['num_atms_donor'].tolist()
        num_atms_radical = df['num_atms_radical'].tolist()
        idx_h0s = df['idx_h0'].tolist()
        idx_rads = df['idx_rad'].tolist()
        bond_orders = df['bond_order'].tolist()
        idx_rads_2 = df['idx_rad2'].tolist()
        
        
        coords = df["coords_h2"].values
        coords_h2_ar = [np.array(np.matrix(c.replace('\n', ';'))) for c in coords]
        coords_h2 = []
        for coord in coords_h2_ar:
            coords_h2.append(list(coord[0]))
        
        
        coords_all, elements_all = fu.readXYZs('{}/{}'.format(self.indir, xyz_name)) #rad_inter_systems_coords.xyz
        #print('len coords init', len(coords_all))
        forces_all, elements_f_all = fu.readXYZs('{}/{}'.format(self.indir, xyz_force_name))
        energies_init = np.load('{}/{}'.format(self.indir, energy_npy_name), allow_pickle = True)
        ## move H1 randomly in between molecules
        

        
        id_init_all = []
        system_names_all = []
        donor_names_all = []
        radical_names_all = []
        h_rad_dist_init_all = []
        num_atms_donor_all = []
        num_atms_radical_all = []
        idx_h0s_all = []
        idx_rads_all = []
        idx_rads_2_all = []
        bond_orders_all = []
        h0_r1_dist_all, h0_r2_dist_all, r1_r2_dist_all = [],[],[]
        
        
        
        for idx, system in enumerate(system_names):
            
            self.coords_hat_all.append(coords_all[idx])
            self.elements_hat_all.append(elements_all[idx])
            
            
            #print('initial coords append', coords_all[idx])
            self.energies_hat_all.append(energies_init[idx])
            
            #forces_hat_all.append(f_system)
            self.forces_hat_all.append(forces_all[idx])
            
            coords_h2_init = coords_h2[idx]
            
            
            center0, r_max = hat.calculate_center_distance(coords_all[idx][idx_h0s[idx]], coords_h2_init)
            
            #center0, r_max = hat.calculate_center_distance(coords_all[idx][idx_h0s[idx]], coords_all[idx][idx_rads[idx]])
            
            h0_r1_dist_init, h0_r2_dist_init, r1_r2_dist_init = hat.get_h0_r1_r2_distances(coords_all[idx], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])

            # initial config

            
            # add h0-r0 distance and h0-r1 distance
            
            id_init_all.append(id_init[idx])
            system_names_all.append(system_names[idx])
            donor_names_all.append(donor_names[idx])
            radical_names_all.append(radical_names[idx])
            h_rad_dist_init_all.append(h_rad_dist_init[idx])
            num_atms_donor_all.append(num_atms_donor[idx])
            num_atms_radical_all.append(num_atms_radical[idx])
            idx_h0s_all.append(idx_h0s[idx])
            idx_rads_all.append(idx_rads[idx])
            bond_orders_all.append(bond_orders[idx])
            idx_rads_2_all.append(idx_rads_2[idx])
            
            h0_r1_dist_all.append(h0_r1_dist_init)
            h0_r2_dist_all.append(h0_r2_dist_init)
            r1_r2_dist_all.append(r1_r2_dist_init)            

            pos_chr = system.count('+')
            neg_chr = -1*system.count('-')
            
            tot_chr = pos_chr+neg_chr
            
            for n_init in range(num_init_hat):
                #try:
                coords_system_shifted, energy_system, f_system = hat.get_new_h1_position(r_max, center0, coords_all[idx], elements_all[idx], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx],self.hat_dist, chrg = tot_chr, solve = self.solvent, check_bonds = check_hat_bonds)
                
                #print(coords_system_shifted)
                #coords_shifted_init.append(coords_system_shifted[:])
                
                #elements_shifted_init.append(elements_all[idx])
                
                # could add different mode here
                
                if opt_mode == 'rand_on_sphere':
                    #print('Sampling mode', opt_mode)
                    

                    
                    # shifted H
                    
                    self.coords_hat_all.append(coords_system_shifted)
                    #print('shifted coords append',coords_system_shifted)
                    self.elements_hat_all.append(elements_all[idx])
                    
                    h0_r1_dist, h0_r2_dist, r1_r2_dist = hat.get_h0_r1_r2_distances(coords_system_shifted, idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                    
                    #energy_system = xtb.single_point_energy(coords_system_shifted, elements_all[idx], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                    
                    self.energies_hat_all.append(energy_system)
                    
                    
                    #f_system = xtb.single_force(coords_system_shifted, elements_all[idx], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                    #forces_hat_all.append(f_system)
                    self.forces_hat_all.append(f_system)
                    
    
                    # add h0-r0 distance and h0-r1 distance
                    
                    id_init_all.append(id_init[idx])
                    system_names_all.append(system_names[idx])
                    donor_names_all.append(donor_names[idx])
                    radical_names_all.append(radical_names[idx])
                    h_rad_dist_init_all.append(h_rad_dist_init[idx])
                    num_atms_donor_all.append(num_atms_donor[idx])
                    num_atms_radical_all.append(num_atms_radical[idx])
                    idx_h0s_all.append(idx_h0s[idx])
                    idx_rads_all.append(idx_rads[idx])
                    bond_orders_all.append(bond_orders[idx])
                    idx_rads_2_all.append(idx_rads_2[idx])
                    
                    h0_r1_dist_all.append(h0_r1_dist)
                    h0_r2_dist_all.append(h0_r2_dist)
                    r1_r2_dist_all.append(r1_r2_dist)
                    

                
                if opt_mode == 'optimize_freeze':
                    print('Sampling mode', opt_mode)
                    freeze_list = hat.get_atms_outside_sphere(coords_system_shifted, idx_h0s[idx], r_cutoff)
    
                    mol_dir = '{}/xtb_opt_hat'.format(self.indir) # überschreiben lassen?
    
                    coords_opt, elements_opt = xtb.optimize_geometry(coords_system_shifted, elements_all[idx], mol_dir, charge= tot_chr, unp_e = 1, freeze_atms = freeze_list, solvent = self.solvent)
            
                    coords_opt_all,elements_opt_all, energies_opt_all = fu.read_xtboptlog(mol_dir)
                    
                    # select
                    
                    coords_selected, elements_selected, energies_selected = hat.select_steps(coords_opt_all,elements_opt_all, energies_opt_all, x, y, z)
                    
                    # append single coords selected 
                    for i in range(len(coords_selected)):
                        #coords_hat_all.append(coords_selected[i])
                        #elements_hat_all.append(elements_selected[i])
                        #energies_hat_all.append(energies_selected[i])
                        self.coords_hat_all.append(coords_selected[i])
                        self.elements_hat_all.append(elements_selected[i])
                        
                        self.energies_hat_all.append(energies_selected[i])
                        
                        f_system = xtb.single_force(coords_selected[i], elements_selected[i], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                        #forces_hat_all.append(f_system)
                        self.forces_hat_all.append(f_system)
                        
                        h0_r1_dist, h0_r2_dist,r1_r2_dist = hat.get_h0_r1_r2_distances(coords_selected[i], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                        
                        
                        
                        id_init_all.append(id_init[idx])
                        system_names_all.append(system_names[idx])
                        donor_names_all.append(donor_names[idx])
                        radical_names_all.append(radical_names[idx])
                        h_rad_dist_init_all.append(h_rad_dist_init[idx])
                        num_atms_donor_all.append(num_atms_donor[idx])
                        num_atms_radical_all.append(num_atms_radical[idx])
                        idx_h0s_all.append(idx_h0s[idx])
                        idx_rads_all.append(idx_rads[idx])
                        bond_orders_all.append(bond_orders[idx])
                        idx_rads_2_all.append(idx_rads_2[idx])
                        
                        h0_r1_dist_all.append(h0_r1_dist)
                        h0_r2_dist_all.append(h0_r2_dist)
                        r1_r2_dist_all.append(r1_r2_dist)
    
            # add final state to samples
            
            coords_final, elements_final, energy_final, force_final, final_state = hat.get_final_state(idx_h0s[idx], coords_all[idx], elements_all[idx],coords_h2_init, tot_chr, self.solvent)
            
            if final_state == True:
                self.coords_hat_all.append(coords_final)
                self.elements_hat_all.append(elements_final)
                self.energies_hat_all.append(energy_final)
                self.forces_hat_all.append(force_final)
                
                id_init_all.append(id_init[idx])
                system_names_all.append(system_names[idx])
                donor_names_all.append(donor_names[idx])
                radical_names_all.append(radical_names[idx])
                h_rad_dist_init_all.append(h_rad_dist_init[idx])
                num_atms_donor_all.append(num_atms_donor[idx])
                num_atms_radical_all.append(num_atms_radical[idx])
                idx_h0s_all.append(idx_h0s[idx])
                idx_rads_all.append(idx_rads[idx])
                bond_orders_all.append(bond_orders[idx])
                idx_rads_2_all.append(idx_rads_2[idx])
                
                h0_r1_dist_final, h0_r2_dist_final, r1_r2_dist_final = hat.get_h0_r1_r2_distances(coords_final, idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                h0_r1_dist_all.append(h0_r1_dist_final)
                h0_r2_dist_all.append(h0_r2_dist_final)
                r1_r2_dist_all.append(r1_r2_dist_final)  
                
                   
        self.sample_parameters_hat['id_init'] = id_init_all
        self.sample_parameters_hat['system_names'] = system_names_all
        self.sample_parameters_hat['donor_names'] = donor_names_all
        self.sample_parameters_hat['radical_names'] = radical_names_all
        self.sample_parameters_hat['num_atms_don'] = num_atms_donor_all
        self.sample_parameters_hat['num_atms_rad'] = num_atms_radical_all
        self.sample_parameters_hat['bonds_order'] = bond_orders_all
        self.sample_parameters_hat['idx_radicals'] = idx_rads_all
        self.sample_parameters_hat['idx_h0s'] = idx_h0s_all
        self.sample_parameters_hat['idx_rad2'] = idx_rads_2_all
        self.sample_parameters_hat['h_rad_distances_init'] = h_rad_dist_init_all
        
        self.sample_parameters_hat['h0_r1_dist'] = h0_r1_dist_all
        self.sample_parameters_hat['h0_r2_dist'] = h0_r2_dist_all
        self.sample_parameters_hat['r1_r2_dist'] = r1_r2_dist_all          
               
                    
                #except:
                    #print('encountered exception (xtb opt)')
                    #continue
        print('Finished HAT sampling')
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        # export init and all selected 
        
        #fu.exportXYZs(coords_hat_all, elements_hat_all, '{}/hat_coords_sampled.xyz'.format(self.outdir))
        fu.exportXYZs(self.coords_hat_all, self.elements_hat_all, '{}/{}_hat_coords_sampled.xyz'.format(self.outdir, self.sample_name))
        
        #np.save('{}/hat_energies_sampled.npy'.format(self.outdir), energies_hat_all, allow_pickle = True)
        np.save('{}/{}_hat_energies_sampled.npy'.format(self.outdir,self.sample_name), self.energies_hat_all, allow_pickle = True)
        
        #fu.exportXYZs(forces_hat_all, elements_hat_all, '{}/hat_forces_sampled.xyz'.format(self.outdir))
        fu.exportXYZs(self.forces_hat_all, self.elements_hat_all, '{}/{}_hat_forces_sampled.xyz'.format(self.outdir, self.sample_name))
        
        # export csv
        print('export csv hat')
        # need to adapt for intra HAT!
        #fu.export_csv_hat(self.outdir, id_init_all, system_names_all, donor_names_all, radical_names_all, h_rad_dist_init_all, num_atms_donor_all, num_atms_radical_all, bond_orders_all, idx_rads_all, idx_h0s_all)
        fu.export_csv_hat(self.outdir, self.sample_name, self.sample_parameters_hat['id_init'], self.sample_parameters_hat['system_names'], self.sample_parameters_hat['donor_names'], self.sample_parameters_hat['radical_names'], self.sample_parameters_hat['h_rad_distances_init'], self.sample_parameters_hat['num_atms_don'], self.sample_parameters_hat['num_atms_rad'], self.sample_parameters_hat['bonds_order'], self.sample_parameters_hat['idx_radicals'], self.sample_parameters_hat['idx_h0s'], self.sample_parameters_hat['idx_rad2'],self.sample_parameters_hat['h0_r1_dist'],self.sample_parameters_hat['h0_r2_dist'], self.sample_parameters_hat['r1_r2_dist'])





    def do_hat_sampling_intra(self, csv_name, xyz_name,xyz_force_name,energy_npy_name, num_init_hat = 1,x = 5, y = 4, z=5, r_cutoff = 3.5, opt_mode = 'rand_on_sphere', check_hat_bonds = False):
        
        ## load initial radical systems ##
        
        # indir = samples/radical_system_tests 
        
        # naming convention
        # csv_file_name = 'csv_init_radical_systems_info.csv'
        # rad_inter_systems_coords.xyz rad_inter_systems_energies.npy rad_inter_systems_forces.xyz 
        
        df = pd.read_csv('{}/{}'.format(self.indir, csv_name))
        # csv_init_radical_systems_info.csv
        id_init = df['ID'].tolist()
        system_names = df['system_name'].tolist()
        h_rad_dist_init = df['h_rad_dist_init'].tolist()
        idx_h0s = df['idx_h0'].tolist()
        idx_rads = df['idx_rad'].tolist()
        idx_rads_2 = df['idx_rad2'].tolist()
        bond_orders = df['bond_order'].tolist()
        #coords_h2 = df['coords_h2_before'].tolist()
        
        coords = df["coords_h2"].values
        coords_h2_ar = [np.array(np.matrix(c.replace('\n', ';'))) for c in coords]
        coords_h2 = []
        for coord in coords_h2_ar:
            coords_h2.append(list(coord[0]))
        
        coords_all, elements_all = fu.readXYZs('{}/{}'.format(self.indir, xyz_name)) #rad_inter_systems_coords.xyz
        #print('len coords init', len(coords_all))
        forces_all, elements_f_all = fu.readXYZs('{}/{}'.format(self.indir, xyz_force_name))
        energies_init = np.load('{}/{}'.format(self.indir, energy_npy_name), allow_pickle = True)
        ## move H1 randomly in between molecules
        
        # collect sample parameters ? id, etc, shifted coords
               
        ## optimize systems
        
        # sphere & xtb optimization
        
        #save
        
        #coords_shifted_init = []
        #elements_shifted_init = []
        
        #coords_hat_all = []
        #elements_hat_all = []
        #energies_hat_all = []
        #forces_hat_all = []
        
        
        id_init_all = []
        system_names_all = []
        h_rad_dist_init_all = []
        idx_h0s_all = []
        idx_rads_all = []
        idx_rads_2_all = []
        bond_orders_all = []
        h0_r1_dist_all, h0_r2_dist_all, r1_r2_dist_all = [],[],[]
        
        
        
        for idx, system in enumerate(system_names):
            
            # append initial state
            self.coords_hat_all_intra.append(coords_all[idx])
            self.elements_hat_all_intra.append(elements_all[idx])
            
            
            #print('initial coords append', coords_all[idx])
            self.energies_hat_all_intra.append(energies_init[idx])
            
            #forces_hat_all.append(f_system)
            self.forces_hat_all_intra.append(forces_all[idx])
            
            ##
            coords_h2_init = coords_h2[idx]
            
            
            center0, r_max = hat.calculate_center_distance(coords_all[idx][idx_h0s[idx]], coords_h2_init)
            
            h0_r1_dist_init, h0_r2_dist_init, r1_r2_dist_init = hat.get_h0_r1_r2_distances(coords_all[idx], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])

            # initial config
            
            
            # add h0-r0 distance and h0-r1 distance
            
            id_init_all.append(id_init[idx])
            system_names_all.append(system_names[idx])

            h_rad_dist_init_all.append(h_rad_dist_init[idx])

            idx_h0s_all.append(idx_h0s[idx])
            idx_rads_all.append(idx_rads[idx])
            bond_orders_all.append(bond_orders[idx])
            idx_rads_2_all.append(idx_rads_2[idx])
            
            h0_r1_dist_all.append(h0_r1_dist_init)
            h0_r2_dist_all.append(h0_r2_dist_init)
            r1_r2_dist_all.append(r1_r2_dist_init)            

            pos_chr = system.count('+')
            neg_chr = -1*system.count('-')
            
            tot_chr = pos_chr+neg_chr
            
            for n_init in range(num_init_hat):
                #try:
                coords_system_shifted, energy_system, f_system = hat.get_new_h1_position(r_max, center0, coords_all[idx], elements_all[idx], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx],self.hat_dist, chrg = tot_chr, solve = self.solvent, check_bonds = check_hat_bonds)
                
                #print(coords_system_shifted)
                #coords_shifted_init.append(coords_system_shifted[:])
                
                #elements_shifted_init.append(elements_all[idx])
                
                # could add different mode here
                
                if opt_mode == 'rand_on_sphere':
                    #print('Sampling mode', opt_mode)
                    

                    
                    # shifted H
                    
                    self.coords_hat_all_intra.append(coords_system_shifted)
                    #print('shifted coords append',coords_system_shifted)
                    self.elements_hat_all_intra.append(elements_all[idx])
                    
                    h0_r1_dist, h0_r2_dist, r1_r2_dist = hat.get_h0_r1_r2_distances(coords_system_shifted, idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                    
                    #energy_system = xtb.single_point_energy(coords_system_shifted, elements_all[idx], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                    
                    self.energies_hat_all_intra.append(energy_system)
                    
                    
                    #f_system = xtb.single_force(coords_system_shifted, elements_all[idx], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                    #forces_hat_all.append(f_system)
                    self.forces_hat_all_intra.append(f_system)
                    
    
                    # add h0-r0 distance and h0-r1 distance
                    
                    id_init_all.append(id_init[idx])
                    system_names_all.append(system_names[idx])

                    h_rad_dist_init_all.append(h_rad_dist_init[idx])
 
                    idx_h0s_all.append(idx_h0s[idx])
                    idx_rads_all.append(idx_rads[idx])
                    bond_orders_all.append(bond_orders[idx])
                    idx_rads_2_all.append(idx_rads_2[idx])
                    
                    h0_r1_dist_all.append(h0_r1_dist)
                    h0_r2_dist_all.append(h0_r2_dist)
                    r1_r2_dist_all.append(r1_r2_dist)
                    
                
                if opt_mode == 'optimize_freeze':
                    print('Sampling mode', opt_mode)
                    freeze_list = hat.get_atms_outside_sphere(coords_system_shifted, idx_h0s[idx], r_cutoff)
    
                    mol_dir = '{}/xtb_opt_hat'.format(self.indir) # überschreiben lassen?
    
                    coords_opt, elements_opt = xtb.optimize_geometry(coords_system_shifted, elements_all[idx], mol_dir, charge= tot_chr, unp_e = 1, freeze_atms = freeze_list, solvent = self.solvent)
            
                    coords_opt_all,elements_opt_all, energies_opt_all = fu.read_xtboptlog(mol_dir)
                    
                    # select
                    
                    coords_selected, elements_selected, energies_selected = hat.select_steps(coords_opt_all,elements_opt_all, energies_opt_all, x, y, z)
                    
                    # append single coords selected 
                    for i in range(len(coords_selected)):
                        #coords_hat_all.append(coords_selected[i])
                        #elements_hat_all.append(elements_selected[i])
                        #energies_hat_all.append(energies_selected[i])
                        self.coords_hat_all_intra.append(coords_selected[i])
                        self.elements_hat_all_intra.append(elements_selected[i])
                        
                        self.energies_hat_all_intra.append(energies_selected[i])
                        
                        f_system = xtb.single_force(coords_selected[i], elements_selected[i], charge = tot_chr, unp_e = 1, solvent = self.solvent)
                        #forces_hat_all.append(f_system)
                        self.forces_hat_all_intra.append(f_system)
                        
                        h0_r1_dist, h0_r2_dist,r1_r2_dist = hat.get_h0_r1_r2_distances(coords_selected[i], idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                        
                        
                        
                        id_init_all.append(id_init[idx])
                        system_names_all.append(system_names[idx])
                      
                        h_rad_dist_init_all.append(h_rad_dist_init[idx])
                        
                        idx_h0s_all.append(idx_h0s[idx])
                        idx_rads_all.append(idx_rads[idx])
                        bond_orders_all.append(bond_orders[idx])
                        idx_rads_2_all.append(idx_rads_2[idx])
                        
                        h0_r1_dist_all.append(h0_r1_dist)
                        h0_r2_dist_all.append(h0_r2_dist)
                        r1_r2_dist_all.append(r1_r2_dist)
    
        
            ## add final state ##
            coords_final, elements_final, energy_final, force_final, final_state = hat.get_final_state(idx_h0s[idx], coords_all[idx], elements_all[idx],coords_h2_init, tot_chr, self.solvent)
            
            if final_state == True:

                self.coords_hat_all_intra.append(coords_final)
                self.elements_hat_all_intra.append(elements_final)
                self.energies_hat_all_intra.append(energy_final)
                self.forces_hat_all_intra.append(force_final)
                
                id_init_all.append(id_init[idx])
                system_names_all.append(system_names[idx])
                h_rad_dist_init_all.append(h_rad_dist_init[idx])
                idx_h0s_all.append(idx_h0s[idx])
                idx_rads_all.append(idx_rads[idx])
                bond_orders_all.append(bond_orders[idx])
                idx_rads_2_all.append(idx_rads_2[idx])
                
                h0_r1_dist_final, h0_r2_dist_final, r1_r2_dist_final = hat.get_h0_r1_r2_distances(coords_final, idx_h0s[idx], idx_rads[idx], idx_rads_2[idx])
                h0_r1_dist_all.append(h0_r1_dist_final)
                h0_r2_dist_all.append(h0_r2_dist_final)
                r1_r2_dist_all.append(r1_r2_dist_final)       
                
                   
        self.sample_parameters_hat_intra['id_init'] = id_init_all
        self.sample_parameters_hat_intra['system_names'] = system_names_all
      
        self.sample_parameters_hat_intra['bonds_order'] = bond_orders_all
        self.sample_parameters_hat_intra['idx_radicals'] = idx_rads_all
        self.sample_parameters_hat_intra['idx_h0s'] = idx_h0s_all
        self.sample_parameters_hat_intra['idx_rad2'] = idx_rads_2_all
        self.sample_parameters_hat_intra['h_rad_distances_init'] = h_rad_dist_init_all
        
        self.sample_parameters_hat_intra['h0_r1_dist'] = h0_r1_dist_all
        self.sample_parameters_hat_intra['h0_r2_dist'] = h0_r2_dist_all
        self.sample_parameters_hat_intra['r1_r2_dist'] = r1_r2_dist_all          
               
                    
                #except:
                    #print('encountered exception (xtb opt)')
                    #continue
        print('Finished HAT sampling')
        
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        # export init and all selected 
        
        #fu.exportXYZs(coords_hat_all, elements_hat_all, '{}/hat_coords_sampled.xyz'.format(self.outdir))
        fu.exportXYZs(self.coords_hat_all_intra, self.elements_hat_all_intra, '{}/{}_hat_intra_coords_sampled.xyz'.format(self.outdir, self.sample_name))
        
        #np.save('{}/hat_energies_sampled.npy'.format(self.outdir), energies_hat_all, allow_pickle = True)
        np.save('{}/{}_hat_intra_energies_sampled.npy'.format(self.outdir,self.sample_name), self.energies_hat_all_intra, allow_pickle = True)
        
        #fu.exportXYZs(forces_hat_all, elements_hat_all, '{}/hat_forces_sampled.xyz'.format(self.outdir))
        fu.exportXYZs(self.forces_hat_all_intra, self.elements_hat_all_intra, '{}/{}_hat_intra_forces_sampled.xyz'.format(self.outdir, self.sample_name))
        
        # export csv
        print('export csv hat')
        # need to adapt for intra HAT!
        #fu.export_csv_hat(self.outdir, id_init_all, system_names_all, donor_names_all, radical_names_all, h_rad_dist_init_all, num_atms_donor_all, num_atms_radical_all, bond_orders_all, idx_rads_all, idx_h0s_all)
        fu.export_csv_hat_intra(self.outdir, self.sample_name, self.sample_parameters_hat_intra['id_init'], self.sample_parameters_hat_intra['system_names'], self.sample_parameters_hat_intra['h_rad_distances_init'], self.sample_parameters_hat_intra['bonds_order'], self.sample_parameters_hat_intra['idx_radicals'], self.sample_parameters_hat_intra['idx_h0s'], self.sample_parameters_hat_intra['idx_rad2'],self.sample_parameters_hat_intra['h0_r1_dist'],self.sample_parameters_hat_intra['h0_r2_dist'], self.sample_parameters_hat_intra['r1_r2_dist'])


    

    def hat_sampling_plot_energies(self, outdir = None, system_type = 'Inter'):
        if outdir == None:
            outdir = self.outdir
        
        # energies

        fig, ax = plt.subplots()
        if system_type == 'Inter':
            b, bins, patches = ax.hist(self.energies_hat_all, 20, density=True, facecolor='royalblue', alpha=0.75) 
        if system_type == 'Intra':
            b, bins, patches = ax.hist(self.energies_hat_all_intra, 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$ E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title('HAT {} Radical Systems: E distribution num_samples={}'.format(system_type, len(self.energies_hat_all)))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/{}_energies_hat_{}_radical_systems_hist_{}.png'.format(outdir,self.sample_name, system_type, self.sample_name), dpi = 300)
        #plt.show()  
    
    def hat_sampling_plot_H0_distances(self, outdir = None, system_type = 'Inter'):
        
        if outdir == None:
            outdir = self.outdir
        
        if system_type == 'Inter':
            distances_h0_r1 = self.sample_parameters_hat['h0_r1_dist']
            distances_h0_r2 = self.sample_parameters_hat['h0_r2_dist']
            distances_r1_r2 = self.sample_parameters_hat['r1_r2_dist']
        if system_type == 'Intra':
            distances_h0_r1 = self.sample_parameters_hat_intra['h0_r1_dist']
            distances_h0_r2 = self.sample_parameters_hat_intra['h0_r2_dist']
            distances_r1_r2 = self.sample_parameters_hat_intra['r1_r2_dist']
        #distances_flat = [element for innerList in distances for element in innerList] ?
        
        plt.rcParams.update({'font.size': 8})
        
        n_bins = 15
        
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
        colors = ['lightcoral', 'yellowgreen']
        labels = ['H0 - R1', 'H0 - R2']
        
        ax0.hist([distances_h0_r1, distances_h0_r2], n_bins,histtype='step', stacked=False, fill=False, color= colors, label=labels)
        ax0.set_xlabel(r'Distance $[\AA]$')
        ax0.legend() #prop={'size': 10}
        ax0.set_title('H0 - R1 and H0 - R2 Distances')
        
        if system_type == 'Inter':
            ax1.hist(self.energies_hat_all, n_bins, density=True, histtype='step',fill=False, color='royalblue')
        if system_type == 'Intra':
            ax1.hist(self.energies_hat_all_intra, n_bins, density=True, histtype='step',fill=False, color='royalblue')
        ax1.set_xlabel(r'$ E [eV]$')
        ax1.set_title('Energies Unscaled')
        
        ax2.hist(distances_r1_r2, n_bins, histtype='step', color= 'purple', label='R1 - R2')
        ax2.set_xlabel(r'Distance $[\AA]$')
        ax2.set_title('R1 - R2 Distances')
        
        cm = ax3.hist2d(distances_h0_r1, distances_h0_r2, bins = 20, cmap = 'plasma') #norm=colors.LogNorm()
        fig.colorbar(cm[3], ax=ax3)
        ax3.set_xlabel(r'Distance H0 - R1 $[\AA]$')
        ax3.set_ylabel(r'Distance H0 - R2 $[\AA]$')
        #ax3.set_title()
        
        fig.suptitle(' {} HAT Sampling, Sample Size = {} '.format(system_type, len(distances_h0_r1)), fontsize=12)
        
        fig.tight_layout()
        plt.savefig('{}/{}_hat_{}_hist_{}.png'.format( outdir, self.sample_name,system_type, len(distances_h0_r1)), dpi = 300)
        
        
    def scale_and_plot_energies_hat(self, outdir = None, system_type='inter'):
        if outdir == None:
            outdir = self.outdir
        
        if system_type == 'inter':
            energies_all = self.energies_hat_all
            elements_all = self.elements_hat_all
            forces_all = self.forces_hat_all
        
        if system_type == 'intra':
            energies_all = self.energies_hat_all_intra
            elements_all = self.elements_hat_all_intra
            forces_all = self.forces_hat_all_intra
        
        scaler_en = scaler.ExtensiveEnergyForceScaler()
        
        y = [energies_all, forces_all]
        
        atomic_numbers = scaler_en.convert_elements_to_atn(elements_all)
        #print(len(atomic_numbers))
        scaler_en.fit(atomic_numbers, y)

        #print(scaler.get_weights())
        #print(scaler.get_config())

        scaler_en._plot_predict(atomic_numbers, y, outdir, self.sample_name)

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
        plt.savefig('{}/Scaled_E_rad_systems_hist_{}_{}.png'.format(outdir,system_type, self.sample_name), dpi = 300)
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
        plt.savefig('{}/hat_systems_{}_{}_2d_E_{}.png'.format( outdir,system_type, self.sample_name, len(out_e[0])), dpi = 300)

        '''
        # E-Hdistances 2d
        
        distances = self.sample_parameters_inter['h_rad_distances']
        
        fig, ax = plt.subplots()
        hist = ax.hist2d(distances, out_e[0], bins = 20, cmap = 'plasma',norm =  colors.LogNorm()) #norm=colors.LogNorm() cm2 = 
        fig.colorbar(hist[3], ax=ax) #cm2[3], ax=ax2
        ax.set_xlabel(r'Distance $\AA$')
        ax.set_ylabel(r'$ E [eV]$')
        ax.set_title('H-Radical Distances - Scaled Energies')
        fig.tight_layout()
        plt.savefig('{}/rad_systems_{}_2d_E_dist_{}.png'.format( outdir, sample_name, len(out_e[0])), dpi = 300)
        '''