#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import mol_data_generation.utils.file_utils as fu
import mol_data_generation.utils.xtb_fcts as xtb
import mol_data_generation.nms.nms_fcts as nms
import mol_data_generation.radical_systems.radical_functions as rf
import mol_data_generation.utils.scaler as scaler
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import json
import csv
import os

class NormalModeSampling():
    
    
    def __init__(self, indir,sample_name, num_samples, delta_E_max, temperature, ci_range, solvent):
        
        self.indir = indir
        self.sample_name = sample_name
        self.num_samples = num_samples
        self.delta_E_max = delta_E_max
        self.temperature = temperature
        self.ci_range = ci_range
        self.solvent = solvent
        
        self.sample_parameters = {}
        self.mol_names_list = []
        self.charge_info_list = []
        
        self.energies_all_list = []
        self.delta_energies_all_list = []
        self.relative_bonds = []
        
        self.scaled_energies_all_list = []
        
    def do_nms(self, list_dirs = None, list_dir_files_coords= None, list_dir_files_energies = None, outdir = None):
       
        # if list of indirectories is not given, then get access to files
       #
       if list_dirs == None:
           path_to_files = self.indir
           #path to conformer files    
           list_dir_files_coords =  glob.glob("{}/*/**/coords_conformers_selected.xyz".format(path_to_files))
           #print(list_dir_files_coords)
           # here function to randomly select conformers from all conformers file and write into file selected
           
           #list_dir_files_energies =  glob.glob("{}/*/**/energy_conformers_selected.npy".format(path_to_files))
           list_dirs = glob.glob("{}/*/**/".format(path_to_files))
       
       # prepare dicts_i
       dict_ki = {}
       dict_ci = {}
       dict_c_sum = {}
       dict_r_i = {}
       dict_rel_bond_i = {}
       dict_num_atoms = {}
       dict_num_samples_per_mol = {}
       mol_sequence = []

       # for each molecule
       for idx, path_dir in enumerate(list_dirs):
           
           #file_name_coords = list_dir_files_coords[idx].split("/")[-1]
           #file_name_energies = list_dir_files_energies[idx].split("/")[-1]
           
           coords_conf_in, elements_conf_in = fu.readXYZs(list_dir_files_coords[idx])
           #energies_conf_in = np.load(list_dir_files_energies[idx], allow_pickle = True)
           
           num_conf = len(coords_conf_in)
           # prepare sample_parameter dictionary
           mol_name = list_dirs[idx].split("/")[-2]
           
           mol_sequence.append(mol_name)
           
           # get num_samples/conformer
           num_samples_conf_list = nms.get_num_samples_conf(self.num_samples, num_conf)
           #print(num_samples_conf_list)
           print('Start generating samples for {}'.format(mol_name))
           # prepare 
           coords_sampled_all,elements_sampled_all = [],[]
           energies_sampled_all, delta_energies_sampled_all = [], []
           
           forces_sampled_all = []
           
           c_i_all, c_sum_all, r_i_A_all, force_constants_all, rel_bonds_all = [],[],[],[], []
           energy_0 = []
           
           pos_chr = mol_name.count('+')
           neg_chr = -1*(mol_name.count('-'))

           charg = pos_chr+neg_chr
           
           
        
           
           ## for each conformer
           for i in range(num_conf):
               #print(num_conf)
               # prepare parameters
               n_atoms = len(elements_conf_in[i])
               # get xtb vibrations
               conf_dir_name = '{}vib_analysis_conf_{}'.format(list_dirs[idx],i)
               coords_init, elements_init, normal_coords, force_constants, nf = xtb.vibrational_frequencies(coords_conf_in[i], elements_conf_in[i], conf_dir_name, charge = charg, solvent = self.solvent)
               
               force_constants_all.append(force_constants)
               
               energy_init = xtb.single_point_energy(coords_init, elements_init, charge = charg, solvent = self.solvent)
               force_init = xtb.single_force(coords_init, elements_init, charge = charg, solvent = self.solvent)
               
               # get bond info initial conformer
               bond_idx_list_conf,bond_atm_list_conf, bond_lengths_list_conf = rf.read_out_bonds(conf_dir_name)
               
               if i == 0:
                   energy_0.append(energy_init)
               
               
               ## do nms in range num_samples_conf_list[i]
               
               coords_sampled_conf, elements_sampled_conf, energies_sampled_conf, delta_energies_sampled_conf,forces_sampled_conf, c_sum_conf,c_i_conf, r_i_conf,relative_bonds_nms = nms.do_sampling(coords_init, elements_init, normal_coords, force_constants,n_atoms, nf, num_samples_conf_list[i],energy_init, force_init, energy_0[0], self.delta_E_max, self.temperature,self.ci_range, bond_idx_list_conf, bond_lengths_list_conf, chrg = charg, solve = self.solvent)
               #print(len(coords_sampled_conf))
               # 
               coords_sampled_all.append(coords_sampled_conf)
               elements_sampled_all.append(elements_sampled_conf)
               #print('len energies conf', len(energies_sampled_conf))
               #print(energies_sampled_conf)
               for en in energies_sampled_conf:
                   energies_sampled_all.append(en)
               for de in delta_energies_sampled_conf:
                   delta_energies_sampled_all.append(de)
                   
               for fc in forces_sampled_conf:
                   forces_sampled_all.append(fc)
               
               #print('len rel bonds conf', len(relative_bonds_nms))
               #print(relative_bonds_nms)
               for rb in relative_bonds_nms:
                   rel_bonds_all.append(rb)
                   #self.relative_bonds.append(rb)
                   
               for ci in c_i_conf:
                   c_i_all.append(ci)
               for csum in c_sum_conf:
                   c_sum_all.append(csum)
               for riA in r_i_conf:
                   r_i_A_all.append(riA)
            
               # also return other parameters for evaluation
               #c_i_all.append(c_i_conf)
               #c_sum_all.append(c_sum_conf)
               #r_i_A_all.append(r_i_conf)
               #rel_bonds_all.append(relative_bonds)
               #self.relative_bonds.append(relative_bonds)
           
           ## export xyz, npy
           # add stuff to dictionary, rest export
           dict_ki[mol_name] = force_constants_all
           dict_ci[mol_name] = c_i_all
           dict_c_sum[mol_name] = c_sum_all
           dict_r_i[mol_name] = r_i_A_all
           dict_rel_bond_i[mol_name] = rel_bonds_all
           dict_num_atoms[mol_name] = len(coords_init)
           dict_num_samples_per_mol[mol_name] = len(energies_sampled_all)
           
           self.mol_names_list.append(self.num_samples*[mol_name])
           self.charge_info_list.append(self.num_samples*[charg])
           
           # export 
           if outdir == None:
               outdir_ex = list_dirs[idx]
               
           else:
               if not os.path.exists(outdir):
                   os.makedirs(outdir)
                   
               outdir_ex = outdir
               
           fu.export_sampled_coords_energies(coords_sampled_all, elements_sampled_all, energies_sampled_all, delta_energies_sampled_all,forces_sampled_all, outdir_ex, mol_name,self.num_samples, self.delta_E_max, self.temperature)
           
           print('Finsished nms for {}'.format(mol_name)) # change where to export 
           
           #print('Calculate Forces for Samples')
           
           # do in inner loop
           #forces_sampled_all = xtb.calculate_forces(coords_sampled_all, elements_sampled_all)
           
           #fu.export_forces(forces_sampled_all, elements_sampled_all, list_dirs[idx], mol_name,self.num_samples, self.delta_E_max, self.temperature)
           
           #print('Finished force calculations for {}'.format(mol_name))
           
       
       
       
       # export parameter dictionary
       self.sample_parameters['force_constants'] = dict_ki
       self.sample_parameters['c_i'] = dict_ci
       self.sample_parameters['c_sum'] = dict_c_sum
       self.sample_parameters['r_i_A'] = dict_r_i
       self.sample_parameters['rel_bond_dist'] = dict_rel_bond_i
       self.sample_parameters['num_atoms'] = dict_num_atoms
       self.sample_parameters['num_samples_per_mol'] = dict_num_samples_per_mol
       
       # export all
       # concat all
       
       if outdir == None:
           outdir_conc = self.indir
       else:
           outdir_conc = outdir
       
       self.concat_all_data(outdir_all = outdir_conc, mol_seq = mol_sequence)
       
       
       if outdir == None:
           outdir_dict = self.indir
           
       else:
           outdir_dict = outdir
           
       ## change output dir
       np.save('{}/sample_parameters_dict_{}.npy'.format(outdir_dict, self.sample_name), self.sample_parameters, allow_pickle=True) 
       #json.dump( self.sample_parameters, open( '{}sample_parameters_dict_{}.json'.format(self.indir, self.sample_name), 'w' ) )
       print('Saved NMS Parameter dictionary.')
           
    
    def plot_bond_lengths(self, outdir = None, rel_bond_parameter_dict_path = None, hist_width = 15, parameter_dict_path = None):
        
        if outdir == None:
            outdir = self.indir
        
        #datanew = np.load('file.npy')[()]
        if parameter_dict_path == None:
            parameter_dict_name = '{}/sample_parameters_dict_{}.npy'.format(self.indir, self.sample_name)
        else:
            parameter_dict_name = '{}/sample_parameters_dict_{}.npy'.format(parameter_dict_path, self.sample_name)
        
        param_dict = np.load('{}'.format(parameter_dict_name), allow_pickle=True)[()]
        
    
        ## get into right shape
        
        rel_bonds = [] 
        
        for mol in param_dict['rel_bond_dist']:
            for i in range(len(param_dict['rel_bond_dist'][mol])):
                rel_bonds.append(param_dict['rel_bond_dist'][mol][i])
        
        rel_bonds_flat = [element for innerList in rel_bonds for element in innerList] #self.relative_bonds 
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(rel_bonds_flat, hist_width, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$r/r_0$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title(r'NMS: Relative bond distances num_samples={}, $\Delta Emax$ = {} [eV]'.format(len(self.energies_all_list), self.delta_E_max))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/relative_bond_dist_nms_hist_{}.png'.format(outdir, self.sample_name), dpi = 300)
        #plt.show()

    
    def plot_nms_energies(self, outdir = None):
        
        if outdir == None:
            outdir = self.indir
        
        # energies
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(self.energies_all_list, 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$ E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title('NMS: E distribution num_samples={}, $\Delta Emax$ = {} [eV]'.format(len(self.energies_all_list), self.delta_E_max))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/E_nms_hist_{}.png'.format(outdir, self.sample_name), dpi = 300)
        #plt.show()  
        
        # delta energies
        
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(self.delta_energies_all_list, 20, density=True, facecolor='royalblue', alpha=0.75) 
        ax.set_xlabel(r'$\Delta E [eV]$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title(r'$\Delta E$ distribution NMS num_samples={}, $\Delta Emax$ = {} [eV]'.format(len(self.delta_energies_all_list), self.delta_E_max))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/delta_E_nms_hist_{}.png'.format(outdir, self.sample_name), dpi = 300)


    def concat_all_data(self, list_dirs = None, outdir_all = None, mol_seq = None):
        
        if mol_seq == None:
            print('Mol Sequence = 0')
        # concat and save
        
        list_files_coords = []
        list_files_forces = []
        list_files_energies = []
        list_files_delta_energies = []
        
        if list_dirs == None:
            #path to files    
            if outdir_all == None:
                print('Specify outdir')
                #list_files_coords = glob.glob("{}/*/**/*_nms_samples_num{}_T{}_Emax{}.xyz".format(self.indir, self.num_samples, self.temperature, int(self.delta_E_max) ))
                #print(list_files_coords)
                #list_files_forces = glob.glob("{}/*/**/*_nms_forces_num{}_T{}_Emax{}.xyz".format(self.indir, self.num_samples, self.temperature, int(self.delta_E_max) ))               
                #list_files_energies = glob.glob("{}/*/**/*_nms_energies_num{}_T{}_Emax{}.npy".format(self.indir, self.num_samples, self.temperature, int(self.delta_E_max) ))                
                #list_files_delta_energies = glob.glob("{}/*/**/*_nms_delta_energies_num{}_T{}_Emax{}.npy".format(self.indir, self.num_samples, self.temperature, int(self.delta_E_max) ))
            else: 
                
                for mol in mol_seq:
                    #list_dirs = glob.glob("{}/*/**/".format(path_to_files))
                    
                    list_files_coords.append(glob.glob("{}/*/{}/{}_nms_samples_num{}_T{}_Emax{}.xyz".format(outdir_all,mol,mol, self.num_samples, self.temperature, int(self.delta_E_max) ))[0])
                    list_files_forces.append(glob.glob("{}/*/{}/{}_nms_forces_num{}_T{}_Emax{}.xyz".format(outdir_all,mol,mol, self.num_samples, self.temperature, int(self.delta_E_max) ))[0])
                    list_files_energies.append(glob.glob("{}/*/{}/{}_nms_energies_num{}_T{}_Emax{}.npy".format(outdir_all, mol, mol,self.num_samples, self.temperature, int(self.delta_E_max) ))[0])
                    list_files_delta_energies.append(glob.glob("{}/*/{}/{}_nms_delta_energies_num{}_T{}_Emax{}.npy".format(outdir_all, mol,mol, self.num_samples, self.temperature, int(self.delta_E_max) ))[0])
                    
                    #list_files_coords.append("{}/{}_nms_samples_num{}_T{}_Emax{}.xyz".format(outdir_all,mol, self.num_samples, self.temperature, int(self.delta_E_max) ))
                    #list_files_forces.append("{}/{}_nms_forces_num{}_T{}_Emax{}.xyz".format(outdir_all,mol, self.num_samples, self.temperature, int(self.delta_E_max) ))
                    #list_files_energies.append("{}/{}_nms_energies_num{}_T{}_Emax{}.npy".format(outdir_all, mol, self.num_samples, self.temperature, int(self.delta_E_max) ))
                    #list_files_delta_energies.append("{}/{}_nms_delta_energies_num{}_T{}_Emax{}.npy".format(outdir_all, mol, self.num_samples, self.temperature, int(self.delta_E_max) ))
                
        
        #print('files coords', list_files_coords)    
        #print('files_forces', list_files_forces)
        #print('files en', list_files_energies)
        
        if outdir_all == None:
            outdir_all = self.indir                
        
        # coords
        coords_all, elements_all = [],[]
        len_list = []
        for file in list_files_coords:
            coords, elements = fu.readXYZs(file)
            len_list.append(len(coords))
            
            for i in range(len(coords)):
                coords_all.append(coords[i])
                elements_all.append(elements[i])
                
        fu.exportXYZs(coords_all, elements_all, '{}/nms_coords_all_{}.xyz'.format(outdir_all, self.sample_name))
        
        # forces
        forces_all, elements_all_f = [], []

        for path in list_files_forces:
            forces, elements = fu.readXYZs(path)
            len_list.append(len(forces))
            for i in range(len(forces)):
                forces_all.append(forces[i])
                elements_all_f.append(elements[i])

        fu.exportXYZs(forces_all, elements_all_f, '{}/nms_forces_all_{}.xyz'.format(outdir_all, self.sample_name))
       
        # energies
        
        for path in list_files_energies:
            energies = np.load(path, allow_pickle=True)

            for i in range(len(energies)):
                self.energies_all_list.append(energies[i])

        np.save('{}/nms_energies_all_{}.npy'.format(outdir_all, self.sample_name),
                np.array(self.energies_all_list), allow_pickle=True)
        
        # delta energies

        for path in list_files_delta_energies:
            d_energies = np.load(path, allow_pickle=True)

            for i in range(len(d_energies)):
                self.delta_energies_all_list.append(d_energies[i])

        np.save('{}/nms_delta_energies_all_{}.npy'.format(outdir_all, self.sample_name),
                np.array(self.delta_energies_all_list), allow_pickle=True)
       
        print('concatenated data and saved {}'.format(self.sample_name))
        
        # csv nms

        names_all_flat = [element for innerList in self.mol_names_list for element in innerList]
        
        charges_all_flat = [element for innerList in self.charge_info_list for element in innerList]
        id_list = list(range(len(names_all_flat)))
        
        with open('{}/nms_info_{}.csv'.format(outdir_all, self.sample_name), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'names', 'charge', 'energy', 'force'])
            print('written header')
            for i in range(len(id_list)):
                writer.writerow([id_list[i], names_all_flat[i], charges_all_flat[i],
                                self.energies_all_list[i], forces_all[i]])
            
        with open('{}/qm.csv'.format(outdir_all), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'energy', 'force'])
            print('written header')
            for i in range(len(id_list)):
                writer.writerow([id_list[i], self.energies_all_list[i], forces_all[i]])
            


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
        
        
        
    def plot_analysis(self, outdir = None, parameter_dict_path= None, hist_width = 20):
        
        if outdir == None:
            outdir = self.indir
            
        # plot c sum , 2d hist
        #datanew = np.load('file.npy')[()]
        if parameter_dict_path == None:
            parameter_dict_name = '{}/sample_parameters_dict_{}.npy'.format(self.indir, self.sample_name)
        else:
            parameter_dict_name = '{}/sample_parameters_dict_{}.npy'.format(parameter_dict_path, self.sample_name)
        
        param_dict = np.load('{}'.format(parameter_dict_name), allow_pickle=True)[()]
        ## get into right shape
        
        rel_bonds = [] 
        num_atoms_rel_bonds = []
        for mol in param_dict['rel_bond_dist']:
            for i in range(len(param_dict['rel_bond_dist'][mol])):
                rel_bonds.append(param_dict['rel_bond_dist'][mol][i])
                num_atoms_rel_bonds.append([param_dict['num_atoms'][mol]]*len(param_dict['rel_bond_dist'][mol][i]))
                
        rel_bonds_flat = [element for innerList in rel_bonds for element in innerList] #self.relative_bonds 
        num_atoms_rel_bonds_flat = [element for innerList in num_atoms_rel_bonds for element in innerList] 
        
        c_sum_all = [] 
        num_atoms_csum = []
        for mol in param_dict['c_sum']:
            for i in range(len(param_dict['c_sum'][mol])):
                c_sum_all.append(param_dict['c_sum'][mol][i])
                num_atoms_csum.append(param_dict['num_atoms'][mol]) #[0]??
        
        # c_sum hist
        fig, ax = plt.subplots()
        b, bins, patches = ax.hist(c_sum_all, hist_width, density=True, facecolor='yellowgreen', alpha=0.75) 
        ax.set_xlabel(r'$c_{sum}$')
        #ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title(r'NMS: $csum$ Distribution num_samples={}, $\Delta Emax$ = {} [eV]'.format(len(self.energies_all_list), self.delta_E_max))
        #plt.grid(True)
        fig.tight_layout()
        plt.savefig('{}/c_sum_dist_nms_hist_{}.png'.format(outdir, self.sample_name), dpi = 300)
        #plt.show()
        
        
        num_atoms_E = []
        for mol in param_dict['num_atoms']:
            num_atoms_E.append([param_dict['num_atoms'][mol]]* param_dict['num_samples_per_mol'][mol])

        num_atoms_E_flat = [element for innerList in num_atoms_E for element in innerList]        
        #print('len num atoms', len(num_atoms_E_flat))
        #print(num_atoms_E_flat)
        #print(len(self.scaled_energies_all_list))
        scaled_en_all = list(self.scaled_energies_all_list[0])
        
        #print('num scaled en', len(scaled_en_all))
        
        # plot 2D hist
        
        # number of atoms
        
        plt.rcParams.update({'font.size': 8})
        
        #n_bins = 15
        
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
        #colors = ['lightcoral', 'yellowgreen']
        #labels = ['H0 - R1', 'H0 - R2']
        
        cm0 = ax0.hist2d(num_atoms_csum, c_sum_all, bins = 20, cmap = 'plasma', norm = colors.LogNorm()) #norm=colors.LogNorm()
        fig.colorbar(cm0[3], ax=ax0)
        ax0.set_xlabel(r'Number of atoms/molecule')
        ax0.set_ylabel(r'$c_{sum}$')
        ax0.set_title(r'$c_{sum}$ Distribution')
       
        
        cm1 = ax1.hist2d(num_atoms_rel_bonds_flat, rel_bonds_flat, bins = 20, cmap = 'plasma', norm =  colors.LogNorm()) #norm=colors.LogNorm()
        fig.colorbar(cm1[3], ax=ax1)
        ax1.set_xlabel(r'Number of atoms/molecule')
        ax1.set_ylabel(r'$r/r_0$')
        ax1.set_title(r'Bond Length Distribution')

        cm2 = ax2.hist2d(num_atoms_E_flat, scaled_en_all, bins = 20, cmap = 'plasma',norm =  colors.LogNorm()) #norm=colors.LogNorm()
        fig.colorbar(cm2[3], ax=ax2)
        ax2.set_xlabel('Number of atoms/molecule')
        ax2.set_ylabel(r'$ E [eV]$')
        ax2.set_title('Scaled Energies')
        
        cm = ax3.hist2d(num_atoms_E_flat, self.delta_energies_all_list, bins = 20, cmap = 'plasma',norm =  colors.LogNorm()) #norm=colors.LogNorm()
        fig.colorbar(cm[3], ax=ax3)
        ax3.set_xlabel('Number of atoms/molecule')
        ax3.set_ylabel(r'$ E [eV]$')
        ax3.set_title(r'$\Delta$ Energies')
        
        fig.suptitle('NMS Sampling, Sample Size = {} '.format(len(scaled_en_all)), fontsize=12)
        
        fig.tight_layout()
        plt.savefig('{}/nms_{}_2d_hist_{}.png'.format( outdir, self.sample_name, len(scaled_en_all)), dpi = 300)
        
        
        
        plt.rcParams.update({'font.size': 8})
        
        #n_bins = 15
        
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
        #colors = ['lightcoral', 'yellowgreen']
        #labels = ['H0 - R1', 'H0 - R2']
        
        cm0 = ax0.scatter(num_atoms_csum, c_sum_all, marker='.', s=3**2) #norm=colors.LogNorm()
        #fig.colorbar(cm0[3], ax=ax0)
        ax0.set_xlabel(r'Number of atoms/molecule')
        ax0.set_ylabel(r'$c_{sum}$')
        ax0.set_title(r'$c_{sum}$ Distribution')
       
        
        cm1 = ax1.scatter(num_atoms_rel_bonds_flat, rel_bonds_flat, marker='.', s=3**2) #norm=colors.LogNorm()
        #fig.colorbar(cm1[3], ax=ax1)
        ax1.set_xlabel(r'Number of atoms/molecule')
        ax1.set_ylabel(r'$r/r_0$')
        ax1.set_title(r'Bond Length Distribution')

        cm2 = ax2.scatter(num_atoms_E_flat, scaled_en_all, marker='.', s=3**2) #norm=colors.LogNorm()
        #fig.colorbar(cm2[3], ax=ax2)
        ax2.set_xlabel('Number of atoms/molecule')
        ax2.set_ylabel(r'$ E [eV]$')
        ax2.set_title('Scaled Energies')
        
        cm = ax3.scatter(num_atoms_E_flat, self.delta_energies_all_list, marker='.', s=3**2) #norm=colors.LogNorm()
        #fig.colorbar(cm[3], ax=ax3)
        ax3.set_xlabel('Number of atoms/molecule')
        ax3.set_ylabel(r'$ E [eV]$')
        ax3.set_title(r'$\Delta$ Energies')
        
        fig.suptitle('NMS Sampling, Sample Size = {} '.format(len(scaled_en_all)), fontsize=12)
        
        fig.tight_layout()
        plt.savefig('{}/nms_{}_scatter_analysis_{}.png'.format( outdir, self.sample_name, len(scaled_en_all)), dpi = 300)

    


'''
# make csv with names of files

## names
# samples_aa
indir_aa = 'samples_aa/aa'
files_list_aa_names = glob.glob('{}/*/'.format(indir_aa))
# samples aa caps, amide
indir_aa_caps = 'samples_test_aa'
files_list_aa_caps_names = glob.glob('{}/*/*/'.format(indir_aa_caps))
# samples aa caps, amide rest
indir_aa_caps_rest = 'samples_rest_aa_caps_amide'
files_list_aa_caps_rest_names = glob.glob('{}/*/*/'.format(indir_aa_caps_rest))

list_dir_names = files_list_aa_names + files_list_aa_caps_names + files_list_aa_caps_rest_names
names_all = []

for idx, path in enumerate(list_dir_names):
    path_str_list = path.split("/")
    # print(path_str_list)
    names_all.append([path_str_list[-2]]*len_list[idx])

names_all_flat = [element for innerList in names_all for element in innerList]
id_list = list(range(len(names_all_flat)))

with open('aa_nms_all_info.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'names', 'energy', 'force'])
    print('written header')
    for i in range(len(id_list)):
        writer.writerow([id_list[i], names_all_flat[i],
                        e_all[i], f_all[i]])
    #writer.writerows(list(zip(id_list, names_all_flat, energies_all, grads_all)))

with open('qm.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'energy', 'force'])
    print('written header')
    for i in range(len(id_list)):
        writer.writerow([id_list[i], e_all[i], f_all[i]])
    #writer.writerows(list(zip(id_list, names_all_flat, energies_all, grads_all)))

#aa_nms_coords_all.xyz  aa_nms_energies_all.npy  aa_nms_forces_all.xyz  qm.csv hyper_nms_aa.py train_schnet_nms_aa.py


           self.sample_parameters['force_constants'] = {'{}'.format(mol_name): force_constants_all}
           self.sample_parameters['c_i'] = {'{}'.format(mol_name): c_i_all}
           self.sample_parameters['c_sum'] = {'{}'.format(mol_name): c_sum_all}
           self.sample_parameters['r_i_A'] = {'{}'.format(mol_name): r_i_A_all}

#json.dump( self.sample_parameters, open("{}/sample_parameters_dict.json".format(self.indir), 'w' ) )

my_dict = {'a' : np.array(range(3)), 'b': np.array(range(4))}

np.save('my_dict.npy',  my_dict)    

my_dict_back = np.load('my_dict.npy')

print(my_dict_back.item().keys())    
print(my_dict_back.item().get('a'))

                for e in nms_single.energies_sampled:
                    energies_sampled_all.append(e)
                #energies_sampled_all.append(nms.energies_sampled)
                
                for e in nms_single.delta_energies_sampled:
                    delta_energies_sampled_all.append(e)
                #delta_energies_sampled_all.append(nms.delta_energies_sampled)



coords_conformers_selected.xyz
energy_conformers_selected.npy 
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
            
            coords_in, elements_in = fu.readXYZ(list_dir_files[idx])
            
            # for each file/ mol generate conf according to settings
            coords_out, elements_out, energies_out = xtb.run_crest(coords_in, elements_in, path_dir, file_name, settings_conf['crest_fast'], settings_conf['crest_method'])
            
            # save all energies of conf to npy
            # unit energies??
            # check dir ?
            np.save('{}conformer_all_energies.npy'.format(path_dir), np.array(energies_out), allow_pickle=True)

'''