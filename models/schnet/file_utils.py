#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import csv

HToeV = 27.21138624598853

def readXYZ(filename):
    infile=open(filename,"r")
    coords=[]
    elements=[]
    lines=infile.readlines()
    if len(lines)<3:
        exit("ERROR: no coordinates found in %s/%s"%(os.getcwd(), filename))
    for line in lines[2:]:
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    coords=np.array(coords)
    return coords,elements



def readXYZs(filename):
    infile=open(filename,"r")
    coords=[[]]
    elements=[[]]
    for line in infile.readlines():
        if len(line.split())==1 and len(coords[-1])!=0:
            coords.append([])
            elements.append([])
        elif len(line.split())==4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    return coords,elements

def exportXYZ(coords,elements,filename, mask=[]):
    outfile=open(filename,"w")

    if len(mask)==0:
        outfile.write("%i\n\n"%(len(elements)))
        for atomidx,atom in enumerate(coords):
            outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))
    else:
        outfile.write("%i\n\n"%(len(mask)))
        for atomidx in mask:
            atom = coords[atomidx]
            outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()

def exportXYZs(coords,elements,filename):
    outfile=open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements[idx])))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[idx][atomidx].capitalize(),atom[0],atom[1],atom[2]))

    outfile.close()
    

# g98.out
# Frequencies
def read_frequencies(dirname):
    vib_frequencies = []
    for line in open('{}g98.out'.format(dirname),'r'): #/
        if 'Frequencies' in line:
            for x in line.split()[2:]:
                try:
                    vib_frequencies.append(float(x))
                except x == None:
                    pass
    vib_frequencies = np.array(vib_frequencies)
    
    return vib_frequencies
    
def read_red_masses(dirname):
    red_masses = []
    for line in open('{}g98.out'.format(dirname),'r'): #/
        if 'Red. masses' in line:
            for x in line.split()[3:]:
                try:
                    red_masses.append(float(x))
                except x == None:
                    pass
    red_masses = np.array(red_masses)
    # change units?
    return red_masses

def read_normal_coordinates(dirname, num_atoms):
    normal_coords = []
    
    with open('{}g98.out'.format(dirname),'r') as f: #/
        lines = f.readlines()
        for i in range(len(lines)):
            if 'Atom AN' in lines[i]:
                mode_1, mode_2, mode_3 = [], [], []
                for j in range(i+1, i+1+num_atoms):
                    line_j = lines[j].split()
                    try:
                        mode_1.append([float(line_j[2]),float(line_j[3]),float(line_j[4])])
                        mode_2.append([float(line_j[5]),float(line_j[6]),float(line_j[7])])
                        mode_3.append([float(line_j[8]),float(line_j[9]),float(line_j[10])])
                    except IndexError:
                        continue
                if mode_1 != []:
                    normal_coords.append(mode_1)
                if mode_2 != []:
                    normal_coords.append(mode_2)                    
                if mode_3 != []:
                    normal_coords.append(mode_3)           
    
    
    normal_coords_arr = []
    for nm in normal_coords:
        nm_i = []
        for i in range(len(nm)):
            nm_i.append(np.array(nm[i]))
        normal_coords_arr.append(np.array(nm_i))
    
    return normal_coords_arr

def export_sampled_coords_energies(coords_sampled_all, elements_sampled_all, energies_sampled_all, delta_energies_sampled_all, forces_sampled_all, outdir, mol_name,num_samples, delta_E_max, temperature):
    #print(len(elements_sampled_all), elements_sampled_all[0])
    coords_sampled_all_list, elements_sampled_all_list = [],[]
    for j in range(len(coords_sampled_all)):
        for l in range(len(coords_sampled_all[j])):
            coords_sampled_all_list.append(coords_sampled_all[j][l])
            elements_sampled_all_list.append(elements_sampled_all[j][l])
    
    exportXYZs(coords_sampled_all_list, elements_sampled_all_list, '{}{}_nms_samples_num{}_T{}_Emax{}.xyz'.format(outdir, mol_name,num_samples, temperature,int(delta_E_max)))
    
    #forces_sampled_all_list = [] # needed?
    #for j in range(len(forces_sampled_all)):
    #    for l in range(len(forces_sampled_all[j])):
    #        forces_sampled_all_list.append(forces_sampled_all[j][l])
    
    exportXYZs(forces_sampled_all, elements_sampled_all_list, '{}{}_nms_forces_num{}_T{}_Emax{}.xyz'.format(outdir, mol_name,num_samples, temperature,int(delta_E_max)))
    
    np.save('{}{}_nms_energies_num{}_T{}_Emax{}.npy'.format(outdir, mol_name,num_samples, temperature,int(delta_E_max)), energies_sampled_all, allow_pickle =True)
    np.save('{}{}_nms_delta_energies_num{}_T{}_Emax{}.npy'.format(outdir, mol_name,num_samples, temperature,int(delta_E_max)), delta_energies_sampled_all, allow_pickle =True)

def export_forces(forces_sampled_all, elements_sampled_all, outdir, mol_name,num_samples, delta_E_max, temperature):   
    elements_sampled_all_list = [] #forces_sampled_all_list, 
    #print(len(forces_sampled_all), len(forces_sampled_all[0]))
    #print(len(elements_sampled_all), len(elements_sampled_all[0]))
    for j in range(len(elements_sampled_all)):
        for l in range(len(elements_sampled_all[j])):
            elements_sampled_all_list.append(elements_sampled_all[j][l])
    
    exportXYZs(forces_sampled_all, elements_sampled_all_list, '{}{}_nms_forces_num{}_T{}_Emax{}.xyz'.format(outdir, mol_name,num_samples, temperature,int(delta_E_max)))
    

def export_csv_rad_systems(outdir,sample_name, system_names, donor_names, radical_names, h_rad_distance, num_atms_don, num_atms_rad, bonds_systems, idx_radicals, idx_h0s, idx_r2):
    id_list = list(range(len(system_names)))
    with open('{}/{}_csv_init_radical_systems_info.csv'.format(outdir, sample_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'system_name', 'donor_name', 'radical_name', 'h_rad_dist_init', 'num_atms_donor', 'num_atms_radical', 'idx_h0', 'idx_rad', 'idx_rad2' ,'bond_order'])
        
        for i in range(len(id_list)):
            writer.writerow([id_list[i], system_names[i], donor_names[i], radical_names[i], h_rad_distance[i], num_atms_don[i], num_atms_rad[i], idx_h0s[i], idx_radicals[i], idx_r2[i] ,bonds_systems[i]])
            
    print('csv file with initial radical system information written.')




def export_csv_rad_systems_intra(outdir,sample_name, system_names, h_rad_distance, bonds_systems, idx_radicals, idx_h0s, idx_r2, coords_h2):
    id_list = list(range(len(system_names)))
    with open('{}/{}_csv_init_radical_systems_intra_info.csv'.format(outdir, sample_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'system_name', 'h_rad_dist_init', 'idx_h0', 'idx_rad', 'idx_rad2' ,'coords_h2', 'bond_order'])
        
        for i in range(len(id_list)):
            writer.writerow([id_list[i], system_names[i], h_rad_distance[i], idx_h0s[i], idx_radicals[i], idx_r2[i] ,coords_h2[i] ,bonds_systems[i]])
            
    print('csv file with initial radical system information written.')

def export_csv_rad_systems_inter_V2(outdir,sample_name, system_names, h_rad_distance, bonds_systems, idx_radicals, idx_h0s, idx_r2, coords_h2,donor_names, radical_names,num_atms_don, num_atms_rad):
    id_list = list(range(len(system_names)))
    with open('{}/{}_csv_init_radical_systems_inter_info.csv'.format(outdir, sample_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'system_name', 'h_rad_dist_init', 'idx_h0', 'idx_rad', 'idx_rad2' ,'coords_h2','donor_name','radical_name','num_atms_donor','num_atms_radical','bond_order'])
        
        for i in range(len(id_list)):
            writer.writerow([id_list[i], system_names[i], h_rad_distance[i], idx_h0s[i], idx_radicals[i], idx_r2[i] ,coords_h2[i] ,donor_names[i],radical_names[i],num_atms_don[i],num_atms_rad[i], bonds_systems[i]])
            
    print('csv file with initial radical system information written.')

# fu.export_csv_rad_systems_inter_V2(outdir,sample_name, system_names, h_rad_distances, bonds_systems, idx_radicals, idx_h0s, idx_r2, coords_h2, donor_names, radical_names,num_atms_don, num_atms_rad)

def export_csv_hat(outdir, sample_name, id_init, system_names, donor_names, radical_names, h_rad_distance, num_atms_don, num_atms_rad, bonds_systems, idx_radicals, idx_h0s, idx_rad2, h0r1dist, h0r2dist, r1r2dist):
    id_list = list(range(len(system_names)))
    with open('{}/{}_csv_hat_systems_info.csv'.format(outdir, sample_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'system_name', 'donor_name', 'radical_name', 'h_rad_dist_init', 'num_atms_donor', 'num_atms_radical', 'idx_h0', 'idx_rad', 'idx_rad2', 'h0_r1_dist', 'h0_r2_dist' , 'r1_r2_dist', 'bond_order'])
        
        for i in range(len(id_list)):
            writer.writerow([id_init[i], system_names[i], donor_names[i], radical_names[i], h_rad_distance[i], num_atms_don[i], num_atms_rad[i], idx_h0s[i], idx_radicals[i],idx_rad2[i],h0r1dist[i],h0r2dist[i],r1r2dist[i] ,bonds_systems[i]])
            
    print('csv file with hat system information written.')


def export_csv_hat_intra(outdir, sample_name, id_init, system_names, h_rad_distance, bonds_systems, idx_radicals, idx_h0s, idx_rad2, h0r1dist, h0r2dist, r1r2dist):
    id_list = list(range(len(system_names)))
    with open('{}/{}_csv_hat_intra_systems_info.csv'.format(outdir, sample_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'system_name','h_rad_dist_init', 'idx_h0', 'idx_rad', 'idx_rad2', 'h0_r1_dist', 'h0_r2_dist' , 'r1_r2_dist', 'bond_order'])
        
        for i in range(len(id_list)):
            writer.writerow([id_init[i], system_names[i], h_rad_distance[i], idx_h0s[i], idx_radicals[i],idx_rad2[i],h0r1dist[i],h0r2dist[i],r1r2dist[i] ,bonds_systems[i]])
            
    print('csv file with hat intra system information written.')


def read_xtboptlog(moldir):
    
    infile = open('{}/xtb_geo_opt/xtbopt.log'.format(moldir))
    
    coords = [[]]
    elements = [[]]
    energies = []
    
    for line in infile.readlines():
        if len(line.split())==1 and len(coords[-1])!=0:
            coords.append([])
            elements.append([])
        if len(line.split())==7:
            energy = float(line.split()[1])*HToeV # need to transform Eh energy in eV
            energies.append(energy) 
        if len(line.split())==4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    return coords,elements, energies




        