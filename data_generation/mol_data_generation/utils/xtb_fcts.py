#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import absolute_import
import shutil
import uuid
import os
import sys
import numpy as np
import time
import subprocess
import shlex
import yaml

import mol_data_generation.utils.file_utils as fu
import mol_data_generation.nms.nms_fcts as nms

kcal_to_eV=0.0433641153
kB=8.6173303e-5 #eV/K
T=298.15
kBT=kB*T
AToBohr=1.889725989
HToeV = 27.21138624598853
HBtoevA = 27.21138624598853/0.52917721090380

# add E, G -> F xtb calculations here
### single point energy ###
def single_point_energy(coords, elements, charge = 0, unp_e = 0, outdir = None, solvent = None):
    results = xtb_calc(coords, elements, opt=False, grad=False, hess=False, chrg=charge, uhf = unp_e, freeze=[],dirname = outdir, solve = solvent)
    e = results["energy"] # in eV
    return e

def single_force(coords, elements, charge = 0, unp_e = 0, solvent = None):
    results = xtb_calc(coords, elements, opt=False, grad=True, hess=False, chrg=charge, uhf = unp_e, freeze=[], dirname=None, solve = solvent)
    forces = results['gradient']
    return forces
### hessian & vibrational frequencies ###
def vibrational_frequencies(coords, elements, moldir, charge = 0, solvent = None):
    
    startdir=os.getcwd()
   
    if not os.path.exists(moldir):
        os.makedirs(moldir)
    
    if startdir != '{}'.format(moldir):    
        os.chdir('{}'.format(moldir))    
    
    
    if not os.path.exists("g98.out"):
        # calculate vib freq
        os.chdir(startdir)
        results = xtb_calc(coords, elements, opt=True, grad=False, hess=True, chrg=charge, freeze=[], dirname=moldir, solve = solvent)
        coords_new = results['coords']
        elements_new = results['elements']        
        # function to read g98.out and return k, nf
        moldir = moldir+'/'
        normal_coords, force_constants, nf = nms.get_vib_parameters(moldir, len(elements_new))
        
    else:
        print("   ---   found g98.out and read output")
        # function to read g98.out and return k, nf
        coords_new = coords
        elements_new = elements
        moldir_ex = ''
        normal_coords, force_constants, nf = nms.get_vib_parameters(moldir_ex, len(elements_new))

    os.chdir(startdir)
    
    return coords_new, elements_new, normal_coords, force_constants, nf

### calculate forces ###
def calculate_forces(coords_all, elements_all):
    
    coords_sampled_all_list, elements_sampled_all_list = [],[]
    for j in range(len(coords_all)):
        for l in range(len(coords_all[j])):
            coords_sampled_all_list.append(coords_all[j][l])
            elements_sampled_all_list.append(elements_all[j][l])
    
    forces_sampled_all = []
    
    for coords, elements in zip(coords_sampled_all_list, elements_sampled_all_list):
        results = xtb_calc(coords, elements, opt=False, grad=True, hess=False, charge=0, freeze=[], dirname=None)
        
        forces = results['gradient']
        forces_sampled_all.append(forces)
        #print(forces.shape)
        #print(forces)
    return forces_sampled_all
        
### calculate energies ###
def calculate_energies(coords_all, elements_all):
    
    coords_sampled_all_list, elements_sampled_all_list = [],[]
    for j in range(len(coords_all)):
        for l in range(len(coords_all[j])):
            coords_sampled_all_list.append(coords_all[j][l])
            elements_sampled_all_list.append(elements_all[j][l])
    
    energies_sampled_all = []
    
    for coords, elements in zip(coords_sampled_all_list, elements_sampled_all_list):
        results = xtb_calc(coords, elements, opt=False, grad=False, hess=False, charge=0, freeze=[], dirname=None)
        
        e = results["energy"]
        energies_sampled_all.append(e)
        #print(forces.shape)
        #print(forces)
    return energies_sampled_all

### optimization ###
def optimize_geometry(coords, elements, mol_dir= None, charge = 0, unp_e = 0, freeze_atms = [], solvent = None):
    try:
        if mol_dir != None:
            results = xtb_calc(coords, elements, opt=True, grad=False, hess=False, chrg = charge, uhf = unp_e, freeze= freeze_atms, dirname=mol_dir+'/xtb_geo_opt', solve = solvent)
        else:
            results = xtb_calc(coords, elements, opt=True, grad=False, hess=False, chrg = charge, uhf = unp_e, freeze=[], solve = solvent)
        coords_new = results['coords']
        elements_new = results['elements']
        print('Geometry optimized.')
        return(coords_new, elements_new)
    except:
        coords_new = []
        return(coords_new)


### xtb calculations ###
def xtb_calc(coords, elements, opt=False, grad=False, hess=False, chrg=0, uhf = 0, freeze=[], dirname=None, solve = None):

    if opt and grad:
        exit("opt and grad are exclusive")
    if hess and grad:
        exit("hess and grad are exclusive")

    if hess or grad:
        if len(freeze)!=0:
            print("WARNING: please test the combination of hess/grad and freeze carefully")

    if dirname is None:
        rundir="xtb_tmpdir_%s"%(uuid.uuid4())
    else:
        rundir=dirname
        
    if not os.path.exists(rundir):
        os.makedirs(rundir)
        
    #else:
        #if len(os.listdir(rundir))>0:
            #os.system("rm %s/*"%(rundir))

    startdir=os.getcwd()
    os.chdir(rundir)
    
    fu.exportXYZ(coords, elements, "in.xyz")

    if len(freeze)>0:
        outfile=open("xcontrol","w")
        outfile.write("$fix\n")
        outfile.write(" atoms: ")
        for counter,i in enumerate(freeze):
            if (counter+1)<len(freeze):
                outfile.write("%i,"%(i+1))
            else:
                outfile.write("%i\n"%(i+1))
        #outfile.write("$gbsa\n solvent=toluene\n")
        outfile.close()
        add="-I xcontrol"
    else:
        add="" # new
    
    if solve != None:
        add_solve = '--alpb ' + solve
    else:
        add_solve = ''
    #method = '--gfn2'
    #print(os.popen('which xtb').read())

    if chrg==0 and uhf == 0:
        if opt:
            if hess:
                command = "xtb --gfn2 {} in.xyz --ohess vtight {}".format(add, add_solve)
            else:
                #print(command)
                command = "xtb --gfn2 {} in.xyz --opt vtight {}".format(add, add_solve) # vtight new
        else:
            if grad:
                command = "xtb --gfn2 {} in.xyz --grad {}".format(add,add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz {}".format(add,add_solve)

    if chrg!= 0 and uhf == 0:
        if opt:
            if hess:
                command = "xtb --gfn2 {} in.xyz --ohess vtight --chrg {} {}".format(add,chrg,add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz --opt vtight --chrg {} {}".format(add,chrg,add_solve)
        else:
            if grad:
                command = "xtb --gfn2 {} in.xyz --grad --chrg {} {}".format(add,chrg,add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz --chrg {} {}".format(add,chrg,add_solve)
    
    if chrg == 0 and uhf != 0:
        if opt:
            if hess:
                command = "xtb --gfn2 {} in.xyz --ohess vtight --uhf {} {}".format(add,uhf,add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz --opt vtight --uhf {} {}".format(add,uhf,add_solve)
        else:
            if grad:
                command = "xtb --gfn2 {} in.xyz --grad --uhf {} {}".format(add,uhf, add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz --uhf {} {}".format(add,uhf,add_solve)
    
    if chrg != 0 and uhf != 0:
        if opt:
            if hess:
                command = "xtb --gfn2 {} in.xyz --ohess vtight --chrg {} --uhf {} {}".format(add,chrg, uhf,add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz --opt vtight --chrg {} --uhf {} {}".format(add,chrg, uhf,add_solve)
        else:
            if grad:
                command = "xtb --gfn2 {} in.xyz --grad --chrg {} --uhf {} {}".format(add,chrg, uhf,add_solve)
            else:
                command = "xtb --gfn2 {} in.xyz --chrg {} --uhf {} {}".format(add,chrg,uhf,add_solve)


    #os.environ["OMP_NUM_THREADS"]="10" # "%s"%(settings["OMP_NUM_THREADS"])
    #os.environ["MKL_NUM_THREADS"]="10" # "%s"%(settings["MKL_NUM_THREADS"])


    args = shlex.split(command)

    mystdout = open("xtb.log", "a")
    process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
    out, err = process.communicate()
    mystdout.close()


    if opt:
        if not os.path.exists("xtbopt.xyz"):
            #raise ValueError('xtb geometry optimization did not work.') ##
            print("WARNING: xtb geometry optimization did not work")
            coords_new, elements_new = None, None
        else:
            coords_new, elements_new = fu.readXYZ("xtbopt.xyz")
    else:
        coords_new, elements_new = None, None

    if grad:
        grad = read_xtb_grad()
    else:
        grad = None

    if hess:
        hess, vibspectrum = read_xtb_hess()
    else:
        hess, vibspectrum = None, None

    e = read_xtb_energy()

    os.chdir(startdir)


    if dirname is None:
        os.system("rm -r %s"%(rundir))

    results={"energy": e, "coords": coords_new, "elements": elements_new, "gradient": grad, "hessian": hess, "vibspectrum": vibspectrum}
    return(results)

def read_xtb_energy():
    if not os.path.exists("xtb.log"):
        return(None)
    energy=None
    for line in open("xtb.log"):
        if "| TOTAL ENERGY" in line:
            energy = float(line.split()[3])*HToeV
    return(energy)


def read_xtb_grad():
    if not os.path.exists("gradient"):
        return(None)
    grad = []
    for line in open("gradient","r"):
        if len(line.split())==3:
            grad.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
    if len(grad)==0:
        grad=None
    else:
        grad = np.array(grad)*27.21138624598853/0.52917721090380# from H/B to eV/A
        grad = grad*(-1.0)
    return(grad) ###or return forces ? *(-1.0)


def read_xtb_hess():
    hess = None
    if not os.path.exists("hessian"):
        return(None, None)
    hess = []
    for line in open("hessian","r"):
        if "hess" not in line:
            for x in line.split():
                hess.append(float(x))
    if len(hess)==0:
        hess=None
    else:
        hess = np.array(hess)

    vibspectrum = None
    if not os.path.exists("vibspectrum"):
        return(None, None)
    vibspectrum = []
    read=False
    for line in open("vibspectrum","r"):
        if "end" in line:
            read=False

        if read:
            if len(line.split())==5:
                vibspectrum.append(float(line.split()[1]))
            elif len(line.split())==6:
                vibspectrum.append(float(line.split()[2]))
            else:
                print("WARNING: weird line length: %s"%(line))
        if "RAMAN" in line:
            read=True
        
    if len(vibspectrum)==0:
        vibspectrum=None
    else:
        vibspectrum = np.array(vibspectrum)

    return(hess, vibspectrum)

### crest ###
def call_crest(mol_init_coords, crest_fast, chrg = 0, conf_method = 'gff', reduce_output = True, solve = None, num_thr = 1):

    #os.environ["OMP_NUM_THREADS"]="%s"%(settings["OMP_NUM_THREADS"])
    #os.environ["MKL_NUM_THREADS"]="%s"%(settings["MKL_NUM_THREADS"])
    #command="crest %s --gbsa toluene -metac -nozs"%(filename)
    
    if solve != None:
        add_solve = '--alpb ' + solve
    else:
        add_solve = ''

    if num_thr != 1:
        add_threads = '-T {}'.format(num_thr)
    else:
        add_threads = ''
    
    if crest_fast == 'true':
        command = 'crest {} --{} {} {}'.format(mol_init_coords, conf_method, add_solve, add_threads) #--ewin 4 --squick --mquick --quick --gff

    else:
        if chrg == 0:
            command='crest {} --gfn2 {} {}'.format(mol_init_coords, add_solve,add_threads)
        else:
            command='crest {} --gfn2 --chrg {} {} {}'.format(mol_init_coords, chrg, add_solve, add_threads)

    
    #command="crest %s --gbsa toluene -metac"%(filename)
    #command="crest %s -ethr %f -pthi %f -metac"%(filename, settings["max_E"], settings["max_p"])
    # crest -chrg %i is used for charges
    
    args = shlex.split(command)
    mystdout = open("crest.log","a")

    process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
    out, err = process.communicate()
    mystdout.close()
    time.sleep(5)
    if reduce_output:
        for x in os.listdir("."):
            if x.startswith("M") or x.startswith("N") or x=="wbo" or x=="coord" or "_rotamers_" in x or ".tmp" in x or x.startswith(".") or x=="coord.original":
                if os.path.isdir(x):
                    try:
                        shutil.rmtree(x)
                    except:
                        pass
                elif os.path.isfile(x):
                    try:
                        os.remove(x)
                    except:
                        pass
    return()



def run_crest(coords, elements, moldir, mol_init_coords, crest_fast, charge, conf_method_cr, solvent = None, num_threads = 1):
   
    startdir=os.getcwd()
    #print(startdir)
    if startdir != '{}'.format(moldir):
        
        os.chdir('{}'.format(moldir))

    #exportXYZ(coords, elements, filename)
    #exit()
    #time1=time.time()
    if not os.path.exists("crest_best.xyz"):
        call_crest(mol_init_coords, crest_fast, chrg=charge, conf_method = conf_method_cr, solve = solvent, num_thr = num_threads) 
    else:
        print("   ---   found old crest run and read output")
        pass
    #crest_done, coords_all, elements_all, boltzmann_data = get_crest_results()
    coords_all, elements_all, energies = read_crest_conformers(len(elements))
    if len(elements_all)==0:
        exit("ERROR: No conformers found for %s"%(mol_init_coords))
        
    os.chdir(startdir)
    
    return(coords_all, elements_all, energies) #boltzmann_data


def read_crest_conformers(number_atoms): 
    infile=open('crest_conformers.xyz',"r")
    coords=[[]]
    elements=[[]]    
    energies = []
    for line in infile.readlines():
        if len(line.split())==1 and len(coords[-1])!=0 and float(line.split()[0]) == number_atoms:
            coords.append([])
            elements.append([])
        elif len(line.split())==1 and float(line.split()[0]) != number_atoms:
            energies.append(float(line.split()[0])*HToeV)
        elif len(line.split())==4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    infile.close()
    return coords,elements, energies