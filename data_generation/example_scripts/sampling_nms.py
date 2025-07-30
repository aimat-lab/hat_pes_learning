#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import yaml
import os

from mol_data_generation.mol_construction.construction_module import ConstructInitData
from mol_data_generation.nms.nms_module import NormalModeSampling
from mol_data_generation.radical_systems.radical_module import RadicalSampling
from mol_data_generation.reactions.hat_module import HATSampling

'''
Example script how to use data generation methods

You can start from scratch by either providing or generating a SMILES dictionary of your systems
or you can use methods seperately.

'''

### initial data generation ###

# load config file

with open("config_nms.yaml", "r") as f:
    settings_construction = yaml.load(f, Loader=yaml.FullLoader)


# specify output location

sample_name = 'aa_test_nms'

# local
outdir = 'samples/{}'.format(sample_name)

'''
cluster:
tmp_path = os.getenv('TMP')

outdir = '{}/{}'.format(tmp_path, sample_name)
'''
if not os.path.exists(outdir):
    os.makedirs(outdir)



# smiles dictionary
# provide dictionary with {molecule_type:{molecule_name: SMILES, ...}}
dict_smiles_init = {'aa':{'Alanine':'CC(C(=O)O)N', 'Arginine':'C(CC(C(=O)O)N)CN=C(N)N'}}

# alternatively you can generate SMILES dictionary using methods provided in the data construction class

# solvent
if settings_construction['solvent'] == 'None':
    solve = None
else:
    solve = settings_construction['solvent']


# set number of threads for conformer generation on cluster
set_thread = settings_construction['num_threads']

# initialize construction class
init_data = ConstructInitData(outdir, dict_smiles= dict_smiles_init, solvent = solve, num_threads = set_thread) 



## coordinate generation ##

# generated coords from smiles dictionary and save 
# optimize True: optimize generated coordinates using xtb
init_data.coord_generation(overwrite = False, optimize = True)


## conformer generation ##

settings_conf = settings_construction['conformers']

init_data.conf_generation(settings_conf)  



### normal mode sampling ###

settings_nms = settings_construction['nms']

# indir of normal mode sampling is the outdir from conformer generation 
indir = '{}'.format(outdir)

normal_mode_sampling = NormalModeSampling(indir, sample_name,settings_nms['num_samples'], settings_nms['delta_E_max'], settings_nms['temperature'], settings_nms['ci_range'], solve)

# do nms
normal_mode_sampling.do_nms()

# optional: concatenate all data
#normal_mode_sampling.concat_all_data()


## plot bond length distribution
normal_mode_sampling.plot_bond_lengths()
# scale and plot energies
normal_mode_sampling.scale_and_plot_energies()
# plots for analysis
normal_mode_sampling.plot_analysis()