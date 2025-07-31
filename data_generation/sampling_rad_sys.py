import os
from mol_data_generation.radical_systems.radical_module import RadicalSampling

# Settings
sample_name = 'aa_test_nms'
indir_list = ['samples/{}/'.format(sample_name)]  # list of NMS directories
init_indir_list = indir_list
indir_info = indir_list[0]  # same for this example
solvent = None
temperature = 300
delta_E_max = 5 # eV maximum energy difference of generated configurations wrt the lowest energy configuration

# Output directory
outdir = 'samples/{}_radicals'.format(sample_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Initialize radical generator
rad_gen = RadicalSampling(
    insystem_list=None,
    indir_list=indir_list,
    indir_info=indir_info,
    init_indir_list=init_indir_list,
    solvent=solvent,
    t_nms=temperature,
    delE_nms=delta_E_max,
    protonated=False
)

# Intra-molecular radical sampling 
rad_gen.do_intra_rad_sampling(
    sample_name=sample_name,
    num_systems=0,           # Use all available NMS systems
    num_config=1,            # One radical config per system
    outdir=outdir,
    mode='H_radical',
    radical_opt=False        # No post-H-removal optimization
)

#  Inter-molecular radical sampling
rad_gen.do_inter_rad_sampling_V2(
    sample_name=sample_name,
    num_systems=0,           # All possible pairs (can set >0 for a subset)
    num_config=1,            # One sampled config per pair
    outdir=outdir,
    mode='H_radical',
    rad_radius=[2.0, 4.2]    # Default separation range of molecules in Angstroms
)
