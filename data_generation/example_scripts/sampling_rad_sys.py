import os
from mol_data_generation.radical_systems.radical_module import RadicalSampling

# Settings
sample_name = 'aa_test_nms'
indir_list = ['samples/{}/'.format(sample_name)]  # list of NMS directories
init_indir_list = indir_list
indir_info = indir_list[0]  # same for this example
solvent = None
temperature = 300
delta_E_max = 0.25

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

# Generate radicals from NMS samples
rad_gen.do_intra_rad_sampling(
    sample_name=sample_name,
    num_systems=0,           # use all found NMS systems
    num_config=1,            # one radical config per input
    outdir=outdir,
    mode='H_radical',
    radical_opt=False
)
