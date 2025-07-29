#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os
import argparse
import keras as ks
#import kgcnn.training.schedule
#from kgcnn.data.transform.scaler.serial import deserialize as deserialize_scaler
from kgcnn.utils.devices import check_device, set_cuda_device
from kgcnn.models.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training.hyper import HyperParameter
from kgcnn.data import MemoryGraphList
import matplotlib.pyplot as plt
import json
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler
from kgcnn.molecule.dynamics.base import MolDynamicsModelPredictor
from kgcnn.graph.postprocessor import ExtensiveEnergyForceScalerPostprocessor
from kgcnn.graph.preprocessor import SetRange, CountNodesAndEdges
#from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
import pandas as pd
import file_utils as fu
#from ase.io import read

from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error

from collections import defaultdict


## utils
global_proton_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                      'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                      'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                      'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
                      'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47,
                      'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                      'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
                      'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
                      'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
                      'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
                      'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                      'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
                      'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
                      'Og': 118, 'Uue': 119}

def read_csv(filepath):
    
    df = pd.read_csv(filepath)
    
    id_init = df['ID'].tolist()
    system_names = df['system_name'].tolist()
    h0_dist = df['h0_dist'].tolist()
 
    return id_init, system_names, h0_dist

def parse_extxyz_steps(filename):
    data = []
    with open(filename, 'r') as f:
        entry_count = 0
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            
            prop_line = f.readline().strip()
            properties = {}
            for item in prop_line.split():
                if '=' in item:
                    key, value = item.split('=', 1)
                    properties[key] = value
            
            # Read atom data
            elements = []
            coordinates = []
            forces = []
            for _ in range(n_atoms):
                atom_line = f.readline().split()
                elements.append(atom_line[0])
                coordinates.append([float(x) for x in atom_line[1:4]])
                forces.append([float(x) for x in atom_line[4:7]])
            
            try:
                unique_id = properties['unique_id']
                system_id, step = unique_id.split('_')
                
                entry = {
                    'system_id': system_id,
                    'step': int(step),
                    'energy': float(properties['energy']),
                    'h0_dist': float(properties['h0_dist']),
                    'n_atoms': n_atoms,
                    'elements': elements,  # Now storing elements
                    'coordinates': coordinates,  # And coordinates
                    'forces': forces
                }
                data.append(entry)
                entry_count += 1
                
            except Exception as e:
                print(f"Error parsing entry {entry_count}: {e}")
                print(f"Properties line: {prop_line}")
                continue
    return data

def parse_extxyz(file_path):
    data = {
        "energy": [],
        "unique_id": [],
        "mol_name": [],
        "h0_dist": [],
        "coordinates": [],
        "forces": [],
        "elements": [],
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # Read the number of atoms
        num_atoms = int(lines[i].strip())
        i += 1

        # Read the properties line
        properties_line = lines[i].strip()
        i += 1

        # Extract energy, unique_id, mol_name, and h0_dist
        energy = None
        unique_id = None
        mol_name = None
        h0_dist = None

        for prop in properties_line.split():
            key, value = prop.split("=", 1)
            if key == "energy":
                energy = float(value)
            elif key == "unique_id":
                unique_id = value
            elif key == "mol_name":
                mol_name = value
            elif key == "h0_dist":
                h0_dist = float(value)

        data["energy"].append(energy)
        data["unique_id"].append(unique_id)
        data["mol_name"].append(mol_name)
        data["h0_dist"].append(h0_dist)

        # Read atom data (coordinates, forces, and element types)
        coordinates = []
        forces = []
        element_types = []

        for _ in range(num_atoms):
            atom_line = lines[i].strip().split()
            i += 1

            # Extract element type (column 0), position (columns 1-3), and forces (columns 4-6)
            element_type = atom_line[0]
            pos = list(map(float, atom_line[1:4]))
            force = list(map(float, atom_line[4:7]))

            element_types.append(element_type)
            coordinates.append(pos)
            forces.append(force)

        data["coordinates"].append(coordinates)
        data["forces"].append(forces)
        data["elements"].append(element_types)

    return data

## metrics
def process_force_data(true_force_list, pred_force_list, label):
    """Process force data and compute metrics."""
    # Flatten forces to (N_total_atoms, 3)
    f_true = np.vstack([np.array(config) for config in true_force_list])
    f_pred = np.vstack([np.array(config) for config in pred_force_list])
    
    # Compute metrics
    metrics = {
        'total_mae': mean_absolute_error(f_true, f_pred),
        'component_mae': [
            mean_absolute_error(f_true[:, 0], f_pred[:, 0]),
            mean_absolute_error(f_true[:, 1], f_pred[:, 1]),
            mean_absolute_error(f_true[:, 2], f_pred[:, 2])
        ],
        'std_true': [f_true[:, 0].std(), f_true[:, 1].std(), f_true[:, 2].std()],
        'std_pred': [f_pred[:, 0].std(), f_pred[:, 1].std(), f_pred[:, 2].std()]
    }
    
    metrics['relative_mae'] = [
        metrics['component_mae'][i] / metrics['std_true'][i] 
        for i in range(3)
    ]
    
    print(f"\n{label} Force Analysis:")
    print(f"Total MAE: {metrics['total_mae']:.4f} eV/Å")
    for i, comp in enumerate(['X', 'Y', 'Z']):
        print(f"{comp}-MAE: {metrics['component_mae'][i]:.4f} eV/Å | "
              f"Normalized {comp}-MAE: {metrics['relative_mae'][i]:.4f} "
              f"(σ={metrics['std_true'][i]:.2f} eV/Å)")

    return f_true, f_pred, metrics

def relative_mae(true_values, pred_values):
    """Compute the Relative Mean Absolute Error (RMAE)."""
    # relative mean absolute error (RMAE) for energies
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)
    
    numerator = np.abs(pred_values - true_values).sum()
    denominator = np.abs(true_values).sum()
    
    return numerator / denominator if denominator != 0 else np.nan  # Avoid division by zero


def normalized_mae(true_values, pred_values):
    # This metric tells you how large the absolute error is relative to the spread of the data
    # This normalization helps compare models across datasets with different energy distributions
    """Compute MAE normalized by the standard deviation of true values."""
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)
    
    mae = np.mean(np.abs(pred_values - true_values))
    std = np.std(true_values)
    
    return mae / std if std != 0 else np.nan

def relative_force_error(true_forces, pred_forces):
    """Compute the Relative Mean Absolute Error (RMAE) for forces."""
    total_error = 0.0
    total_norm = 0.0
    force_errors = []

    for true_f, pred_f in zip(true_forces, pred_forces):
        true_f = np.array(true_f)  # Shape (N_atoms, 3)
        pred_f = np.array(pred_f)  # Shape (N_atoms, 3)
        
        error_per_atom = np.linalg.norm(pred_f - true_f, axis=1)
        norm_per_atom = np.linalg.norm(true_f, axis=1)
        
        force_errors.extend(error_per_atom)  # Collect per-atom errors
        
        total_error += error_per_atom.sum()
        total_norm += norm_per_atom.sum()

    rmae_f = total_error / total_norm if total_norm != 0 else np.nan
    return rmae_f, force_errors

def normalized_mae_forces(true_forces, pred_forces):
    """Compute the normalized MAE for forces."""
    force_errors = []
    true_force_magnitudes = []

    for true_f, pred_f in zip(true_forces, pred_forces):
        true_f = np.array(true_f)  # Shape (N_atoms, 3)
        pred_f = np.array(pred_f)  # Shape (N_atoms, 3)
        
        error_per_atom = np.linalg.norm(pred_f - true_f, axis=1)  # MAE per atom
        norm_per_atom = np.linalg.norm(true_f, axis=1)  # Magnitude of true forces
        
        force_errors.extend(error_per_atom)
        true_force_magnitudes.extend(norm_per_atom)

    mae_force = np.mean(force_errors)  # Mean absolute error for forces
    std_force = np.std(true_force_magnitudes)  # Standard deviation of true force magnitudes
    
    return mae_force / std_force if std_force != 0 else np.nan


## barriers
def calculate_barriers(data):
    systems = defaultdict(list)
    for entry in data:
        systems[entry['system_id']].append(entry)
    
    results = {}
    for system_id, entries in systems.items():
        try:
            sorted_entries = sorted(entries, key=lambda x: x['step'])
            h0_dists = [e['h0_dist'] for e in sorted_entries]
            
            #print(f"\nSystem {system_id}:")
            #print(f"Steps found: {[e['step'] for e in sorted_entries]}")
            #print(f"h0_dists: {h0_dists}")
            energies = [e['energy'] for e in sorted_entries]
            max_energy = max(energies)
            max_index = energies.index(max_energy)
            steps = [e['step'] for e in sorted_entries]
            
            results[system_id] = {
                'left_barrier': max([e['energy'] for e in sorted_entries]) - sorted_entries[0]['energy'],
                'right_barrier': max([e['energy'] for e in sorted_entries]) - sorted_entries[-1]['energy'],
                'transition_state_step': steps[max_index],
                'final_h0_dist': h0_dists[-1],
                'n_atoms': sorted_entries[0]['n_atoms'],
                'num_steps': len(sorted_entries)
            }
            
        except Exception as e:
            print(f"Error processing system {system_id}: {e}")
            continue
            
    return results



## plots
def plot_predict_true(y_predict, y_true, data_unit: list = None, model_name: str = "",
                      filepath: str = None, file_name: str = "", dataset_name: str = "", target_names: list = None,
                      figsize: list = None, dpi: float = None, show_fig: bool = False,
                      scaled_predictions: bool = False):
    r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.

    Args:
        y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        data_unit (list): String or list of string that matches `n_targets`. Name of the data's unit.
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        target_names (list): String or list of string that matches `n_targets`. Name of the targets.
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
        scaled_predictions (bool): Whether predictions had been standardized. Default is False.

    Returns:
        matplotlib.pyplot.figure: Figure of the scatter plot.
    """
    if len(y_predict.shape) == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    if data_unit is None:
        data_unit = ""
    if isinstance(data_unit, str):
        data_unit = [data_unit]*num_targets
    if len(data_unit) != num_targets:
        print("WARNING:kgcnn: Targets do not match units for plot.")
    if target_names is None:
        target_names = ""
    if isinstance(target_names, str):
        target_names = [target_names]*num_targets
    if len(target_names) != num_targets:
        print("WARNING:kgcnn: Targets do not match names for plot.")

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(num_targets):
        delta_valid = y_true[:, i] - y_predict[:, i]
        mae_valid = np.mean(np.abs(delta_valid[~np.isnan(delta_valid)]))
        plt.scatter(y_predict[:, i], y_true[:, i], alpha=0.3,
                    label=target_names[i] + " MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit[i] + "]")
    min_max = np.amin(y_true[~np.isnan(y_true)]).astype("float"), np.amax(y_true[~np.isnan(y_true)]).astype("float")
    plt.plot(np.arange(*min_max, 0.05), np.arange(*min_max, 0.05), color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plot_title = "Prediction of %s for %s " % (model_name, dataset_name)
    if scaled_predictions:
        plot_title = "(SCALED!) " + plot_title
    plt.title(plot_title)
    plt.legend(loc='upper left', fontsize='x-large')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig

def plot_barrier_parity(true_values, pred_values, dataset_name, barrier_type, outdir):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, pred_values, alpha=0.6)
    plt.plot([min(true_values), max(true_values)], 
             [min(true_values), max(true_values)], 'r--')
    plt.xlabel(f'True {barrier_type} Barrier (eV)')
    plt.ylabel(f'Predicted {barrier_type} Barrier (eV)')
    plt.title(f'{dataset_name} {barrier_type} Barrier Parity')
    plt.grid(True)
    plt.savefig(f'{outdir}/{dataset_name}_{barrier_type}_parity.png', dpi = 300)
    plt.close()

def plot_force_distributions(f_true, f_pred, label, outdir):
    """Plot force component distributions and errors."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    components = ['X', 'Y', 'Z']
    
    for i in range(3):
        # True vs Predicted distribution
        axs[0, i].hist(f_true[:, i], bins=50, alpha=0.5, label='True')
        axs[0, i].hist(f_pred[:, i], bins=50, alpha=0.5, label='Predicted')
        axs[0, i].set_title(f'{label} {components[i]} Component Distribution')
        axs[0, i].set_xlabel('Force (eV/Å)')
        axs[0, i].legend()

        # Error distribution
        errors = f_pred[:, i] - f_true[:, i]
        axs[1, i].hist(errors, bins=50, color='red', alpha=0.7)
        axs[1, i].set_title(f'{label} {components[i]} Error Distribution')
        axs[1, i].set_xlabel('Predicted - True (eV/Å)')

    plt.tight_layout()
    plt.savefig(f'{outdir}/{label}_force_distributions.png', dpi=300)
    plt.close()
    
def plot_predictions(y_true, y_pred, title, filename, xlabel='True Value', ylabel='Predicted Value'):
    """Generate parity plot with metrics"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title}\nMAE: {mean_absolute_error(y_true, y_pred):.3f} | R²: {r2_score(y_true, y_pred):.3f}')
    plt.grid(True)
    plt.savefig(filename, dpi = 300)
    plt.close()

def plot_system_energy_curves(system_id, h0_dists, true_energies, pred_energies, dataset_name, outdir):
    """Plot energy vs H0 distance for a single system"""
    plt.figure(figsize=(10, 6))
    plt.plot(h0_dists, true_energies, 'b-o', label='True Energy')
    plt.plot(h0_dists, pred_energies, 'r--s', label='Predicted Energy')
    plt.xlabel('H0 Distance (Å)')
    plt.ylabel('Energy (eV)')
    plt.title(f'System {system_id} Energy Profile ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{outdir}/{dataset_name}_system_{system_id}_energy_profile.png', dpi=300)
    plt.close()

def plot_error_distribution(errors, title, filename, xlabel='Error'):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

### LAOD HYPER SCHNET ###
parser = argparse.ArgumentParser(description='Train a GNN on an Energy-Force Dataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper/hyper_local_tests.py") #  local_tests
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
parser.add_argument("--category", required=False, help="Graph model to train.", default="Schnet.EnergyForceModel")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--module", required=False, help="Name of the module for model.", default=None)
parser.add_argument("--fold", required=False, help="Split or fold indices to run.", default=None, nargs="+", type=int)
parser.add_argument("--seed", required=False, help="Set random seed.", default=42, type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Check and set device
if args["gpu"] is not None:
    set_cuda_device(args["gpu"])
print(check_device())

# Set seed.
np.random.seed(args["seed"])
ks.utils.set_random_seed(args["seed"])

# HyperParameter is used to store and verify hyperparameter.
hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"], model_module=args["module"])
hyper.verify()

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
dataset = deserialize_dataset(hyper["dataset"])

# Check if dataset has the required properties for model input. This includes a quick shape comparison.

dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. 
dataset.clean(hyper["model"]["config"]["inputs"])
data_length = len(dataset)  # Length of the cleaned dataset.

# Always train on `energy` .

label_names, label_units = dataset.set_multi_target_labels(
    "energy",
    hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper["training"] else None,
    data_unit=hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else None
)

filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

### LOAD ###
indir_data = 'my_data/dft_data' # dft_IDs

name_data_train = 'dft_ID2'    

## DATA ##
# TEST
name_data_test = 'dft_test'
data_test_all = parse_extxyz('{}/{}/{}_full.extxyz'.format(indir_data,name_data_test, name_data_test ))

# EVAL
name_data_eval_dir = 'dft_eval'
name_data_eval = 'dft_eval'

data_eval_all = parse_extxyz('{}/{}/{}_full.extxyz'.format(indir_data, name_data_eval_dir, name_data_eval))

# LIN ID10 TEST
name_data_lin_ID10_dir = 'dft_lin_ID10'
name_data_lin_ID10 = 'dft_lin_ID10'

indir_lin_id10 = '{}/{}'.format(indir_data, name_data_lin_ID10_dir) 

data_lin_id10 = parse_extxyz_steps('{}/{}_full.extxyz'.format(indir_lin_id10, name_data_lin_ID10))
barriers_lin_id10 = calculate_barriers(data_lin_id10)

# LIN EVAL
name_data_lin_eval_dir = 'dft_lin_eval'
name_data_lin_eval = 'dft_lin_eval'

data_lin_eval = parse_extxyz_steps('{}/{}/{}_full.extxyz'.format(indir_data, name_data_lin_eval_dir, name_data_lin_eval))
barriers_lin_eval = calculate_barriers(data_lin_eval)

## LOAD MODEL ##
inpath_model = 'results/ForceDataset/Schnet_EnergyForceModel{}'.format(name_data_train)

max_train = 65514
len_total = max_train + 7291
train_index = list(range(0,max_train))              # range(0, max_train) max_train = 1114
test_index = list(range(max_train,len_total))   # range(max_train+1, -1)

model = deserialize_model(hyper["model"])

model.load_weights('{}/model{}_fold_1.keras'.format(filepath, name_data_train))

model_config = model.get_config()

# load scaler
scaler = EnergyForceExtensiveLabelScaler(standardize_scale=False)
scaler.load('{}/scaler{}_fold_1.json'.format(filepath,name_data_train))

# prep dyn 
dyn_model = MolDynamicsModelPredictor(
    model=model, 
    use_predict = True,
    model_inputs=model_config["model_energy"]["config"]["inputs"], 
    model_outputs={"energy":"energy", "forces": "force"}, 
    graph_preprocessors=[
        SetRange(node_coordinates="node_coordinates", max_distance=5.0, max_neighbours=10000),
        CountNodesAndEdges(total_edges="total_ranges", count_edges="range_indices", count_nodes="atomic_number", total_nodes="total_nodes")
    ],
    graph_postprocessors=[
        ExtensiveEnergyForceScalerPostprocessor(
            scaler, force="forces", atomic_number="atomic_number")]
)

# prepare output forces
eval_name = 'dft_ID2' 
outdir = 'schnet_eval/{}'.format(eval_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)

### sanity check ###
dataset_hat_test = dataset[test_index]
dataset_hat_test = scaler.transform_dataset(dataset_hat_test, copy_dataset=True, copy=True)

x_hat_test = dataset_hat_test.tensor(hyper["model"]["config"]["inputs"])
y_hat_test = dataset_hat_test.tensor(hyper["model"]["config"]["outputs"]) 

predicted_hat_y = model.predict(x_hat_test, verbose=0)

rescaled_predicted_hat_y = scaler.inverse_transform(y=[predicted_hat_y["energy"],predicted_hat_y["force"]], X = dataset_hat_test.get('atomic_number'))

rescaled_true_hat_y = scaler.inverse_transform(y=[y_hat_test["energy"],y_hat_test["force"]], X = dataset_hat_test.get('atomic_number'))

# get errors E and F , E_err/ system_size 
atom_num_padded_hat_test = x_hat_test[0]
system_sizes_hat_test = []

for i in range(len(atom_num_padded_hat_test)):
    atom_num_true = []
    for atom_num_i in atom_num_padded_hat_test[i]:
        if atom_num_i != 0:
            atom_num_true.append(atom_num_i)
        else:
            continue

    system_sizes_hat_test.append(len(atom_num_true))

pred_forces_san = []
true_forces_san = []

for i in range(len(rescaled_predicted_hat_y)):
    forces_i = rescaled_predicted_hat_y[1][i]
    forces_i_all = []
    for j in range(system_sizes_hat_test[i]):
        forces_i_all.append(forces_i[j])
    pred_forces_san.append(forces_i_all)

for i in range(len(rescaled_true_hat_y)):
    forces_i = rescaled_true_hat_y[1][i]
    forces_i_all = []
    for j in range(system_sizes_hat_test[i]):
        forces_i_all.append(forces_i[j])
    true_forces_san.append(forces_i_all)


mae_en_hat_test = mean_absolute_error(rescaled_true_hat_y[0], rescaled_predicted_hat_y[0])
r2_en_hat_test = r2_score(rescaled_true_hat_y[0], rescaled_predicted_hat_y[0])

mae_en_sys_hat_test = mean_absolute_error(np.array(rescaled_true_hat_y[0])/np.array(system_sizes_hat_test), np.array(rescaled_predicted_hat_y[0])/np.array(system_sizes_hat_test))

f_true_san, f_pred_san, metrics_san = process_force_data(true_forces_san, pred_forces_san, 'sanity_check')

print('SANITY CHECK on test data')
print('mae e', mae_en_hat_test)
print('r2 e', r2_en_hat_test)
print('mae/sys', mae_en_sys_hat_test)

### continue ###

## evaluation  functions##
def evaluate_barriers(model, dataset_name, data_lin, barriers_true, scaler, outdir):
    """Evaluate barrier predictions matching your working barrier calculation approach"""
    # 1. Prepare MemoryGraphList manually 
    graph_list = []
    #system_sizes = []
    system_map = defaultdict(list)
    
    for idx, entry in enumerate(data_lin):
        # Convert elements to atomic numbers (no padding!)
        atomic_numbers = [global_proton_dict[atom] for atom in entry['elements']]
        #system_sizes.append(len(atomic_numbers))  
        
        graph_list.append({
            'atomic_number': np.array(atomic_numbers, dtype=np.int32),
            'node_coordinates': np.array(entry['coordinates'], dtype=np.float32)
        })
        system_map[entry['system_id']].append(idx) 
    
    data = MemoryGraphList(graph_list)
    
    # 2. Predict using dyn_model
    print(f"Predicting energies for {dataset_name}...")
    predictions = dyn_model(data)
    
    pred_energies = [p['energy'][0] for p in predictions]
    if len(pred_energies) != len(data_lin):
        raise ValueError(f"Prediction count mismatch: {len(pred_energies)} vs {len(data_lin)}")

   
    #print('Len pred energies', len(pred_energies))
    #print('Len true energies', len(data_lin['energy']))

    # 3. Calculate predicted barriers
    barriers_pred = {}
    system_ids = list(barriers_true.keys())
    
    #print('len sys ids', len(system_ids))
    #print(system_ids[0])
    
    system_data = {}
    
    for sys_id in system_ids:
        sys_indices = system_map.get(sys_id, [])
        if not sys_indices:
            print(f"No predictions for system {sys_id}")
            continue
        # Get all entries for this system
        #sys_entries = [entry for entry in data_lin if entry['system_id'] == sys_id]
        #sorted_entries = sorted(sys_entries, key=lambda x: x['step'])
        sys_entries = [data_lin[i] for i in sys_indices]
        sorted_entries = sorted(sys_entries, key=lambda x: x['step'])
        
        # Get corresponding predicted energies
        #energy_indices = [i for i, entry in enumerate(data_lin) if entry['system_id'] == sys_id]
        
        #energy_indices = system_map.get(sys_id, [])
        step_pred_pairs = sorted(
            zip([e['step'] for e in sys_entries],  # Unsorted steps
            [pred_energies[i] for i in sys_indices]  # Predictions in original order
        ), key=lambda x: x[0])  # Sort by step
        
        sorted_pred_energies = [p for (step, p) in step_pred_pairs]

        #rint('len en indices', len(energy_indices))

        #sys_energies = [pred_energies[i] for i in energy_indices]
        
        #sorted_pairs = sorted(zip(sorted_entries, sys_energies), 
        #               key=lambda x: x[0]['step'])
        #sorted_entries, sorted_energies = zip(*sorted_pairs)
        #
        # Sort by step (double-check alignment)
        #sorted_energies = [e for _, e in sorted(zip([e['step'] for e in sorted_entries], sys_energies))]
        
        # Calculate barriers
        max_energy = max(sorted_pred_energies)
        barriers_pred[sys_id] = {
            'left_barrier': max_energy - sorted_pred_energies[0],
            'right_barrier': max_energy - sorted_pred_energies[-1]
        }
        
        h0_dists = [e['h0_dist'] for e in sorted_entries]
        true_energies = [e['energy'] for e in sorted_entries]
        #pred_energies = [pred_energies[i] for i in energy_indices]
        
        system_data[sys_id] = {
            'h0_dists': h0_dists,
            'true_energies': true_energies,
            'pred_energies': sorted_pred_energies
        }
    
    
    # plot individual systems
    for sys_id, data in system_data.items():
        plot_system_energy_curves(
            sys_id, data['h0_dists'], 
            data['true_energies'], data['pred_energies'],
            dataset_name, outdir
        )
    

    # 5. Calculate metrics (unchanged)
    left_true = [barriers_true[sys_id]['left_barrier'] for sys_id in system_ids]
    left_pred = [barriers_pred[sys_id]['left_barrier'] for sys_id in system_ids]
    
    right_true = [barriers_true[sys_id]['right_barrier'] for sys_id in system_ids]
    right_pred = [barriers_pred[sys_id]['right_barrier'] for sys_id in system_ids]
    
    plot_barrier_parity(left_true, left_pred, dataset_name, 'left', outdir)
    plot_barrier_parity(right_true, right_pred, dataset_name, 'right', outdir)
    plot_barrier_parity(left_true + right_true, left_pred + right_pred, dataset_name, 'all', outdir)
    
    left_errors = [a - b for a, b in zip(left_pred, left_true)]
    right_errors = [a - b for a, b in zip(right_pred, right_true)]
    
    plot_error_distribution(left_errors, 
                       f'{dataset_name} Left Barrier Errors',
                       f'{outdir}/{dataset_name}_left_barrier_errors.png',
                       xlabel='Predicted - True Barrier (eV)')

    plot_error_distribution(right_errors,
                       f'{dataset_name} Right Barrier Errors',
                       f'{outdir}/{dataset_name}_right_barrier_errors.png',
                       xlabel='Predicted - True Barrier (eV)')
    
    plot_error_distribution(left_errors+right_errors,
                       f'{dataset_name} Barrier Errors',
                       f'{outdir}/{dataset_name}_barrier_errors_all.png',
                       xlabel='Predicted - True Barrier (eV)')
    
    metrics = {
        'left_mae': mean_absolute_error(left_true, left_pred),
        'left_r2': r2_score(left_true, left_pred),
        'right_mae': mean_absolute_error(right_true, right_pred),
        'right_r2': r2_score(right_true, right_pred),
        'total_mae': mean_absolute_error(left_true + right_true, left_pred + right_pred),
        'total_r2': r2_score(left_true + right_true, left_pred + right_pred)
    }
    
    metrics_barriers = {}
    for sys_id in system_ids:
        metrics_barriers[sys_id] = {
            'true_left': barriers_true[sys_id]['left_barrier'],
            'pred_left': barriers_pred[sys_id]['left_barrier'],
            'true_right': barriers_true[sys_id]['right_barrier'],
            'pred_right': barriers_pred[sys_id]['right_barrier']
        }
    
    return metrics, barriers_pred, metrics_barriers

def evaluate_dataset(model, elements, coords, energies_true, forces_true, dataset_name, outdir):
    """Evaluate model on a complete dataset with energies and forces"""
    
    graph_list = []
    for mol_elements, mol_coords in zip(elements, coords):
        atomic_numbers = [global_proton_dict[atom] for atom in mol_elements]
        graph_list.append({
            'atomic_number': np.array(atomic_numbers, dtype=np.int32),
            'node_coordinates': np.array(mol_coords, dtype=np.float32)
        })
    
    data = MemoryGraphList(graph_list)
    
    # 2. Predict using dyn_model (no padding!)
    predictions = dyn_model(data)
    
    # 3. Extract energies and forces using working snippet's method
    pred_energies = [p['energy'][0] for p in predictions]
    
    
    system_sizes = [len(g['atomic_number']) for g in graph_list]  # From input data!
    
    
    pred_forces = []
    for i, p in enumerate(predictions):
        # Get true number of atoms for this molecule
        n_atoms = len(elements[i])  # elements[i] has actual atoms without padding
        # Trim predicted forces to match true atoms
        trimmed_forces = p['forces'][:n_atoms]  # Shape (n_atoms, 3) forces
        pred_forces.append(trimmed_forces) #
    
    # Now process forces (they'll match true_force_list shapes)
    f_true = np.vstack([np.array(config) for config in forces_true])
    f_pred = np.vstack(pred_forces)  # No padding
    
    assert f_true.shape == f_pred.shape, \
        f"Force shape mismatch after trimming: {f_true.shape} vs {f_pred.shape}"

    f_true_flat = f_true.flatten()  # Shape (N_total_atoms*3,)
    f_pred_flat = f_pred.flatten()

    # True values
    true_energies = np.array(energies_true)
    #true_forces = np.concatenate(forces_true)
    
    #f_true_flat, f_pred_flat, metrics = process_force_data(forces_true, pred_forces, dataset_name)      

    RMAE_e = relative_mae(energies_true, pred_energies)
    RMAE_f, f_err_list = relative_force_error(forces_true, pred_forces)
    
    mae_e_norm = normalized_mae(energies_true, pred_energies)

    mae_f_norm = normalized_mae_forces(forces_true, pred_forces)

    true_energies_array = np.array(true_energies)
    pred_energies_array = np.array(pred_energies)
    system_sizes_array = np.array(system_sizes)

    # Calculate MAE per atom
    mae_per_atom = mean_absolute_error(
    true_energies_array / system_sizes_array,
    pred_energies_array / system_sizes_array)
    
       
    # Energy metrics
    energy_metrics = {
        'mae': mean_absolute_error(true_energies, pred_energies),
        'r2': r2_score(true_energies, pred_energies),
        'mae_per_atom': mae_per_atom,
        "rmae": RMAE_e,
        'rmae_perc': RMAE_e*100,
        'mae_norm': mae_e_norm
    }
    
    # Force metrics
    force_metrics = {
        'mae': mean_absolute_error(f_true_flat, f_pred_flat),
        'r2': r2_score(f_true_flat, f_pred_flat),
        'mae_components': {
            'x': mean_absolute_error(f_true[:,0], f_pred[:,0]),
            'y': mean_absolute_error(f_true[:,1], f_pred[:,1]),
            'z': mean_absolute_error(f_true[:,2], f_pred[:,2])
        },
        'rmae': RMAE_f,
        'rmae_perc': RMAE_f*100,
        'mae_norm': mae_f_norm
    }
    
    # Plotting
    plot_predictions(
        true_energies, pred_energies, 
        title=f'{dataset_name} Energy Parity', 
        filename=f'{outdir}/{dataset_name}_energy_parity.png'
    )
    
    plot_predictions(
        f_true_flat, f_pred_flat,
        title=f'{dataset_name} Force Parity', 
        filename=f'{outdir}/{dataset_name}_force_parity.png',
        xlabel='True Force Component (eV/Å)',
        ylabel='Predicted Force Component (eV/Å)'
    )
    
    # plot predict true
    plot_predict_true(
        y_predict=np.array(pred_energies), 
        y_true=np.array(true_energies),
        target_names=["Energy"],
        data_unit=["eV"],
        model_name='SchNet',
        dataset_name=dataset_name,
        filepath = outdir,
        figsize=[10, 6]
    )
    
    plot_force_distributions(f_true, f_pred, dataset_name, outdir)

    f_pred_reshaped = np.array(f_pred_flat).reshape(-1, 3)
    f_true_reshaped = np.array(f_true_flat).reshape(-1, 3)
    
    pred_forces_array = np.concatenate(pred_forces)  # Shape (N_total_atoms, 3)
    true_forces_array = np.concatenate([np.array(f_mol) for f_mol in forces_true])  # Shape (N_total_atoms, 3)

    
    plot_predict_true(
        y_predict=f_pred_reshaped, 
        y_true=f_true_reshaped,
        target_names=["Fx", "Fy", "Fz"],
        data_unit=["eV/Å", "eV/Å", "eV/Å"],
        model_name='SchNet',
        dataset_name=dataset_name,
        filepath = outdir,
        figsize=[10, 6]
    )
    
    pred_metrics = {
            'energy_metrics': {
                'mae': float(energy_metrics['mae']),
                'r2': float(energy_metrics['r2']),
                'mae_per_atom': float(energy_metrics['mae_per_atom']),
                'rmae': float(energy_metrics['rmae']),
                'rmae_perc': float(energy_metrics['rmae_perc']),
                'mae_norm': float(energy_metrics['mae_norm'])
            },
            'force_metrics': {
                'mae': float(force_metrics['mae']),
                'r2': float(force_metrics['r2']),
                'mae_components': {
                    k: float(v) for k, v in force_metrics['mae_components'].items()
                },
                'rmae': float(force_metrics['rmae']),
                'rmae_perc': float(force_metrics['rmae_perc']),
                'mae_norm': float(force_metrics['mae_norm'])
            }
        }
    pred_data_dic = {
    'pred_energies': [float(e) for e in pred_energies],
    'true_energies': [float(e) for e in energies_true],
    'pred_forces': [float(f) for f in pred_forces_array.flatten().tolist()],
    'true_forces': [float(f) for f in true_forces_array.flatten().tolist()]
    }
    
    return pred_metrics, pred_data_dic
        
    

def print_metrics(metrics_dict, decimal_places=3):
    """Print evaluation metrics in a formatted table"""
    for dataset_name, metrics in metrics_dict.items():
        print(f"\n{' ' + dataset_name.upper() + ' METRICS ':-^80}")
        
        # Energy metrics (updated key check)
        if 'energy_metrics' in metrics:
            print("\nENERGY:")
            energy = metrics['energy_metrics']
            print(f"  {'MAE:':<15} {energy['mae']:.{decimal_places}f} eV")
            print(f"  {'R²:':<15} {energy['r2']:.{decimal_places}f}")
            print(f"  {'MAE/atom:':<15} {energy['mae_per_atom']:.{decimal_places}f} eV/atom")
            print("RMAE Energy Error: {}".format(energy['rmae']))
            print("RMAE Energy Error as Percentage: {}".format(energy['rmae_perc']))
            print("Normalized Energy MAE: {}".format(energy['mae_norm']))
        
        # Force metrics (updated key check)
        if 'force_metrics' in metrics:
            print("\nFORCES:")
            force = metrics['force_metrics']
            print(f"  {'MAE:':<15} {force['mae']:.{decimal_places}f} eV/Å")
            print(f"  {'R²:':<15} {force['r2']:.{decimal_places}f}")
            print("  Component MAE:")
            for comp, value in force['mae_components'].items():
                print(f"    {comp.upper()+':':<7} {value:.{decimal_places}f} eV/Å")
            print("RMAE Force Error: {}".format(force['rmae']))
            print("RMAE Force Error as percentage: {}".format(force['rmae_perc']))
            print("Normalized F MAE per force vector: {}".format(force['mae_norm']))
        
        # Barrier metrics (updated key check)
        if 'barrier_metrics' in metrics:
            print("\nBARRIERS:")
            barrier = metrics['barrier_metrics']
            print(f"  {'Left MAE:':<15} {barrier['left_mae']:.{decimal_places}f} eV")
            print(f"  {'Left R²:':<15} {barrier['left_r2']:.{decimal_places}f}")
            print(f"  {'Right MAE:':<15} {barrier['right_mae']:.{decimal_places}f} eV")
            print(f"  {'Right R²:':<15} {barrier['right_r2']:.{decimal_places}f}")
            print(f"  {'Total MAE:':<15} {barrier['total_mae']:.{decimal_places}f} eV")
            print(f"  {'Total R²:':<15} {barrier['total_r2']:.{decimal_places}f}")
        
        print("-" * 80)

### EVALUATION ###
all_metrics = {}
all_predics = {}

all_metrics['test'], all_predics['test']  = evaluate_dataset(
    model,
    data_test_all['elements'],
    data_test_all['coordinates'],
    data_test_all['energy'],
    data_test_all['forces'],
    'test', 
    outdir
)

# 2. Evaluation dataset

all_metrics['eval'], all_predics['eval']  = evaluate_dataset(
    model,
    data_eval_all['elements'],
    data_eval_all['coordinates'],
    data_eval_all['energy'],
    data_eval_all['forces'],
    'eval', 
    outdir
)


# 3. Linear interpolation datasets
for dataset_name, data, barriers in [('lin_ID10', data_lin_id10, barriers_lin_id10),
                                     ('lin_eval', data_lin_eval, barriers_lin_eval)]:
    
    outdir_barriers = '{}/{}_barries_all'.format(outdir, dataset_name)
    if not os.path.exists(outdir_barriers):
        os.makedirs(outdir_barriers)
    
    elements = [d['elements'] for d in data]
    coords = [d['coordinates'] for d in data]
    energies = [d['energy'] for d in data]
    forces = [d['forces'] for d in data]
    
    #print('LEN EN data', len(energies))
    
    # Energy/force metrics
    lin_metrics, lin_predics = evaluate_dataset(
        model,
        elements,
        coords,
        energies,
        forces,
        dataset_name, 
        outdir
    )
    
    # Barrier metrics
    barrier_metrics, pred_barriers, metrics_barriers_single = evaluate_barriers(
        model,
        dataset_name,
        data,
        barriers,
        scaler,
        outdir_barriers
    )
    
    
    # Combine results
    all_metrics[dataset_name] = {
        **lin_metrics,
        'barrier_metrics': barrier_metrics
        #'pred_barriers': pred_barriers
    }
    all_predics[dataset_name] = {**lin_predics}
    
    # Save barrier predictions
    with open(f'{outdir}/{dataset_name}_barrier_predictions.json', 'w') as f:
        json.dump(pred_barriers, f, indent=4)
        
    with open(f'{outdir}/{dataset_name}_barrier_predictions_all.json', 'w') as f:
        json.dump(metrics_barriers_single, f, indent=4)        

print_metrics(all_metrics, decimal_places=6)

## save and plot ##
def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError

with open(f'{outdir}/all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=4, default=convert_numpy)

with open(f'{outdir}/all_predicts.json', 'w') as f:
    json.dump(all_predics, f, indent=4, default=convert_numpy)


# Generate error histograms
def plot_error_histogram(errors, title, filename, xlabel='Error'):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(filename, dpi = 300)
    plt.close()

# Energy error histograms
for dataset in ['test', 'eval', 'lin_ID10', 'lin_eval']:
    errors = (np.array(all_predics[dataset]['pred_energies']) 
              - np.array(all_predics[dataset]['true_energies']))
    plot_error_histogram(
        errors,
        f'{dataset} Energy Errors',
        f'{outdir}/{dataset}_energy_error_hist.png',
        xlabel='Energy Error (eV)'
    )

# Force error histograms
for dataset in ['test', 'eval', 'lin_ID10', 'lin_eval']:
    errors = (np.array(all_predics[dataset]['pred_forces']) 
              - np.array(all_predics[dataset]['true_forces']))
    plot_error_histogram(
        errors.flatten(),
        f'{dataset} Force Errors',
        f'{outdir}/{dataset}_force_error_hist.png',
        xlabel='Force Error (eV/Å)'
    )

## analyse barriers ##



