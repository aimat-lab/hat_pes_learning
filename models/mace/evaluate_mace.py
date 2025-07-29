#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from ase.io import read
import os
from collections import defaultdict
import json

# utils

def read_results_log(filepath):
    
    for line in open(filepath, 'r'):
        
        if ' f_mae' in line:
            f_mae = float(line.split()[-1])
        
        if 'e_mae' in line:
            e_mae = float(line.split()[-1])
            
        if 'e/N_mae' in line:
            e_Nmae = float(line.split()[-1])
            
    return e_mae, e_Nmae, f_mae

def mace_read_results_log(output_file):
    read_error = False
    
    with open(output_file, "r") as f:
        for line in f:
            if "Loaded Stage two model from epoch" in line:
                read_error = True
                continue
            if read_error:
                parts = line.split()
                #print(parts)
                if "|" in parts and 'train' in parts:
                    e_train_err = float(parts[3])
                    f_train_err = float(parts[5])
                    f_train_err_rel = float(parts[7])
                    
                if "|" in parts and 'valid' in parts:
                    e_valid_err = float(parts[3])
                    f_valid_err = float(parts[5])
                    f_valid_err_rel = float(parts[7])
                
    return e_train_err, f_train_err, e_valid_err, f_valid_err, f_train_err_rel, f_valid_err_rel

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

def mace_parse_extxyz_ase(file_path):
    data = {
        "energy": [],
        "forces": [],
        "elements": [],
        "coordinates": []
    }

    atoms_list = read(file_path, index=":")

    for atoms in atoms_list:
        # Extract energy (should be present in atoms.info)
        energy = atoms.info.get("MACE_energy", None)
        if energy is None:
            print("Warning: Energy missing for an entry!")
        data["energy"].append(energy)

        # Extract atomic forces (should be in atoms.arrays)
        forces = atoms.arrays.get("MACE_forces", None)
        if forces is None:
            print("Warning: Forces missing for an entry!")
        data["forces"].append(forces.tolist() if forces is not None else None)

        # Extract atomic symbols and coordinates
        data["elements"].append(atoms.get_chemical_symbols())
        data["coordinates"].append(atoms.get_positions().tolist())

    return data

def allegro_parse_extxyz_ase(file_path):
    data = {
        "energy": [],
        "forces": [],
        "elements": [],
        "coordinates": []
    }

    atoms_list = read(file_path, index=":")

    for atoms in atoms_list:
        # Extract energy (should be present in atoms.info)
        energy = atoms.info.get("energy", None)
        if energy is None:
            print("Warning: Energy missing for an entry!")
        data["energy"].append(energy)

        # Extract atomic forces (should be in atoms.arrays)
        forces = atoms.arrays.get("forces", None)
        if forces is None:
            print("Warning: Forces missing for an entry!")
        data["forces"].append(forces.tolist() if forces is not None else None)

        # Extract atomic symbols and coordinates
        data["elements"].append(atoms.get_chemical_symbols())
        data["coordinates"].append(atoms.get_positions().tolist())

    return data

# metrics
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

def evaluate_dataset(data_true, data_pred, dataset_name, outdir, modelname):
    
    if isinstance(data_true, list):
        energies_true = [entry['energy'] for entry in data_true]
        forces_true = [entry['forces'] for entry in data_true]
        system_sizes = [entry['n_atoms'] for entry in data_true]
    
    if isinstance(data_true, dict):
        energies_true = data_true['energy']
        forces_true = data_true['forces']
        system_sizes = [len(g) for g in data_true['elements']] 
    
    if isinstance(data_pred, list):
        energies_pred = [entry['energy'] for entry in data_pred]
        forces_pred = [entry['forces'] for entry in data_pred]
    
    if isinstance(data_pred, dict):
        energies_pred = data_pred['energy']
        forces_pred = data_pred['forces']
    
    
    
    # Now process forces (they'll match true_force_list shapes)
    f_true = np.vstack([np.array(config) for config in forces_true])
    f_pred = np.vstack([np.array(config) for config in forces_pred])
    #f_true = np.vstack(forces_true)  
    #f_pred = np.vstack(forces_pred)  
    
    assert f_true.shape == f_pred.shape, \
        f"Force shape mismatch after trimming: {f_true.shape} vs {f_pred.shape}"

    f_true_flat = f_true.flatten()  # Shape (N_total_atoms*3,)
    f_pred_flat = f_pred.flatten()

    #f_true_flat, f_pred_flat, metrics = process_force_data(forces_true, pred_forces, dataset_name)      

    RMAE_e = relative_mae(energies_true, energies_pred)
    RMAE_f, f_err_list = relative_force_error(forces_true, forces_pred)
    
    mae_e_norm = normalized_mae(energies_true, energies_pred)

    mae_f_norm = normalized_mae_forces(forces_true, forces_pred)

    true_energies_array = np.array(energies_true)
    pred_energies_array = np.array(energies_pred)
    system_sizes_array = np.array(system_sizes)

    # Calculate MAE per atom
    mae_per_atom = mean_absolute_error(
    true_energies_array / system_sizes_array,
    pred_energies_array / system_sizes_array)
    
       
    # Energy metrics
    energy_metrics = {
        'mae': mean_absolute_error(true_energies_array, pred_energies_array),
        'r2': r2_score(true_energies_array, pred_energies_array),
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
        energies_true, energies_pred, 
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
        y_predict=np.array(energies_pred), 
        y_true=np.array(energies_true),
        target_names=["Energy"],
        data_unit=["eV"],
        model_name=modelname,
        dataset_name=dataset_name,
        filepath = outdir,
        figsize=[10, 6]
    )
    
    plot_force_distributions(f_true, f_pred, dataset_name, outdir)

    f_pred_reshaped = np.array(f_pred_flat).reshape(-1, 3)
    f_true_reshaped = np.array(f_true_flat).reshape(-1, 3)
    
    pred_forces_array = np.concatenate([np.array(f_mol) for f_mol in forces_pred])  #np.concatenate(pred_forces)  # Shape (N_total_atoms, 3)
    true_forces_array = np.concatenate([np.array(f_mol) for f_mol in forces_true])  # Shape (N_total_atoms, 3)

    
    plot_predict_true(
        y_predict=f_pred_reshaped, 
        y_true=f_true_reshaped,
        target_names=["Fx", "Fy", "Fz"],
        data_unit=["eV/Å", "eV/Å", "eV/Å"],
        model_name=modelname,
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
    'pred_energies': [float(e) for e in energies_pred],
    'true_energies': [float(e) for e in energies_true],
    'pred_forces': [float(f) for f in pred_forces_array.flatten().tolist()],
    'true_forces': [float(f) for f in true_forces_array.flatten().tolist()]
    }
    
    return pred_metrics, pred_data_dic

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


def evaluate_barriers(data_true, data_pred, barriers_true, barriers_pred, dataset_name, outdir):
    
    system_ids = list(barriers_true.keys())
    
    missing = [sys_id for sys_id in barriers_true if sys_id not in barriers_pred]
    if missing:
        raise ValueError(f"Missing predicted barriers for systems: {missing}")
    
    system_data = {}
    
    for sys_id in system_ids:
        # Get true entries (sorted by step)
        true_entries = sorted(
            [e for e in data_true if e['system_id'] == sys_id],
            key=lambda x: x['step']
        )
        
        # Get corresponding predicted entries (sorted by step)
        pred_entries = sorted(
            [e for e in data_pred if e['system_id'] == sys_id],
            key=lambda x: x['step']
        )

        # Safety checks
        if len(true_entries) != len(pred_entries):
            print(f"Step count mismatch for {sys_id}, skipping plot")
            continue

        if any(t['step'] != p['step'] for t, p in zip(true_entries, pred_entries)):
            print(f"Step alignment issue in {sys_id}, skipping plot")
            continue

        # Store data for plotting
        system_data[sys_id] = {
            'h0_dists': [e['h0_dist'] for e in true_entries],
            'true_energies': [e['energy'] for e in true_entries],
            'pred_energies': [e['energy'] for e in pred_entries]
        }

    # 2. Plot individual system energy curves
    for sys_id, data in system_data.items():
        plot_system_energy_curves(
            sys_id, 
            data['h0_dists'],
            data['true_energies'],
            data['pred_energies'],
            dataset_name,
            outdir
        )
        

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

# plots

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

# prep outdir
model = 'mace'

eval_name = 'dft_ID4'

outdir = 'barrier_evaluations/{}'.format(eval_name) # allegro_eval , evaluation

if not os.path.exists(outdir):
    os.makedirs(outdir)

# load data

datasetname = 'dft_ID4'

indir_data_true = 'data/dft_IDs' # 'data/dft_IDs'  xtb_IDs_new

indir_data_pred = 'evaluation/{}'.format(eval_name)

# test true
name_data_test_true = 'dft_test'
data_test_true = parse_extxyz('{}/{}/{}_full.extxyz'.format(indir_data_true, name_data_test_true , name_data_test_true)) # 

# eval true

name_data_eval_dir = 'dft_eval'
name_data_eval = 'dft_eval'

data_eval_true = parse_extxyz('{}/{}/{}_full.extxyz'.format(indir_data_true, name_data_eval_dir, name_data_eval)) # 

# lin ID10 true
name_data_lin_ID10_dir = 'dft_lin_ID10'
name_data_lin_ID10 = 'dft_lin_ID10'

indir_lin_id10 = '{}/{}'.format(indir_data_true, name_data_lin_ID10_dir) 

data_lin_id10_true = parse_extxyz_steps('{}/{}_full.extxyz'.format(indir_lin_id10, name_data_lin_ID10))
barriers_lin_id10_true = calculate_barriers(data_lin_id10_true)

# lin eval true
name_data_lin_eval_dir = 'dft_lin_eval'
name_data_lin_eval = 'dft_lin_eval'

data_lin_eval_true = parse_extxyz_steps('{}/{}/{}_full.extxyz'.format(indir_data_true, name_data_lin_eval_dir, name_data_lin_eval))
barriers_lin_eval_true = calculate_barriers(data_lin_eval_true)

# load predictions

# test
path_data_test_pred = '{}/hat_train.xyz'.format(indir_data_pred) 

if model == 'allegro':
    data_test_pred = allegro_parse_extxyz_ase(path_data_test_pred)
if model == 'mace':
    data_test_pred = mace_parse_extxyz_ase(path_data_test_pred)

# eval
path_data_eval_pred = '{}/hat_eval.xyz'.format(indir_data_pred)

if model == 'allegro':
    data_eval_pred = allegro_parse_extxyz_ase(path_data_eval_pred)
if model == 'mace':
    data_eval_pred = mace_parse_extxyz_ase(path_data_eval_pred)
    
# lin ID10
path_data_lin_test_pred = '{}/hat_lin_train.xyz'.format(indir_data_pred) 

if model == 'allegro':
    data_lin_test_pred = allegro_parse_extxyz_ase(path_data_lin_test_pred)
if model == 'mace':
    data_lin_test_pred = mace_parse_extxyz_ase(path_data_lin_test_pred)

pred_entries_test = []
for i, true_entry in enumerate(data_lin_id10_true):
    pred_entry = {
        'system_id': true_entry['system_id'],
        'step': true_entry['step'],
        'energy': data_lin_test_pred['energy'][i],
        'h0_dist': true_entry['h0_dist'],
        'n_atoms': true_entry['n_atoms'],
    }
    pred_entries_test.append(pred_entry)

barriers_lin_test_pred = calculate_barriers(pred_entries_test)


# lin eval
path_data_lin_eval_pred = '{}/hat_lin_eval.xyz'.format(indir_data_pred)

if model == 'allegro':
    data_lin_eval_pred = allegro_parse_extxyz_ase(path_data_lin_eval_pred)
if model == 'mace':
    data_lin_eval_pred = mace_parse_extxyz_ase(path_data_lin_eval_pred)
 
pred_entries_eval = []
for i, true_entry in enumerate(data_lin_eval_true):
    pred_entry = {
        'system_id': true_entry['system_id'],
        'step': true_entry['step'],
        'energy': data_lin_eval_pred['energy'][i],
        'h0_dist': true_entry['h0_dist'],
        'n_atoms': true_entry['n_atoms'],
    }
    pred_entries_eval.append(pred_entry)   

barriers_lin_eval_pred = calculate_barriers(pred_entries_eval)

# EVALUATION
all_metrics = {}
all_predics = {}
# test
all_metrics['test'], all_predics['test']  = evaluate_dataset(
    data_test_true,
    data_test_pred,
    datasetname, 
    outdir, 
    model
)

# eval
all_metrics['eval'], all_predics['eval']  = evaluate_dataset(
    data_eval_true,
    data_eval_pred,
    datasetname, 
    outdir, 
    model
)

# BARRIERS

# 3. Linear interpolation datasets
for dataset_name, data_true,data_pred, data_pred_barr, barriers_true, barrier_pred in [('lin_test', data_lin_id10_true,data_lin_test_pred, pred_entries_test, barriers_lin_id10_true, barriers_lin_test_pred),
                                     ('lin_eval', data_lin_eval_true,data_lin_eval_pred, pred_entries_eval, barriers_lin_eval_true, barriers_lin_eval_pred)]:
    
    outdir_barriers = '{}/{}_barries_all'.format(outdir, dataset_name)
    if not os.path.exists(outdir_barriers):
        os.makedirs(outdir_barriers)
    
    #print('LEN EN data', len(energies))
    
    # Energy/force metrics
    lin_metrics, lin_predics = evaluate_dataset(
        data_true,
        data_pred,
        dataset_name,
        outdir,
        model
    )
    
    # Barrier metrics
    barrier_metrics, pred_barriers, metrics_barriers_single = evaluate_barriers(
        data_true,
        data_pred_barr,
        barriers_true,
        barrier_pred,
        dataset_name,
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
for dataset in ['test', 'eval', 'lin_test', 'lin_eval']:
    errors = (np.array(all_predics[dataset]['pred_energies']) 
              - np.array(all_predics[dataset]['true_energies']))
    plot_error_histogram(
        errors,
        f'{dataset} Energy Errors',
        f'{outdir}/{dataset}_energy_error_hist.png',
        xlabel='Energy Error (eV)'
    )

# Force error histograms
for dataset in ['test', 'eval', 'lin_test', 'lin_eval']:
    errors = (np.array(all_predics[dataset]['pred_forces']) 
              - np.array(all_predics[dataset]['true_forces']))
    plot_error_histogram(
        errors.flatten(),
        f'{dataset} Force Errors',
        f'{outdir}/{dataset}_force_error_hist.png',
        xlabel='Force Error (eV/Å)'
    )
