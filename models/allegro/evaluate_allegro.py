import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, r2_score
from ase.io import read

# -----------------------------
# Parsing and utility functions
# -----------------------------

def parse_extxyz(filename):
    """Parse extxyz or xyz file with energies, forces, and unique_ids."""
    data = {
        "energy": [],
        "unique_id": [],
        "coordinates": [],
        "forces": [],
        "elements": [],
    }
    with open(filename, 'r') as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        num_atoms = int(lines[i].strip())
        i += 1
        properties_line = lines[i].strip()
        i += 1
        energy = unique_id = None
        for prop in properties_line.split():
            if '=' not in prop: continue
            key, value = prop.split("=", 1)
            if key == "energy":
                energy = float(value)
            elif key == "unique_id":
                unique_id = value
        data["energy"].append(energy)
        data["unique_id"].append(unique_id)
        coordinates, forces, element_types = [], [], []
        for _ in range(num_atoms):
            atom_line = lines[i].strip().split()
            i += 1
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

def compute_force_mae(forces_true, forces_pred):
    """Compute force MAE per atom."""
    mae = []
    for f_true, f_pred in zip(forces_true, forces_pred):
        mae.append(np.abs(np.array(f_true) - np.array(f_pred)).mean())
    return np.mean(mae)

def plot_parity(x, y, xlabel, ylabel, title, filename):
    """Parity plot with y=x reference."""
    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    min_v = min(np.min(x), np.min(y))
    max_v = max(np.max(x), np.max(y))
    plt.plot([min_v, max_v], [min_v, max_v], 'k--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def print_metrics(y_true, y_pred, label=""):
    """Print MAE and R^2 metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MAE: {mae:.3f}")
    print(f"{label} R^2: {r2:.3f}")
    return mae, r2

def extract_barriers(ids, energies, n_points=9):
    """
    For each HAT scan, extract left and right reaction barrier.
    Assumes configs are ordered: scan1_0, scan1_1,... scan1_8, scan2_0, ...
    Returns: dict: scan prefix -> (barrier_left, barrier_right)
    """
    scan_energies = defaultdict(list)
    for uid, energy in zip(ids, energies):
        prefix = '_'.join(uid.split('_')[:-1])  # e.g., "123"
        scan_energies[prefix].append((int(uid.split('_')[-1]), energy))
    barrier_dict = {}
    for prefix, pts in scan_energies.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        es = [e for _, e in pts_sorted]
        reactant = es[0]
        product = es[-1]
        ts = max(es)
        barrier_left = ts - reactant
        barrier_right = ts - product
        barrier_dict[prefix] = (barrier_left, barrier_right)
    return barrier_dict

def collect_barriers_for_all(ids_ref, energies_ref, ids_pred, energies_pred, n_points=9):
    ref_barriers = extract_barriers(ids_ref, energies_ref, n_points)
    pred_barriers = extract_barriers(ids_pred, energies_pred, n_points)
    left_ref, left_pred, right_ref, right_pred = [], [], [], []
    common = set(ref_barriers.keys()) & set(pred_barriers.keys())
    for key in sorted(common):
        l_ref, r_ref = ref_barriers[key]
        l_pred, r_pred = pred_barriers[key]
        left_ref.append(l_ref)
        left_pred.append(l_pred)
        right_ref.append(r_ref)
        right_pred.append(r_pred)
    return np.array(left_ref), np.array(left_pred), np.array(right_ref), np.array(right_pred)

def plot_barrier_parity(ref, pred, side, plot_dir, eval_name):
    plot_parity(
        ref, pred,
        f"Reference barrier ({side}) [energy units]",
        f"Predicted barrier ({side}) [energy units]",
        f"Barrier Parity Plot ({side})",
        os.path.join(plot_dir, f"{eval_name}_barrier_{side}_parity.png")
    )

# -----------------------------
# Main script logic
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Allegro/MACE predictions on HAT datasets.")
    parser.add_argument('--model', choices=['allegro', 'mace'], default='allegro', help="Model type to evaluate.")
    parser.add_argument('--eval_name', type=str, default='dft_ID2', help="Evaluation name or output folder.")
    parser.add_argument('--pred_xyz', type=str, required=True, help="Path to predicted .extxyz or .xyz file.")
    parser.add_argument('--ref_xyz', type=str, required=True, help="Path to reference .extxyz or .xyz file.")
    parser.add_argument('--plot_dir', type=str, default='plots', help="Directory to save output plots.")
    parser.add_argument('--n_scan_points', type=int, default=9, help="Number of points per HAT scan (default 9).")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"Loading predicted data from: {args.pred_xyz}")
    pred_data = parse_extxyz(args.pred_xyz)
    print(f"Loading reference data from: {args.ref_xyz}")
    ref_data = parse_extxyz(args.ref_xyz)

    pred_energies = np.array(pred_data['energy'])
    ref_energies = np.array(ref_data['energy'])
    pred_forces = pred_data['forces']
    ref_forces = ref_data['forces']
    pred_ids = pred_data['unique_id']
    ref_ids = ref_data['unique_id']

    # --- Energy metrics ---
    print("Computing energy metrics...")
    mae_e, r2_e = print_metrics(ref_energies, pred_energies, label="Energy")

    # --- Force metrics ---
    print("Computing force metrics...")
    mae_f = compute_force_mae(ref_forces, pred_forces)
    print(f"Force MAE (per atom): {mae_f:.3f}")

    # --- Energy Parity Plot ---
    print("Plotting energy parity...")
    plot_parity(ref_energies, pred_energies,
                "Reference Energy", "Predicted Energy",
                f"{args.model.capitalize()} Energy Parity",
                os.path.join(args.plot_dir, f"{args.eval_name}_energy_parity.png"))

    # --- Barrier calculation and analysis ---
    print("Extracting and evaluating HAT reaction barriers...")

    l_ref, l_pred, r_ref, r_pred = collect_barriers_for_all(
        ref_ids, ref_energies, pred_ids, pred_energies, n_points=args.n_scan_points
    )

    plot_barrier_parity(l_ref, l_pred, "left", args.plot_dir, args.eval_name)
    plot_barrier_parity(r_ref, r_pred, "right", args.plot_dir, args.eval_name)

    print("\nLeft Barrier:")
    mae_l, r2_l = print_metrics(l_ref, l_pred, label="Left Barrier")
    print("\nRight Barrier:")
    mae_r, r2_r = print_metrics(r_ref, r_pred, label="Right Barrier")

    print("\nEvaluation completed. Plots and metrics saved.")

if __name__ == "__main__":
    main()
