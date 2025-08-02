import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------------
# Utility functions
# -------------------------------------

def parse_extxyz(filename):
    """
    Parse extxyz file and extract energies, forces, and IDs.
    Returns:
        energies: np.array of energies (float)
        forces: list of np.arrays (n_atoms x 3)
        ids: list of unique IDs
    """
    energies, forces, ids = [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        n_atoms = int(lines[i].strip())
        header = lines[i + 1]
        # Try to extract unique_id; fallback to index if missing
        if "unique_id" in header:
            uid = header.split('unique_id=')[1].split()[0]
        else:
            uid = f"conf_{i}"
        ids.append(uid)
        # Try to extract energy (key may differ)
        if "energy=" in header:
            energy = float(header.split('energy=')[1].split()[0])
        else:
            raise ValueError(f"Could not find energy= in header: {header}")
        energies.append(energy)
        atom_lines = lines[i + 2:i + 2 + n_atoms]
        force_block = [list(map(float, l.split()[-3:])) for l in atom_lines]
        forces.append(np.array(force_block))
        i += 2 + n_atoms
    return np.array(energies), forces, ids

def compute_force_mae(forces_true, forces_pred):
    """Compute force MAE per atom."""
    mae = []
    for f_true, f_pred in zip(forces_true, forces_pred):
        mae.append(np.abs(f_true - f_pred).mean())
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

# -------------------------------------
# Barrier calculation logic
# -------------------------------------

def extract_barriers(ids, energies, n_points=9):
    """
    For each HAT scan, extract the left and right reaction barrier.
    Assumes:
        - Each scan is n_points long (e.g. 9 for a symmetric path)
        - Configurations are ordered as: scan1_0, scan1_1, ..., scan1_8, scan2_0, ...
    Returns:
        barrier_dict: dict mapping scan prefix to (barrier_left, barrier_right)
    """
    from collections import defaultdict
    scan_energies = defaultdict(list)
    for uid, energy in zip(ids, energies):
        prefix = '_'.join(uid.split('_')[:-1])  # e.g., "123"
        scan_energies[prefix].append((int(uid.split('_')[-1]), energy))

    barrier_dict = {}
    for prefix, pts in scan_energies.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        es = [e for _, e in pts_sorted]
        # Convention: [reactant, ..., TS, ..., product]
        reactant = es[0]
        product = es[-1]
        ts = max(es)
        # Barrier heights as difference from left and right
        barrier_left = ts - reactant
        barrier_right = ts - product
        barrier_dict[prefix] = (barrier_left, barrier_right)
    return barrier_dict

def collect_barriers_for_all(ids_ref, energies_ref, ids_pred, energies_pred, n_points=9):
    """
    Match reference and predicted barriers by scan ID.
    Returns:
        lists of (ref_barrier, pred_barrier) for left and right
    """
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
    """Parity plot for barrier heights."""
    plot_parity(
        ref, pred,
        f"Reference barrier ({side}) [energy units]",
        f"Predicted barrier ({side}) [energy units]",
        f"Barrier Parity Plot ({side})",
        os.path.join(plot_dir, f"{eval_name}_barrier_{side}_parity.png")
    )

# -------------------------------------
# Main evaluation logic
# -------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate MACE/Allegro predictions on HAT datasets.")
    parser.add_argument('--model', choices=['mace', 'allegro'], default='mace', help="Model type to evaluate.")
    parser.add_argument('--eval_name', type=str, default='dft_ID4', help="Evaluation name or output folder.")
    parser.add_argument('--pred_xyz', type=str, required=True, help="Path to predicted .extxyz file.")
    parser.add_argument('--ref_xyz', type=str, required=True, help="Path to reference .extxyz file.")
    parser.add_argument('--plot_dir', type=str, default='plots', help="Directory to save output plots.")
    parser.add_argument('--n_scan_points', type=int, default=9, help="Number of points per HAT scan (default 9).")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"Loading predicted data from: {args.pred_xyz}")
    pred_energies, pred_forces, pred_ids = parse_extxyz(args.pred_xyz)
    print(f"Loading reference data from: {args.ref_xyz}")
    ref_energies, ref_forces, ref_ids = parse_extxyz(args.ref_xyz)

    # Ensure correspondence of IDs for overall metrics (optional)
    if pred_ids != ref_ids:
        print("Warning: IDs do not match in order between prediction and reference. Proceeding with matched scan IDs for barrier analysis only.")

    # --- Energy metrics (all data) ---
    print("Computing energy metrics...")
    mae_e, r2_e = print_metrics(ref_energies, pred_energies, label="Energy")

    # --- Force metrics (all data) ---
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

    # Parity plots for barriers
    plot_barrier_parity(l_ref, l_pred, "left", args.plot_dir, args.eval_name)
    plot_barrier_parity(r_ref, r_pred, "right", args.plot_dir, args.eval_name)

    # Print barrier metrics
    print("\nLeft Barrier:")
    mae_l, r2_l = print_metrics(l_ref, l_pred, label="Left Barrier")
    print("\nRight Barrier:")
    mae_r, r2_r = print_metrics(r_ref, r_pred, label="Right Barrier")

    print("\nEvaluation completed. Plots and metrics saved.")

if __name__ == "__main__":
    main()
