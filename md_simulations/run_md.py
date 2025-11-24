import argparse
from pathlib import Path
import numpy as np
import math
import re
import csv
import json, time as _time
from ase import units, Atoms
from ase.io import read
from ase.io.extxyz import write_extxyz
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms, FixCom
from ase.calculators.calculator import Calculator, all_changes
from ase.md.velocitydistribution import Stationary, ZeroRotation

from mace.calculators import MACECalculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DistanceBias(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, base_calc, i, j, k_eV_A2=0.5, r0_A=2.3):
        super().__init__()
        self.base = base_calc
        self.i, self.j = int(i), int(j)
        self.k, self.r0 = float(k_eV_A2), float(r0_A)
    def calculate(self, atoms=None, properties=("energy","forces"), system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)
        e = float(self.base.results["energy"])
        f = self.base.results["forces"].copy()
        R = atoms.get_positions()
        rij = R[self.i] - R[self.j]
        r = np.linalg.norm(rij)
        if r > 0.0:
            eij = rij / r
            dEdr = self.k * (r - self.r0)
            e += 0.5 * self.k * (r - self.r0)**2
            f[self.i] += -dEdr * eij
            f[self.j] += +dEdr * eij
        self.results["energy"] = e
        self.results["forces"] = f

class QBias(Calculator):
    "Harmonic bias on q = r(D-H) - r(H-A) to promote H transfer."
    implemented_properties = ["energy", "forces"]
    def __init__(self, base_calc, donor, h, acceptor, k_eV_A2=1.0, q0_A=0.0):
        super().__init__()
        self.base = base_calc
        self.D, self.H, self.A = int(donor), int(h), int(acceptor)
        self.k = float(k_eV_A2)
        self.q0 = float(q0_A)  # can be updated during MD (steering)
    def calculate(self, atoms=None, properties=("energy","forces"), system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)
        e = float(self.base.results["energy"])
        f = self.base.results["forces"].copy()
        R = atoms.get_positions()

        # Distances and unit vectors
        r_DH_vec = R[self.D] - R[self.H]; r_DH = np.linalg.norm(r_DH_vec); e_DH = r_DH_vec / (r_DH + 1e-12)
        r_HA_vec = R[self.H] - R[self.A]; r_HA = np.linalg.norm(r_HA_vec); e_HA = r_HA_vec / (r_HA + 1e-12)

        q = r_DH - r_HA
        dEdq = self.k * (q - self.q0)
        e += 0.5 * self.k * (q - self.q0)**2

        # Forces from DH part
        f[self.D] += -dEdq * e_DH
        f[self.H] += +dEdq * e_DH
        # Forces from HA part (note the minus sign in q definition)
        f[self.H] += +dEdq * e_HA
        f[self.A] += -dEdq * e_HA

        self.results["energy"] = e
        self.results["forces"] = f

def select_frame(frames, index=None, unique_id=None, infile_path=None):
    "Select a frame by index or unique_id, with robust header fallback."
    if isinstance(frames, Atoms):
        return frames if unique_id is None else frames

    if unique_id is None:
        return frames[0] if index is None else frames[index]

    
    for at in frames:
        uid = at.info.get("unique_id")
        if uid is not None and str(uid) == str(unique_id):
            return at

   
    if infile_path is None:
        raise ValueError("unique_id lookup needs infile_path for header scan.")
    headers = []
    with open(infile_path, "r") as f:
        while True:
            nat_line = f.readline()
            if not nat_line:
                break
            hdr_line = f.readline()
            if not hdr_line:
                break
            headers.append(hdr_line)
            try:
                n = int(nat_line.strip())
            except Exception:
                break
            for _ in range(n):
                f.readline()
    target_idx = None
    for i, hdr in enumerate(headers):
        m = re.search(r'unique_id=([^\s]+)', hdr)
        if m and m.group(1) == str(unique_id):
            target_idx = i
            break
    if target_idx is not None:
        return frames[target_idx]
    raise ValueError(f"unique_id '{unique_id}' not found in file headers.")


def friction_to_ase(gamma_ps):
    
    return (gamma_ps / 1000.0) / units.fs


def attach_logging(dyn, out_xyz_path, log_csv_path, stride, hat_indices=None, mic=False):
    "Attach frame writer + CSV logger with energies, T, and optional HAT distances."
    
    def write_frame():
        with open(out_xyz_path, "a") as f:
            write_extxyz(f, dyn.atoms, write_results=True)
    dyn.attach(write_frame, interval=stride)

    header_written = {"flag": False}
    donor_idx, h_idx, acc_idx = (hat_indices if hat_indices is not None else (None, None, None))

    def log_status():
        t_fs  = dyn.get_time() / units.fs
        epot  = dyn.atoms.get_potential_energy()
        ekin  = dyn.atoms.get_kinetic_energy()
        etot  = epot + ekin
        T     = dyn.atoms.get_temperature()

        # HAT distances (Å)
        r_DH = r_HA = q = math.nan
        if donor_idx is not None and h_idx is not None and acc_idx is not None:
            r_DH = dyn.atoms.get_distance(donor_idx, h_idx, mic=mic)
            r_HA = dyn.atoms.get_distance(h_idx, acc_idx, mic=mic)
            q    = r_DH - r_HA

        mode = "a" if header_written["flag"] else "w"
        with open(log_csv_path, mode, newline="") as f:
            if not header_written["flag"]:
                f.write("time_fs,Etot_eV,Epot_eV,Ekin_eV,T_K,r_DH_A,r_HA_A,q_A\n")
                header_written["flag"] = True
            f.write(f"{t_fs:.6f},{etot:.10f},{epot:.10f},{ekin:.10f},{T:.4f},{r_DH:.6f},{r_HA:.6f},{q:.6f}\n")

    dyn.attach(log_status, interval=stride)


def plot_from_csv(log_csv_path, outdir: Path):
    times, Etot, Epot, Ekin, T, rDH, rHA, q = [], [], [], [], [], [], [], []
    with open(log_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_fs"]))
            Etot.append(float(row["Etot_eV"]))
            Epot.append(float(row["Epot_eV"]))
            Ekin.append(float(row["Ekin_eV"]))
            T.append(float(row["T_K"]))
            
            rDH.append(float(row["r_DH_A"]))
            rHA.append(float(row["r_HA_A"]))
            q.append(float(row["q_A"]))

    # 1) Energies
    plt.figure()
    plt.plot(times, Etot, label="Etot")
    #plt.plot(times, Epot, label="Epot")
    #plt.plot(times, Ekin, label="Ekin")
    plt.xlabel("time (fs)")
    plt.ylabel("Energy (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "energy_vs_time.png", dpi=200)
    plt.close()

    # 2) Temperature
    plt.figure()
    plt.plot(times, T, label="T")
    plt.xlabel("time (fs)")
    plt.ylabel("Temperature (K)")
    plt.tight_layout()
    plt.savefig(outdir / "temperature_vs_time.png", dpi=200)
    plt.close()

    # 3) HAT distances if provided
    any_hat = any(not math.isnan(x) for x in rDH) and any(not math.isnan(x) for x in rHA)
    if any_hat:
        plt.figure()
        plt.plot(times, rDH, label="r_D-H")
        plt.plot(times, rHA, label="r H-A")
        plt.xlabel("time (fs)")
        plt.ylabel("Distance (Å)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "hat_distances_vs_time.png", dpi=200)
        plt.close()

        # 4) Reaction coordinate
        plt.figure()
        plt.plot(times, q)
        plt.xlabel("time (fs)")
        plt.ylabel("q = r(D-H) - r(H-A) (Å)")
        plt.tight_layout()
        plt.savefig(outdir / "hat_q_vs_time.png", dpi=200)
        plt.close()


def main():
    p = argparse.ArgumentParser(description="Run MD with MACE+ASE (HAT-ready logging & plots).")
    p.add_argument("--infile", required=True, help="Input .extxyz (single or multi-frame).")
    p.add_argument("--frame-index", type=int, default=None, help="Frame index (default 0).")
    p.add_argument("--unique-id", type=str, default=None, help="Select by info['unique_id'].")
    p.add_argument("--model", default="models/dft_ID2/dft_ID2_stagetwo.model",
                   help="Path to MACE .model (stage two).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device.")
    p.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"], help="NVT (Langevin) or NVE.")
    p.add_argument("--steps", type=int, default=20000, help="Number of MD steps.")
    p.add_argument("--dt-fs", type=float, default=0.5, help="Timestep in fs.")
    p.add_argument("--T", type=float, default=310.0, help="Target temperature (K).")
    p.add_argument("--gamma-ps", type=float, default=2.0, help="Langevin friction γ (ps^-1).")
    p.add_argument("--stride", type=int, default=50, help="Write frame/log every N steps.")
    p.add_argument("--outdir", default="md_out", help="Output directory.")
    p.add_argument("--thermalize-nvt", action="store_true",
                   help="For NVE: do a short ~5 ps NVT thermalization first.")
    # Constraints
    p.add_argument("--fix-indices", type=str, default="", help="Comma-separated 0-based indices to freeze, e.g. '0,5,9'.")
    p.add_argument("--fix-com", action="store_true", help="Fix global center of mass.")
    # HAT metrics
    p.add_argument("--hat-donor", type=int, default=None, help="Donor heavy atom index.")
    p.add_argument("--hat-h", type=int, default=None, help="Transferring H index.")
    p.add_argument("--hat-acceptor", type=int, default=None, help="Acceptor heavy atom index.")
    p.add_argument("--mic", action="store_true", help="Use minimum-image convention for distances (periodic).")
    p.add_argument("--bias-distance", type=str, default="", 
               help="Apply harmonic bias on a distance: 'i,j,r0_A,k_eV_A2'")
    p.add_argument("--bias-q", action="store_true", help="Apply harmonic bias on q=r(D-H)-r(H-A).")
    p.add_argument("--q-k", type=float, default=1.0, help="Bias stiffness for q (eV/Å^2).")
    p.add_argument("--q0", type=float, default=None, help="Initial q0 target (Å). If omitted, use current q.")
    p.add_argument("--q0-final", type=float, default=None, help="Final q0 (Å) for steering.")
    p.add_argument("--q-steps", type=int, default=None, help="Steps to ramp q0 to q0-final.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for velocity initialization.")
    p.add_argument("--drop-bias-after-ramp", action="store_true",
                help="After finishing q0 ramp (--q-steps), set q-bias k->0 for unbiased remainder.")
    p.add_argument("--detect-transfer", action="store_true",
                help="Post-run: detect first sustained crossing to product side.")
    p.add_argument("--q-thresh", type=float, default=0.0,
                help="Threshold for product side detection in Å (default: 0.0).")
    p.add_argument("--min-dwell-fs", type=float, default=200.0,
                help="Minimum dwell time above threshold to count as transfer (fs).")
    p.add_argument("--label", type=str, default="", help="Free-form label stored into run_config.json.")



    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    traj_path = outdir / "trajectory.xyz"
    log_path = outdir / "md_log.csv"

    # Load structures
    frames = read(args.infile, ":")
    atoms = select_frame(frames, index=args.frame_index, unique_id=args.unique_id, infile_path=args.infile)

    # MACE calculator
    base_calc = MACECalculator(model_paths=[args.model], device=args.device, default_dtype="float64")
    calc = base_calc

    if args.bias_distance:
        i, j, r0, k = [float(x) for x in args.bias_distance.split(",")]
        calc = DistanceBias(calc, int(i), int(j), k_eV_A2=k, r0_A=r0)

    # Optional: q-bias (requires --hat-donor/--hat-h/--hat-acceptor)
    q_bias_ref = None
    if args.bias_q:
        if None in (args.hat_donor, args.hat_h, args.hat_acceptor):
            raise ValueError("--bias-q needs --hat-donor, --hat-h, --hat-acceptor.")
        # default q0 = current q if not given
        if args.q0 is None:
            R = atoms.get_positions()
            def dist(a,b): 
                v = R[a]-R[b]; return float(np.linalg.norm(v))
            q0_init = dist(args.hat_donor, args.hat_h) - dist(args.hat_h, args.hat_acceptor)
        else:
            q0_init = args.q0
        q_bias_ref = QBias(calc, args.hat_donor, args.hat_h, args.hat_acceptor, k_eV_A2=args.q_k, q0_A=q0_init)
        calc = q_bias_ref

    atoms.calc = calc


    # Constraints
    constraints = []
    if args.fix_indices:
        fixed = [int(tok) for tok in args.fix_indices.split(",") if tok.strip() != ""]
        constraints.append(FixAtoms(indices=fixed))
    if args.fix_com:
        constraints.append(FixCom())
    if constraints:
        atoms.set_constraint(constraints)

    # Velocities
    if args.seed is not None:
        np.random.seed(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.T)

    try:
        Stationary(atoms)   
        ZeroRotation(atoms)   
    except Exception:
        pass  

    
    hat_indices = None
    if (args.hat_donor is not None) or (args.hat_h is not None) or (args.hat_acceptor is not None):
        if None in (args.hat_donor, args.hat_h, args.hat_acceptor):
            print("[WARN] Provide all three of --hat-donor, --hat-h, --hat-acceptor to log HAT distances.")
        else:
            hat_indices = (args.hat_donor, args.hat_h, args.hat_acceptor)

    
    dt = args.dt_fs * units.fs

    if args.ensemble == "nvt":
        dyn = Langevin(atoms, timestep=dt, temperature_K=args.T, friction=friction_to_ase(args.gamma_ps))
        attach_logging(dyn, traj_path, log_path, args.stride, hat_indices=hat_indices, mic=args.mic)
        
        if q_bias_ref is not None and args.q0_final is not None and args.q_steps:
            q0_start = q_bias_ref.q0            
            q0_end   = args.q0_final            
            total    = max(int(args.q_steps), 1)

            def steer_q0():
                s = min(dyn.nsteps, total)      
                lam = s / total                 
                q_bias_ref.q0 = (1.0 - lam) * q0_start + lam * q0_end

            dyn.attach(steer_q0, interval=1)

            if args.drop_bias_after_ramp:
                dropped = {"done": False}
                def drop_bias_once():
                    if not dropped["done"] and dyn.nsteps >= total:
                        q_bias_ref.k = 0.0
                        dropped["done"] = True

                dyn.attach(drop_bias_once, interval=10)       

        cfg = {
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "infile": args.infile,
            "frame_index": args.frame_index,
            "unique_id": args.unique_id,
            "model": args.model,
            "device": args.device,
            "ensemble": args.ensemble,
            "steps": args.steps,
            "dt_fs": args.dt_fs,
            "T": args.T,
            "gamma_ps": args.gamma_ps,
            "stride": args.stride,
            "fix_indices": args.fix_indices,
            "fix_com": args.fix_com,
            "hat_donor": args.hat_donor,
            "hat_h": args.hat_h,
            "hat_acceptor": args.hat_acceptor,
            "bias_q": bool("q_bias_ref" in locals() and q_bias_ref is not None),
            "q_k": getattr(q_bias_ref, "k", None) if "q_bias_ref" in locals() else None,
            "q0_start": q0_start if "q0_start" in locals() else None,
            "q0_final": args.q0_final,
            "q_steps": args.q_steps,
            "drop_bias_after_ramp": args.drop_bias_after_ramp,
            "seed": args.seed,
            "label": args.label,
        }
        with open(outdir / "run_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        dyn.run(args.steps)

    else:  # nve
        if args.thermalize_nvt:
            steps_th = int(np.ceil(5.0e3 / args.dt_fs))  # ~5 ps
            dyn_th = Langevin(atoms, timestep=dt, temperature_K=args.T, friction=friction_to_ase(args.gamma_ps))
            attach_logging(dyn_th, traj_path, log_path, args.stride, hat_indices=hat_indices, mic=args.mic)

            # ---- q0 RAMP during THERMALIZATION (optional) + optional drop-bias ----
            if q_bias_ref is not None and args.q0_final is not None and args.q_steps:
                
                q0_start_th = q_bias_ref.q0
                q0_end      = args.q0_final
                total_ramp  = max(int(args.q_steps), 1)

                def steer_q0_th():
                    
                    s = min(dyn_th.nsteps, total_ramp)
                    lam = s / total_ramp
                    q_bias_ref.q0 = (1.0 - lam) * q0_start_th + lam * q0_end

                dyn_th.attach(steer_q0_th, interval=1)

                if args.drop_bias_after_ramp:
                    dropped_th = {"done": False}
                    def drop_bias_once_th():
                        
                        if not dropped_th["done"] and dyn_th.nsteps >= total_ramp:
                            q_bias_ref.k = 0.0
                            dropped_th["done"] = True
                    dyn_th.attach(drop_bias_once_th, interval=10)

            dyn_th.run(steps_th)
        dyn = VelocityVerlet(atoms, dt)
        attach_logging(dyn, traj_path, log_path, args.stride, hat_indices=hat_indices, mic=args.mic)

        if q_bias_ref is not None and args.q0_final is not None and args.q_steps:
            q0_start = q_bias_ref.q0
            q0_end   = args.q0_final
            total    = max(int(args.q_steps), 1)  
            def steer_q0():
                s = min(dyn.nsteps, total)
                lam = s / total
                q_bias_ref.q0 = (1.0 - lam) * q0_start + lam * q0_end
            dyn.attach(steer_q0, interval=1)

        cfg = {
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "infile": args.infile,
            "frame_index": args.frame_index,
            "unique_id": args.unique_id,
            "model": args.model,
            "device": args.device,
            "ensemble": args.ensemble,
            "steps": args.steps,
            "dt_fs": args.dt_fs,
            "T": args.T,
            "gamma_ps": args.gamma_ps,
            "stride": args.stride,
            "fix_indices": args.fix_indices,
            "fix_com": args.fix_com,
            "hat_donor": args.hat_donor,
            "hat_h": args.hat_h,
            "hat_acceptor": args.hat_acceptor,
            "bias_q": bool("q_bias_ref" in locals() and q_bias_ref is not None),
            "q_k": getattr(q_bias_ref, "k", None) if "q_bias_ref" in locals() else None,
            "q0_start": q0_start if "q0_start" in locals() else None,
            "q0_final": args.q0_final,
            "q_steps": args.q_steps,
            "drop_bias_after_ramp": args.drop_bias_after_ramp,
            "seed": args.seed,
            "label": args.label,
        }
        with open(outdir / "run_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        dyn.run(args.steps)

    # Plots
    plot_from_csv(log_path, outdir)

    summary = {}
    if args.detect_transfer:
        
        times, qvals = [], []
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row["time_fs"])
                qv = float(row["q_A"])
                if math.isnan(qv): 
                    continue
                times.append(t); qvals.append(qv)
        crossed = False; t_cross = None
        if qvals:
            dwell_steps = max(1, int(args.min_dwell_fs / (args.dt_fs * args.stride)))
            for i in range(len(qvals)):
                if qvals[i] > args.q_thresh:
                    j = i + dwell_steps
                    if j < len(qvals) and all(q > args.q_thresh for q in qvals[i:j]):
                        crossed = True
                        t_cross = times[i]
                        break
        # recross within 2 ps?
        recross = False
        if crossed:
            dwell2 = max(1, int(2000.0 / (args.dt_fs * args.stride)))
            
            for i in range(int(i + dwell_steps), len(qvals)):
                if qvals[i] <= args.q_thresh:
                    j = i + dwell2
                    if j < len(qvals) and all(q <= args.q_thresh for q in qvals[i:j]):
                        recross = True
                        break
        # temperature stats
        Ts = []
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                Ts.append(float(row["T_K"]))

        mean_T = float(np.mean(Ts)) if Ts else None
        std_T  = float(np.std(Ts)) if Ts else None
        summary = {
            "crossed": crossed,
            "t_first_cross_fs": t_cross,
            "recrossed_within_2ps": recross,
            "mean_T": mean_T,
            "std_T": std_T,
            "q_thresh": args.q_thresh,
            "min_dwell_fs": args.min_dwell_fs
        }
        with open(outdir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    print(f"[SUMMARY] {summary}" if summary else "[SUMMARY] detection disabled")



if __name__ == "__main__":
    main()