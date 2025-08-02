# ML Potential Data Generation for Hydrogen Atom Transfer in Peptides

This repository provides a modular pipeline to generate training data (coordinates, energies, forces) for machine-learned potentials targeting hydrogen atom transfer (HAT) reactions in peptides.

The workflow uses:
- Molecular structure construction from SMILES
- Normal Mode Sampling (NMS)
- Radical system generation
- Hydrogen atom transfer (HAT) reaction sampling

---

## Project Structure

mol_data_generation/
├── mol_construction/
├── nms/
├── radical_systems/
├── reactions/
├── utils/
samples/
├── aa_test_nms/
├── aa_test_nms_radicals/
├── aa_test_nms_hat/
sampling_hat.py
sampling_nms.py
sampling_rad_sys.py

### Dependencies

Make sure the following tools and packages are available:
- `xtb` (with environment variable `XTBPATH`) including `crest` for conformer generation ([xtb](https://xtb-docs.readthedocs.io/en/latest/index.html))
- Python 3.8+
- RDKit
- `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`, `pyyaml`

## Demo Scripts
This repository includes demo scripts to illustrate the full workflow for normal mode sampling (NMS), radical system generation, and hydrogen atom transfer sampling.
All demo data and outputs are found in the `data_generation/samples/` folder.
### 1. Normal Mode Sampling 
**Script:** `sampling_nms.py`
**Description:** Generates normal mode samples (coordinates, energies, forces) for molecules.
**Output:** Results written to `data_generation/samples/aa_test_nms`.
### 2. Radical Sampling
**Script:** `sampling_rad_sys.py`
**Description:** Uses NMS samples as input, generates intra- and inter-molecular radical systems.
**Output:** Results are written to `data_generation/samples/aa_test_nms_radicals/`.
### 3. HAT Sampling
**Script:** `sampling_hat.py`
**Description:** Performs hydrogen atom transfer (HAT) sampling using the radical systems as input, producing new coordinates and energies.
**Output:** Results are written to `data_generation/samples/aa_test_nms_hat/`.
