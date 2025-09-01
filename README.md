# hat_pes_learning
Data generation scripts and ML potential training scripts for the paper 'Learning Potential Energy Surfaces of Hydrogen Atom Transfer Reactions in Peptides'.

# Data Generation for Hydrogen Atom Transfer in Peptides

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
- **Script:** `sampling_nms.py`
- **Description:** Generates normal mode samples (coordinates, energies, forces) for molecules.
- **Output:** Results written to `data_generation/samples/aa_test_nms`.
### 2. Radical Sampling
- **Script:** `sampling_rad_sys.py`
- **Description:** Uses NMS samples as input, generates intra- and inter-molecular radical systems.
- **Output:** Results are written to `data_generation/samples/aa_test_nms_radicals/`.
### 3. HAT Sampling
- **Script:** `sampling_hat.py`
- **Description:** Performs hydrogen atom transfer (HAT) sampling using the radical systems as input, producing new coordinates and energies.
- **Output:** Results are written to `data_generation/samples/aa_test_nms_hat/`.

# ML Potentials for Hydrogen Atom Transfer in Peptides

This repository provides configuration files and evaluation scripts for benchmarking machine learning potentials on hydrogen atom transfer (HAT) reactions in peptides. For each model (SchNet, Allegro, MACE), we include the configuration used for training, as well as scripts for model evaluation and HAT reaction barrier calculation.

**Note:**  
To run the training and evaluation of the models, you will need to install the respective machine learning potential packages from their official repositories. Please refer to the documentation of each package for installation instructions.

Trained model weights and the full datasets are provided on Zenodo and can be downloaded [here](https://doi.org/10.5281/zenodo.16572631).

### Evaluate the performance of a trained MACE or Allegro model on hydrogen atom transfer datasets.
`evaluate_mace.py` / `evaluate_allegro.py`  

- Computes energy and force errors (MAE, R2, normalized errors)  
- Calculates HAT reaction barriers from predicted and reference data  
- Produces parity plots and error histograms  

Usage:
`python evaluate_mace.py --model mace --eval_name dft_ID4 --pred_xyz path/to/mace_predictions.extxyz --ref_xyz path/to/dft_reference.extxyz`    

---

If you use this code or data, please cite:

- Neubert, M., Gräter, F., Friederich, P. (2025). [Learning Potential Energy Surfaces of Hydrogen Atom Transfer Reactions in Peptides](http://arxiv.org/abs/2508.00578).
- [Zenodo dataset](https://doi.org/10.5281/zenodo.16572631).




