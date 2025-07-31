# hat_pes_learning
Data generation scripts and ML potential training scripts for the paper 'Learning Potential Energy Surfaces of Hydrogen Atom Transfer Reactions in Peptides'.

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
example_scripts/

### Dependencies

Make sure the following tools and packages are available:
- `xtb` (with environment variable `XTBPATH`) including `crest` for conformer generation ([xtb](https://xtb-docs.readthedocs.io/en/latest/index.html))
- Python 3.8+
- RDKit
- `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`, `pyyaml`

# ML Potentials for Hydrogen Atom Transfer in Peptides

This repository provides configuration files and evaluation scripts for benchmarking machine learning potentials on hydrogen atom transfer (HAT) reactions in peptides. For each model (SchNet, Allegro, MACE), we include the configuration used for training, as well as scripts for model evaluation and HAT reaction barrier calculation.

**Note:**  
To run the training and evaluation scripts, you will need to install the respective machine learning potential packages from their official repositories. Please refer to the documentation of each package for installation instructions.

Trained model weights and the full datasets provided on Zenodo and can be downloaded [here](https://doi.org/10.5281/zenodo.16572631).

---

If you use this code or data, please cite:

- Neubert, M., Gräter, F., Friederich, P. (2025). [Learning Potential Energy Surfaces of Hydrogen Atom Transfer Reactions in Peptides](add link).
- [Zenodo dataset](https://doi.org/10.5281/zenodo.16572631).

For questions, contact .


