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
- `xtb` (with environment variable `XTBPATH`) including `crest` for conformer generation ([text](https://xtb-docs.readthedocs.io/en/latest/index.html))
- Python 3.8+
- RDKit
- `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`, `pyyaml`