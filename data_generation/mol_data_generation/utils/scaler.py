#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np




class ExtensiveEnergyForceScaler():
    """Based on KGCNN ExtensiveMolecularScaler
    """    
    
    _attributes_list_sklearn = ["coef_", "intercept_", "n_iter_", "n_features_in_", "feature_names_in_"]
    _attributes_list_mol = ["scale_", "_fit_atom_selection", "_fit_atom_selection_mask"]
    max_atomic_number = 95
    
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
    inverse_global_proton_dict = {value: key for key, value in global_proton_dict.items()}
    
    def __init__(self, alpha: float = 1e-9, fit_intercept: bool = False,use_x_std=False, **kwargs):

        
        #super(ExtensiveEnergyForceScaler, self).__init__()
        self.ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)

        self._fit_atom_selection_mask = None
        self._fit_atom_selection = None
        self.scale_ = None
        self.use_x_std = use_x_std
        
        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        #self.energy_mean = np.zeros((1, 1))
        #self.energy_std = np.ones((1, 1))
        self.gradient_mean = np.zeros((1, 1, 1, 1))
        self.gradient_std = np.ones((1, 1, 1, 1))

        #self._encountered_y_shape = [None, None]
        #self._encountered_y_std = [None, None]
        #self.scaler_module = scaler_module
        
    def convert_elements_to_atn(self, elements_all):
        
        elements_all_ar = []
        for i in range(len(elements_all)):
            elements_all_ar.append(np.array(elements_all[i]))

        # need to transform element types into integers
        atomic_numbers = [np.array([self.global_proton_dict[atom] for atom in x]) for x in elements_all_ar]
        
        return atomic_numbers
        
    
    def fit(self, atomic_number, y,coords= None, sample_weight=None):
        r"""Fit atomic number to the molecular properties.
        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, #atoms)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            
            coords : 
            y : (energy, force)
            
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.
        Returns:
            self
        """
        
        
        if isinstance(y, list):
            y0 = y[0]
            y1 = y[1]
        elif isinstance(y, dict):
            y0 = y["energy"]
            y1 = y["force"]
        else:
            raise ValueError("Transform for expected [energy, force] but got %s" % y)
        
        npeps = np.finfo(float).eps
        if self.use_x_std:
            self.x_std = np.std(coords) + npeps
        
        if len(atomic_number) != len(y0):
            raise ValueError(
                "`ExtensiveMolecularScaler` different input shape {0} vs. {1}".format(
                    len(atomic_number), len(y0))
            )

        unique_number = [np.unique(x, return_counts=True) for x in atomic_number]
        all_unique = np.unique(np.concatenate([x[0] for x in unique_number], axis=0))
        
        self._fit_atom_selection = all_unique
        atom_mask = np.zeros(self.max_atomic_number, dtype="bool")
        atom_mask[all_unique] = True
        self._fit_atom_selection_mask = atom_mask
        total_number = []
        for unique_per_mol, num_unique in unique_number:
            array_atoms = np.zeros(self.max_atomic_number)
            array_atoms[unique_per_mol] = num_unique
            positives = array_atoms[atom_mask]
            total_number.append(positives)
        total_number = np.array(total_number)
        self.ridge.fit(total_number, y0, sample_weight=sample_weight)
        diff = y0 - self.ridge.predict(total_number)
        self.scale_ = np.std(diff, axis=0, keepdims=True) #
        
        self.gradient_std = np.expand_dims(np.expand_dims(self.scale_, axis=-1), axis=-1) / self.x_std + npeps
        self.gradient_mean = np.zeros_like(self.gradient_std, dtype=np.float32)  # no mean shift expected

        #self._encountered_y_shape = [np.array(y0.shape), np.array(y1.shape)]
        #self._encountered_y_std = [np.std(y0, axis=0), np.std(y1, axis=(0, 2, 3))]
        
        return self        
        
    def predict(self, atomic_number):
        """Predict the offset form atomic numbers. Requires :obj:`fit()` called previously.
        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
        Returns:
            np.ndarray: Offset of atomic properties fitted previously. Shape is `(n_samples, n_properties)`.
        """
        if self._fit_atom_selection_mask is None:
            raise ValueError("ERROR: `ExtensiveMolecularScaler` has not been fitted yet. Can not predict.")
        unique_number = [np.unique(x, return_counts=True) for x in atomic_number]
        total_number = []
        for unique_per_mol, num_unique in unique_number:
            array_atoms = np.zeros(self.max_atomic_number)
            array_atoms[unique_per_mol] = num_unique
            positives = array_atoms[self._fit_atom_selection_mask]
            if np.sum(positives) != np.sum(num_unique):
                print("`ExtensiveMolecularScaler` got unknown atom species in transform.")
            total_number.append(positives)
        total_number = np.array(total_number)
        offset = self.ridge.predict(total_number)
        return offset        
        
    def _plot_predict(self, atomic_number, y ,outdir, sample_name):
        """Debug function to check prediction."""
        
        if isinstance(y, list):
            y0 = y[0]
            #y1 = y[1]
        elif isinstance(y, dict):
            y0 = y["energy"]
            #y1 = y["force"]
        else:
            raise ValueError("Transform for expected [energy, force] but got %s" % y)
        
        molecular_property = np.array(y0)
        if len(molecular_property.shape) <= 1:
            molecular_property = np.expand_dims(molecular_property, axis=-1)
        predict_prop = self.predict(atomic_number)
        if len(predict_prop.shape) <= 1:
            predict_prop = np.expand_dims(predict_prop, axis=-1)
        mae = np.mean(np.abs(molecular_property - predict_prop), axis=0)
        plt.figure()
        for i in range(predict_prop.shape[-1]):
            plt.scatter(predict_prop[:, i], molecular_property[:, i], alpha=0.3,
                        label="Pos: " + str(i) + " MAE: {0:0.4f} ".format(mae[i]))
        plt.plot(np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05),
                 np.arange(np.amin(molecular_property), np.amax(molecular_property), 0.05), color='red')
        plt.xlabel('Fitted')
        plt.ylabel('Actual')
        plt.legend(loc='upper left', fontsize='x-small')
        plt.savefig('{}/scaler_predict_E_{}.png'.format(outdir, sample_name), dpi = 300)
        #plt.show()        
        
    def transform(self, atomic_number, y, coords=None):
        """Transform any atomic number list with matching properties based on previous fit. Also std-scaled.
        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        x_res = coords
        y_res = y
        if coords is not None:
            if self.use_x_std:
                x_res = x_res / self.x_std
        if y is not None:
            if isinstance(y, list):
                energy = y[0]
                gradient = y[1]
            elif isinstance(y, dict):
                energy = y["energy"]
                gradient = y["force"]
            else:
                raise ValueError("Transform for expected [energy, force] but got %s" % y)
                
            grads_out_all = [] #
            #out_e = (energy - self.predict(atomic_number)) / self.scale_
            out_e =  (energy - self.predict(atomic_number)) / np.expand_dims(self.scale_, axis=0)
            #out_g = gradient / self.gradient_std
            #out_g = [gradient / np.expand_dims(self.scale_, axis=0) for f in gradient]
            for i in range(len(gradient)):
                out_g = gradient[i]/self.gradient_std
                grads_out_all.append(out_g)
            #y_res = [out_e, grads_out_all]
                
        
        return x_res, out_e, grads_out_all #out_g
        
    def fit_transform(self,  atomic_number, y,coords = None, sample_weight=None):
        """Combine fit and transform methods in one call.
        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`.
            sample_weight: Sample weights `(n_samples,)` directly passed to :obj:`Ridge()`. Default is None.
        Returns:
            np.ndarray: Transformed atomic properties fitted. Shape is `(n_samples, n_properties)`.
        """
        
        self.fit(atomic_number, y,coords, sample_weight)
        return self.transform(atomic_number, y,coords)      
    
    
    def inverse_transform(self, atomic_number, y,coords= None):
        """Reverse the transform method to original properties without offset and scaled to original units.
        Args:
            atomic_number (list): List of array of atomic numbers. Shape is `(n_samples, <#atoms>)`.
            molecular_property (np.ndarray): Array of atomic properties of shape `(n_samples, n_properties)`
        Returns:
            np.ndarray: Original atomic properties. Shape is `(n_samples, n_properties)`.
        """
        x_res = coords
        y_res = y
        if coords is not None:
            if self.use_x_std:
                x_res = x_res * self.x_std

        if y is not None:
            if isinstance(y, list):
                energy = y[0]
                gradient = y[1]
            elif isinstance(y, dict):
                energy = y["energy"]
                gradient = y["force"]
            else:
                raise ValueError("Transform for expected [energy, force] but got %s" % y)
                
            out_e = energy * self.scale_ + self.predict(atomic_number)
            ## change list
            out_g = gradient * self.gradient_std
            y_res = [out_e, out_g]
        
        return x_res, y_res
        
    def get_config(self):
        """Get configuration for scaler."""
        outdict = {
            "scaler_module": self.scaler_module,
            "ridge_params": self.ridge.get_params(),
            #"use_energy_mean": self.use_energy_mean,
            "use_x_std": self.use_x_std,
            #"use_x_mean": self.use_x_mean,
        }
        
        return outdict 
    
    def get_weights(self):
        weights = dict()
        for x in self._attributes_list_mol:
            weights.update({x: np.array(getattr(self, x))})
        for x in self._attributes_list_sklearn:
            if hasattr(self.ridge, x):
                weights.update({x: np.array(getattr(self.ridge, x))})
        return weights
    
    def save_weights(self, file_path):
        out_dict = self.get_weights()
        out_dict['energy_std'] = self.scale_
        
        
        np.save(file_path, out_dict)
    
    '''
    def save_weights(self, file_path):
        out_dict = {
            #'x_mean': self.x_mean,
            'x_std': self.x_std,
            #'energy_mean': self.energy_mean,
            'energy_std': self.energy_std,
            'gradient_mean': self.gradient_mean,
            'gradient_std': self.gradient_std
        }
        np.save(file_path, out_dict)
    
    def load_weights(self, file_path):
        indict = np.load(file_path, allow_pickle=True).item()
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])
    
    def print_params_info(self):
        print("Info: Total-Data gradient std", self._encountered_y_shape[1], ":", self._encountered_y_std[1])
        print("Info: Total-Data energy std", self._encountered_y_shape[0], ":", self._encountered_y_std[0])
        print("Info: Using energy-std", self.energy_std.shape, ":", self.energy_std[0])
        print("Info: Using energy-mean", self.energy_mean.shape, ":", self.energy_mean[0])
        print("Info: Using gradient-std", self.gradient_std.shape, ":", self.gradient_std[0, :, 0, 0])
        print("Info: Using gradient-mean", self.gradient_mean.shape, ":", self.gradient_mean[0, :, 0, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)
    '''
