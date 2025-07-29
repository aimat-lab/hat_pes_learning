hyper= {
    "Schnet.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "SchnetEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                            {"shape": (), "name": "total_nodes", "dtype": "int64"},
                            {"shape": (), "name": "total_ranges", "dtype": "int64"}
                        ],
                        "cast_disjoint_kwargs": {"padded_disjoint": False},
                        "input_node_embedding": {"input_dim": 100, "output_dim": 128},
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                     "activation": [{"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                                    {"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                                    'linear']},
                        "interaction_args": {
                            "units": 128, "use_bias": True, "activation": {"class_name": "function",
                                                                           "config": 'kgcnn>shifted_softplus'},
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_sum"},
                        "depth": 6,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3)}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 250}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 1.0, "force": 49.0}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "dataset": {
            "class_name": "ForceDataset",
            "module_name": "kgcnn.data.force",
            "config": {
                "data_directory": "./my_data/dft_IDs_gcnn/dft_ID2/",
                "dataset_name": "dft_ID2",
                "file_name": "dft_ID2_gcnn.csv",
                "file_directory": None,
                "file_name_xyz": "dft_ID2_coords.xyz",
                "file_name_mol": None,
                "file_name_force_xyz": "dft_ID2_forces.xyz"
            },
            "methods": [
                {"prepare_data": {"make_sdf": False}},
                {"read_in_memory": {"label_column_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "graph_labels",
                                               "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "node_number",
                                               "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "dft_ID2_forces.xyz",
                                               "new_property_name": "force"}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "dft_ID2",
            "postfix_file": "dft_ID2",
            "kgcnn_version": "4.0.1"
        }
    },
    "PAiNN.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "PAiNN",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.PAiNN",
                    "config": {
                        "name": "PAiNNEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                            {"shape": (), "name": "total_nodes", "dtype": "int64"},
                            {"shape": (), "name": "total_ranges", "dtype": "int64"}
                        ],
                        "input_embedding": None,
                        "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                        # If unstable change to other initialize methods.
                        "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
                        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                        "pooling_args": {"pooling_method": "scatter_sum"},
                        "conv_args": {"units": 128, "cutoff": None},
                        # If unstable set to True to add an eps to norm.
                        "update_args": {"units": 128, "add_eps": False},
                        "depth": 3,
                        "verbose": 10,
                        "output_embedding": "graph",
                        "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3)}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam", "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                "learning_rate": 0.001, "warmup_steps": 150.0, "decay_steps": 20000.0,
                                "decay_rate": 0.01
                            }
                        }, "amsgrad": True, "use_ema": True
                    }
                },
                "loss_weights": {"energy": 1.0, "force": 49.0}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "dataset": {
            "class_name": "ForceDataset",
            "module_name": "kgcnn.data.force",
            "config": {
                "data_directory": "./my_data/hat_train_ID_9/",
                "dataset_name": "hat_train_se1_ID_9",
                "file_name": "hat_train_se1_ID_9_info.csv",
                "file_directory": None,
                "file_name_xyz": "hat_train_se1_ID_9_coords.xyz",
                "file_name_mol": None,
                "file_name_force_xyz": "hat_train_se1_ID_9_forces.xyz"
            },
            "methods": [
                {"prepare_data": {"make_sdf": False}},
                {"read_in_memory": {"label_column_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "graph_labels",
                                               "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "node_number",
                                               "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "hat_train_se1_ID_9_forces.xyz",
                                               "new_property_name": "force"}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "postfix_file": "_test_0",
            "kgcnn_version": "4.0.1"
        }
    },
    "SchnetRagged.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "SchnetEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32", "ragged": True},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                        ],
                        "input_tensor_type": "ragged",
                        "cast_disjoint_kwargs": {},
                        "input_node_embedding": {"input_dim": 100, "output_dim": 128},
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                     "activation": [
                                         {"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                         {"class_name": "function", "config": 'kgcnn>shifted_softplus'}, 'linear']},
                        "interaction_args": {
                            "units": 128, "use_bias": True,
                            "activation": {"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_sum"},
                        "depth": 6,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3), "ragged": True}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 250}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 1.0, "force": 49.0},
                "loss": {
                    "energy": "mean_absolute_error",
                    "force": {"class_name": "kgcnn>RaggedValuesMeanAbsoluteError", "config": {}}
                }
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "dataset": {
            "class_name": "ForceDataset",
            "module_name": "kgcnn.data.force",
            "config": {
                "data_directory": "./data/",
                "dataset_name": "hat_train_se3_ID_4",
                "file_name": "hat_train_se3_ID_4_info.csv",
                "file_directory": None,
                "file_name_xyz": "hat_train_se3_ID_4_coords.xyz",
                "file_name_mol": None,
                "file_name_force_xyz": "hat_train_se3_ID_4_forces.xyz"
            },
            "methods": [
                {"prepare_data": {"make_sdf": False}},
                {"read_in_memory": {"label_column_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "graph_labels",
                                               "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "node_number",
                                               "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "hat_train_se3_ID_4_forces.xyz",
                                               "new_property_name": "force"}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.1"
        }
    },
    "PAiNNRagged.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "PAiNN",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32", "ragged": True},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.PAiNN",
                    "config": {
                        "name": "PAiNNEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32", "ragged": True},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                        ],
                        "input_tensor_type": "ragged",
                        "input_embedding": None,
                        "input_node_embedding": {"input_dim": 95, "output_dim": 128},
                        # If unstable change to other initialize methods.
                        "equiv_initialize_kwargs": {"dim": 3, "method": "eps"},
                        "bessel_basis": {"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
                        "pooling_args": {"pooling_method": "scatter_sum"},
                        "conv_args": {"units": 128, "cutoff": None},
                        # If unstable set to True to add an eps to norm.
                        "update_args": {"units": 128, "add_eps": False},
                        "depth": 3,
                        "verbose": 10,
                        "output_embedding": "graph",
                        "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3), "ragged": True}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": []
            },
            "compile": {
                "optimizer": {
                    "class_name": "Adam", "config": {
                        "learning_rate": {
                            "class_name": "kgcnn>LinearWarmupExponentialDecay", "config": {
                                "learning_rate": 0.001, "warmup_steps": 150.0, "decay_steps": 20000.0,
                                "decay_rate": 0.01
                            }
                        }, "amsgrad": True, "use_ema": True
                    }
                },
                "loss_weights": {"energy": 1.0, "force": 49.0},
                "loss": {
                    "energy": "mean_absolute_error",
                    "force": {"class_name": "kgcnn>RaggedValuesMeanAbsoluteError", "config": {}}
                }
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "dataset": {
            "class_name": "ForceDataset",
            "module_name": "kgcnn.data.force",
            "config": {
                "data_directory": "./data/",
                "dataset_name": "hat_train_se3_ID_4",
                "file_name": "hat_train_se3_ID_4_info.csv",
                "file_directory": None,
                "file_name_xyz": "hat_train_se3_ID_4_coords.xyz",
                "file_name_mol": None,
                "file_name_force_xyz": "hat_train_se3_ID_4_forces.xyz"
            },
            "methods": [
                {"prepare_data": {"make_sdf": False}},
                {"read_in_memory": {"label_column_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "graph_labels",
                                               "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "node_number",
                                               "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "hat_train_se3_ID_4_forces.xyz",
                                               "new_property_name": "force"}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.1"
        }
    },
    "SchnetPadded.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                    {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "total_nodes", "dtype": "int64"},
                    {"shape": (), "name": "total_ranges", "dtype": "int64"}
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "SchnetEnergy",
                        "inputs": [
                            {"shape": [None], "name": "atomic_number", "dtype": "int32"},
                            {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32"},
                            {"shape": [None, 2], "name": "range_indices", "dtype": "int64"},
                            {"shape": (), "name": "total_nodes", "dtype": "int64"},
                            {"shape": (), "name": "total_ranges", "dtype": "int64"}
                        ],
                        "cast_disjoint_kwargs": {"padded_disjoint": True},
                        "input_node_embedding": {"input_dim": 100, "output_dim": 128},
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                     "activation": [{"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                                    {"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                                    'linear']},
                        "interaction_args": {
                            "units": 128, "use_bias": True, "activation": {"class_name": "function",
                                                                           "config": 'kgcnn>shifted_softplus'},
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_sum"},
                        "depth": 6,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,)},
                            "force": {"name": "force", "shape": (None, 3)}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 250}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 1.0, "force": 49.0}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "dataset": {
            "class_name": "ForceDataset",
            "module_name": "kgcnn.data.force",
            "config": {
                "data_directory": "./data/",
                "dataset_name": "hat_train_se3_ID_4",
                "file_name": "hat_train_se3_ID_4_info.csv",
                "file_directory": None,
                "file_name_xyz": "hat_train_se3_ID_4_coords.xyz",
                "file_name_mol": None,
                "file_name_force_xyz": "hat_train_se3_ID_4_forces.xyz"
            },
            "methods": [
                {"prepare_data": {"make_sdf": False}},
                {"read_in_memory": {"label_column_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "graph_labels",
                                               "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "node_number",
                                               "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "hat_train_se3_ID_4_forces.xyz",
                                               "new_property_name": "force"}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.1"
        }
    },
    "SchnetDisjoint.EnergyForceModel": {
        "model": {
            "class_name": "EnergyForceModel",
            "module_name": "kgcnn.models.force",
            "config": {
                "name": "Schnet",
                "nested_model_config": True,
                "output_to_tensor": False,
                "output_squeeze_states": True,
                "coordinate_input": 1,
                "inputs": [
                    {"shape": (), "name": "atomic_number", "dtype": "int32"},
                    {"shape": [3], "name": "node_coordinates", "dtype": "float32"},
                    {"shape": [None], "name": "range_indices", "dtype": "int64"},
                    {"shape": (), "name": "batch_id_node", "dtype": "int64"},
                    {"shape": (), "name": "batch_id_edge", "dtype": "int64"},
                    {"shape": (), "name": "node_id", "dtype": "int64"},
                    {"shape": (), "name": "edge_id", "dtype": "int64"},
                    {"shape": (), "name": "count_nodes", "dtype": "int64"},
                    {"shape": (), "name": "count_edges", "dtype": "int64"},
                ],
                "model_energy": {
                    "class_name": "make_model",
                    "module_name": "kgcnn.literature.Schnet",
                    "config": {
                        "name": "SchnetEnergy",
                        "inputs": [
                            {"shape": (), "name": "atomic_number", "dtype": "int32"},
                            {"shape": [3], "name": "node_coordinates", "dtype": "float32"},
                            {"shape": [None], "name": "range_indices", "dtype": "int64"},
                            {"shape": (), "name": "batch_id_node", "dtype": "int64"},
                            {"shape": (), "name": "batch_id_edge", "dtype": "int64"},
                            {"shape": (), "name": "node_id", "dtype": "int64"},
                            {"shape": (), "name": "edge_id", "dtype": "int64"},
                            {"shape": (), "name": "count_nodes", "dtype": "int64"},
                            {"shape": (), "name": "count_edges", "dtype": "int64"},
                        ],
                        "input_tensor_type": "disjoint",
                        "output_tensor_type": "disjoint",
                        "cast_disjoint_kwargs": {"padded_disjoint": True},
                        "input_node_embedding": {"input_dim": 100, "output_dim": 128},
                        "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                                     "activation": [{"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                                    {"class_name": "function", "config": 'kgcnn>shifted_softplus'},
                                                    'linear']},
                        "interaction_args": {
                            "units": 128, "use_bias": True, "activation": {"class_name": "function",
                                                                           "config": 'kgcnn>shifted_softplus'},
                            "cfconv_pool": "scatter_sum"
                        },
                        "node_pooling_args": {"pooling_method": "scatter_sum"},
                        "depth": 6,
                        "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
                        "output_embedding": "graph",
                        "use_output_mlp": False,
                        "output_mlp": None,
                    }
                },
                "outputs": {"energy": {"name": "energy", "shape": (1,), "dtype": "float32"},
                            "force": {"name": "force", "shape": (3,), "dtype": "float32"}}
            }
        },
        "training": {
            "fit": {
                "batch_size": 32, "epochs": 1000, "validation_freq": 1, "verbose": 2,
                "callbacks": [
                    {"class_name": "kgcnn>LinearWarmupExponentialLRScheduler", "config": {
                        "lr_start": 1e-03, "gamma": 0.995, "epo_warmup": 1, "verbose": 1, "steps_per_epoch": 250}}
                ]
            },
            "compile": {
                "optimizer": {"class_name": "Adam", "config": {"learning_rate": 1e-03}},
                "loss_weights": {"energy": 1.0, "force": 49.0}
            },
            "scaler": {"class_name": "EnergyForceExtensiveLabelScaler",
                       "config": {"standardize_scale": False}},
        },
        "dataset": {
            "class_name": "ForceDataset",
            "module_name": "kgcnn.data.force",
            "config": {
                "data_directory": "./data/",
                "dataset_name": "hat_train_se3_ID_4",
                "file_name": "hat_train_se3_ID_4_info.csv",
                "file_directory": None,
                "file_name_xyz": "hat_train_se3_ID_4_coords.xyz",
                "file_name_mol": None,
                "file_name_force_xyz": "hat_train_se3_ID_4_forces.xyz"
            },
            "methods": [
                {"prepare_data": {"make_sdf": False}},
                {"read_in_memory": {"label_column_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "graph_labels",
                                               "new_property_name": "energy"}},
                {"rename_property_on_graphs": {"old_property_name": "node_number",
                                               "new_property_name": "atomic_number"}},
                {"rename_property_on_graphs": {"old_property_name": "hat_train_se3_ID_4_forces.xyz",
                                               "new_property_name": "force"}},
                {"set_train_test_indices_k_fold": {"n_splits": 5, "random_state": 42, "shuffle": True}},
                {"map_list": {"method": "set_range", "max_distance": 5, "max_neighbours": 10000,
                              "node_coordinates": "node_coordinates"}},
                {"map_list": {"method": "count_nodes_and_edges", "total_edges": "total_ranges",
                              "count_edges": "range_indices", "count_nodes": "atomic_number",
                              "total_nodes": "total_nodes"}},
            ]
        },
        "data": {
        },
        "info": {
            "postfix": "",
            "postfix_file": "",
            "kgcnn_version": "4.0.1"
        }
    },
}
