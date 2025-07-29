import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.callbacks
import keras as ks
import os

def R_squared(y, y_pred):
    '''
    R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.
    '''
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2





'''
def plot_train_test_loss(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.

    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.

    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]
    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]
    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]
    if not isinstance(data_unit, list):
        data_unit = [data_unit]

    if len(data_unit) < len(val_loss_name):
        data_unit = data_unit + [str(data_unit[-1])]*(len(val_loss_name)-len(data_unit))

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    for i, y in enumerate(val_loss):
        val_step = len(train_loss[i][0]) / len(val_loss[i][0])
        vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                      label=val_loss_name[i])
        plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                         np.mean(y, axis=0) - np.std(y, axis=0),
                         np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
        plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                    label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                        val_loss_name[i], np.mean(y, axis=0)[-1],
                        np.std(y, axis=0)[-1]) + data_unit[i], color=vp[0].get_color()
                    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("%s training curve for %s" % (dataset_name, model_name))
    plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig


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
'''

def plot_en_err_size(predicted_y, true_y, x_test, fold_i, model_name: str = "", filepath: str = None, file_name: str = "", dataset_name: str = "", figsize: list = None, dpi: float = None, show_fig: bool = True):
    
    atom_num_padded = x_test[0]
    system_sizes = []

    for i in range(len(atom_num_padded)):
        atom_num_true = []
        for atom_num_i in atom_num_padded[i]:
            if atom_num_i != 0:
                atom_num_true.append(atom_num_i)
            else:
                continue

        system_sizes.append(len(atom_num_true))
    
    
    er_en_all_i = []
    er_en_all_sys_i = []
    
    for i in range(len(predicted_y[0])):
        #predicted
        en_i = predicted_y[0][i]
        


        er_en = abs(en_i - true_y[0][i])
        
        er_en_sys = abs(en_i - true_y[0][i])/system_sizes[i]
        
        er_en_all_i.append(er_en)
        er_en_all_sys_i.append(er_en_sys)
        

    # plot
    fig, ax = plt.subplots()
    #labels = ['XTB', 'ML Model']
    colors_i = ['royalblue','darkorange']
    
    #print(len(system_sizes), system_sizes[0])
    #print(len(er_en_all_i))
    ax.scatter(system_sizes, er_en_all_i, c = colors_i[0], alpha = 0.4) #label = labels[0]
    ax.set_xlabel(r'System Size (Number of Atoms per Molecule)')
    ax.set_ylabel(r'E Error/System [eV]')
    #ax.legend()
    #ax.set_title('Prediction Differences {}'.format(system_name))
    fig.tight_layout()
    plt.savefig('{}/en_er_sys_fold_{}.png'.format(filepath, fold_i), dpi = 300)
    
    fig, ax = plt.subplots()
    ax.scatter(system_sizes, er_en_all_sys_i, c = colors_i[0], alpha = 0.4) #label = labels[0]
    ax.set_xlabel(r'System Size (Number of Atoms per Molecule)')
    ax.set_ylabel(r'E Error/Atom [eV]')
    #ax.legend()
    #ax.set_title('Prediction Differences {}'.format(system_name))
    fig.tight_layout()
    plt.savefig('{}/en_er_atom_fold_{}.png'.format(filepath, fold_i), dpi = 300)
    
    ## log
    fig, ax = plt.subplots()
    #labels = ['XTB', 'ML Model']
    colors_i = ['royalblue','darkorange']
    
    #print(len(system_sizes), system_sizes[0])
    #print(len(er_en_all_i))
    ax.scatter(system_sizes, er_en_all_i, c = colors_i[0], alpha = 0.4) #label = labels[0]
    ax.set_yscale('log')
    ax.set_xlabel(r'System Size (Number of Atoms per Molecule)')
    ax.set_ylabel(r'E Error/System [eV]')
    #ax.legend()
    #ax.set_title('Prediction Differences {}'.format(system_name))
    fig.tight_layout()
    plt.savefig('{}/en_er_sys_log_fold_{}.png'.format(filepath, fold_i), dpi = 300)
    
    fig, ax = plt.subplots()
    ax.scatter(system_sizes, er_en_all_sys_i, c = colors_i[0], alpha = 0.4) #label = labels[0]
    ax.set_yscale('log')
    ax.set_xlabel(r'System Size (Number of Atoms per Molecule)')
    ax.set_ylabel(r'E Error/Atom [eV]')
    #ax.legend()
    #ax.set_title('Prediction Differences {}'.format(system_name))
    fig.tight_layout()
    plt.savefig('{}/en_er_atom_log_fold_{}.png'.format(filepath, fold_i), dpi = 300)

    
    
    '''
    fig, ax = plt.subplots()
    ax.scatter(system_sizes, er_force_all_i, c = colors_i[1]) #label = labels[0]
    ax.set_xlabel(r'System Size (Number of Atoms per Molecule)')
    ax.set_ylabel(r'F Error/Atom [eV/$\AA$]')
    #ax.legend()
    #ax.set_title('Prediction Differences {}'.format(system_name))
    fig.tight_layout()
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_force_er_atom_fold"+ str(fold_i) ), bbox_inches='tight')
    if show_fig:
        plt.show()
    #plt.savefig('{}/force_er_atom_fold_{}.png'.format(filepath, fold_i), dpi = 300)
    '''

def plot_r_squared(r2_dict, model_name: str = "", filepath: str = None, file_name: str = "", dataset_name: str = "", figsize: list = None, dpi: float = None, show_fig: bool = True):

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(dpi=dpi) #figsize=figsize, 

    for key in r2_dict.keys():       
        vp = plt.plot(np.arange(len(np.mean(r2_dict[key], axis=0))), np.mean(r2_dict[key], axis=0), alpha=0.85, label=key)
        plt.fill_between(np.arange(len(np.mean(r2_dict[key], axis=0))),
                         np.mean(r2_dict[key], axis=0) - np.std(r2_dict[key], axis=0),
                         np.mean(r2_dict[key], axis=0) + np.std(r2_dict[key], axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    plt.xlabel('Epochs')
    plt.ylabel(r'$R^{2}$')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_loss(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]
    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(dpi=dpi) #figsize=figsize, 
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    for i, y in enumerate(val_loss):
        val_step = len(train_loss[i][0]) / len(val_loss[i][0])
        vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                      label=val_loss_name[i])
        plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                         np.mean(y, axis=0) - np.std(y, axis=0),
                         np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
        plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                    label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                        val_loss_name[i], np.mean(y, axis=0)[-1],
                        np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig


def plot_train_test_mae_val(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    '''
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    '''
    for i, y in enumerate(val_loss):
        if val_loss_name[i] == 'val_energy_scaled_mean_absolute_error': #'val_energy_scaled_mean_absolute_error'
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                          label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                             np.mean(y, axis=0) - np.std(y, axis=0),
                             np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1],
                            np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                        )
        if val_loss_name[i] == 'val_force_scaled_mean_absolute_error':
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                          label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                             np.mean(y, axis=0) - np.std(y, axis=0),
                             np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1],
                            np.std(y, axis=0)[-1]) + '$eV/ \AA$', color=vp[0].get_color()
                        )            
        else:
            continue
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

# type(history_list[0].history) == dict

def plot_train_test_lr(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    
    for i, x in enumerate(train_loss):
        if loss_name[i] == 'learning_rate':
            vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
            plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                             np.mean(x, axis=0) - np.std(x, axis=0),
                             np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                             )


    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_val_loss_all(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    '''
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    '''
    for i, y in enumerate(val_loss):
        if val_loss_name[i] == 'val_loss' or val_loss_name[i] == 'val_energy_scaled_mean_absolute_error' or val_loss_name[i] ==  'val_force_scaled_mean_absolute_error': #'val_energy_scaled_mean_absolute_error'
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                          label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                             np.mean(y, axis=0) - np.std(y, axis=0),
                             np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1],
                            np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                        )
        else:
            continue
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_val_loss_eg(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    '''
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    '''
    for i, y in enumerate(val_loss):
        if val_loss_name[i] == 'val_energy_scaled_mean_absolute_error' or val_loss_name[i] == 'val_force_scaled_mean_absolute_error': #'val_energy_scaled_mean_absolute_error'
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                          label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                             np.mean(y, axis=0) - np.std(y, axis=0),
                             np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1],
                            np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                        )
        else:
            continue
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_val_loss(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    '''
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    '''
    for i, y in enumerate(val_loss):
        if val_loss_name[i] == 'val_loss': #'val_energy_scaled_mean_absolute_error'
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                          label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                             np.mean(y, axis=0) - np.std(y, axis=0),
                             np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1],
                            np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                        )
        else:
            continue
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_train_loss_all(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    
    for i, x in enumerate(train_loss):
        if loss_name[i] == 'loss' or loss_name[i] == 'energy_scaled_mean_absolute_error' or loss_name[i] == 'force_scaled_mean_absolute_error':
            vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
            plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                             np.mean(x, axis=0) - np.std(x, axis=0),
                             np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                             )


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_train_loss_eg(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    
    for i, x in enumerate(train_loss):
        if loss_name[i] == 'energy_loss' or loss_name[i] == 'force_loss':
            vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
            plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                             np.mean(x, axis=0) - np.std(x, axis=0),
                             np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
        else:
            continue


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig

def plot_train_test_loss_all(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.
    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.
    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    histories = [hist.history if isinstance(hist, ks.callbacks.History) else hist for hist in histories]

    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    
    for i, x in enumerate(train_loss):
        if loss_name[i] == 'loss':
            vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
            plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                             np.mean(x, axis=0) - np.std(x, axis=0),
                             np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
        else:
            continue

    for i, y in enumerate(val_loss):
        if val_loss_name[i] == 'val_loss': #'val_energy_scaled_mean_absolute_error'
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                          label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                             np.mean(y, axis=0) - np.std(y, axis=0),
                             np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                             )
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1],
                            np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                        )
        else:
            continue
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(bbox_to_anchor = (1.04, 1), loc='upper left')
    #plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name), bbox_inches='tight')
    if show_fig:
        plt.show()
    return fig
