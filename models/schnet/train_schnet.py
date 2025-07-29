import numpy as np
import time
import os
import argparse
import keras as ks
from datetime import timedelta
import kgcnn.training.schedule
import kgcnn.training.scheduler
from kgcnn.data.utils import save_pickle_file
from kgcnn.data.transform.scaler.serial import deserialize as deserialize_scaler
from kgcnn.utils.devices import check_device, set_cuda_device
from kgcnn.training.history import save_history_score, load_history_list, load_time_list
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.models.serial import deserialize as deserialize_model
from kgcnn.data.serial import deserialize as deserialize_dataset
from kgcnn.training.hyper import HyperParameter
from kgcnn.losses.losses import ForceMeanAbsoluteError
from kgcnn.metrics.metrics import ScaledMeanAbsoluteError, ScaledForceMeanAbsoluteError
from kgcnn.data.transform.scaler.force import EnergyForceExtensiveLabelScaler

import additional_plots as add_plots

#import jax


#jax.config.update("jax_enable_x64", True)

# Input arguments from command line.
parser = argparse.ArgumentParser(description='Train a GNN on an Energy-Force Dataset.')
parser.add_argument("--hyper", required=False, help="Filepath to hyper-parameter config file (.py or .json).",
                    default="hyper_marlen.py")
parser.add_argument("--category", required=False, help="Graph model to train.", default="SchnetRagged.EnergyForceModel")
parser.add_argument("--model", required=False, help="Graph model to train.", default=None)
parser.add_argument("--dataset", required=False, help="Name of the dataset.", default=None)
parser.add_argument("--make", required=False, help="Name of the class for model.", default=None)
parser.add_argument("--module", required=False, help="Name of the module for model.", default=None)
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
parser.add_argument("--fold", required=False, help="Split or fold indices to run.", default=None, nargs="+", type=int)
parser.add_argument("--seed", required=False, help="Set random seed.", default=42, type=int)
args = vars(parser.parse_args())
print("Input of argparse:", args)

# Check and set device
if args["gpu"] is not None:
    set_cuda_device(args["gpu"])
print(check_device())

# Set seed.
np.random.seed(args["seed"])
ks.utils.set_random_seed(args["seed"])

# HyperParameter is used to store and verify hyperparameter.
hyper = HyperParameter(
    hyper_info=args["hyper"], hyper_category=args["category"],
    model_name=args["model"], model_class=args["make"], dataset_class=args["dataset"], model_module=args["module"])
hyper.verify()

# Loading a specific per-defined dataset from a module in kgcnn.data.datasets.
# However, the construction must be fully defined in the data section of the hyperparameter,
# including all methods to run on the dataset. Information required in hyperparameter are for example 'file_path',
# 'data_directory' etc.
# Making a custom training script rather than configuring the dataset via hyperparameter can be
# more convenient.
dataset = deserialize_dataset(hyper["dataset"])

# Check if dataset has the required properties for model input. This includes a quick shape comparison.
# The name of the keras `Input` layer of the model is directly connected to property of the dataset.
# Example 'edge_indices' or 'node_attributes'. This couples the keras model to the dataset.
dataset.assert_valid_model_input(hyper["model"]["config"]["inputs"])

# Filter the dataset for invalid graphs. At the moment invalid graphs are graphs which do not have the property set,
# which is required by the model's input layers, or if a tensor-like property has zero length.
dataset.clean(hyper["model"]["config"]["inputs"])
data_length = len(dataset)  # Length of the cleaned dataset.

# Always train on `energy` .
# Just making sure that the target is of shape `(N, #labels)`. This means output embedding is on graph level.
label_names, label_units = dataset.set_multi_target_labels(
    "energy",
    hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper["training"] else None,
    data_unit=hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else None
)

# Make output directory
filepath = hyper.results_file_path()
postfix_file = hyper["info"]["postfix_file"]

datasetname = hyper["dataset"]["config"]["dataset_name"]
train_name = "{}_{}".format(datasetname, postfix_file) ##update

# Training on splits. Since training on Force datasets can be expensive, there is a 'execute_splits' parameter to not
# train on all splits for testing. Can be set via command line or hyperparameter.
'''
if "cross_validation" in hyper["training"]:
    from sklearn.model_selection import KFold
    splitter = KFold(**hyper["training"]["cross_validation"]["config"])
    train_test_indices = [
        (train_index, test_index) for train_index, test_index in splitter.split(X=np.zeros((data_length, 1)))]
else:
    train_test_indices_kwargs = hyper[
        "training"]["train_test_indices"] if "train_test_indices" in hyper["training"] else {}
    train_test_indices = dataset.get_train_test_indices(**train_test_indices_kwargs)
train_indices_all, test_indices_all = [], []
'''
# Run Splits.
execute_folds = args["fold"] if "execute_folds" not in hyper["training"] else hyper["training"]["execute_folds"]
model, scaled_predictions, splits_done, current_split = None, False, 0, 1

start_training = True

r2_E_list = []
r2_F_list = []

max_train = 65513
len_total = max_train + 7291
train_index = list(range(0,max_train))              # range(0, max_train) max_train = 1114
test_index = list(range(max_train+1,len_total-1))   # range(max_train+1, -1)

#for current_split, (train_index, test_index) in enumerate(train_test_indices):

# Keep list of train/test indices.
#test_indices_all.append(test_index)
#train_indices_all.append(train_index)

# Only do execute_splits out of the k-folds of cross-validation.
#if execute_folds:
#    if current_split not in execute_folds:
#       continue
#print("Running training on split: '%s'." % current_split)

# Make the model for current split using model kwargs from hyperparameter.
model = deserialize_model(hyper["model"])

# First select training and test graphs from indices, then convert them into tensorflow tensor
# representation. Which property of the dataset and whether the tensor will be ragged is retrieved from the
dataset_train, dataset_test = dataset[train_index], dataset[test_index]

# Normalize training and test targets.
# For Force datasets this training script uses the `EnergyForceExtensiveLabelScaler` class.
# Note that `EnergyForceExtensiveLabelScaler` uses both energy and forces for scaling.
# Adapt output-scale via a transform.
# Scaler is applied to target if 'scaler' appears in hyperparameter. Only use for regression.
scaled_metrics = None
if "scaler" in hyper["training"]:
    print("Using Scaler to adjust output scale of model.")
    scaler = deserialize_scaler(hyper["training"]["scaler"])
    scaler.fit_dataset(dataset_train)
    if hasattr(model, "set_scale"):
        print("Setting scale at model.")
        model.set_scale(scaler)
    else:
        print("Transforming dataset.")
        dataset_train = scaler.transform_dataset(dataset_train, copy_dataset=True, copy=True)
        dataset_test = scaler.transform_dataset(dataset_test, copy_dataset=True, copy=True)
        # If scaler was used we add rescaled standard metrics to compile, since otherwise the keras history will not
        # directly log the original target values, but the scaled ones.
        scaler_scale = scaler.get_scaling()
        force_output_parameter = hyper["model"]["config"]["outputs"]["force"]
        is_ragged = force_output_parameter["ragged"] if "ragged" in force_output_parameter else False
        mae_metric_energy = ScaledMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
        if is_ragged:
            mae_metric_force = ScaledMeanAbsoluteError(
                scaler_scale.shape, name="scaled_mean_absolute_error", ragged=True)
        else:
            mae_metric_force = ScaledForceMeanAbsoluteError(scaler_scale.shape, name="scaled_mean_absolute_error")
        if scaler_scale is not None:
            mae_metric_energy.set_scale(scaler_scale)
            mae_metric_force.set_scale(scaler_scale)
        scaled_metrics = {"energy": [mae_metric_energy], "force": [mae_metric_force]}
        scaled_predictions = True

    # Save scaler to file
    scaler.save(os.path.join(filepath, f"scaler{postfix_file}_fold_{current_split}"))

# Convert dataset to tensor information for model.
x_train = dataset_train.tensor(hyper["model"]["config"]["inputs"])
x_test = dataset_test.tensor(hyper["model"]["config"]["inputs"])

# Convert targets into tensors.
y_train = dataset_train.tensor(hyper["model"]["config"]["outputs"])
y_test = dataset_test.tensor(hyper["model"]["config"]["outputs"])

# Compile model with optimizer and loss
model.compile(**hyper.compile(
    loss={
        "energy": "mean_absolute_error",
        "force": ForceMeanAbsoluteError()
    },
    metrics=scaled_metrics
))

cbcp = ks.callbacks.ModelCheckpoint(
    "checkpoints_%s/cp_fold_%i-{epoch:04d}.model.keras" % (train_name, current_split),
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    mode='min',
    save_freq=int(len(dataset)/32*100), ## adapt batch size
    initial_value_threshold=None,
)


# Build model with reasonable data.
model.predict(x_test, batch_size=2, steps=2)
model._compile_metrics.build(y_test, y_test)
model._compile_loss.build(y_test, y_test)

# Model summary
model.summary()
print(" Compiled with jit: %s" % model._jit_compile)  # noqa
print(" Model is built: %s" % all([layer.built for layer in model._flatten_layers()]))  # noqa

if start_training:
    os.makedirs('checkpoints_{}'.format(train_name), exist_ok=True)
    epoch_start = 0
else:
    os.makedirs('checkpoints_{}'.format(train_name), exist_ok=True)
    cpk_list = [int(x.split("-")[1].split(".")[0]) for x in os.listdir("checkpoints_{}".format(train_name)) if x[-5:] == "keras"]
    epoch_start = int(np.amax(cpk_list))
    model.load_weights("checkpoints_{dataname}/cp_fold_{fold}-{epoch:04d}.model.keras".format(dataname = train_name, fold=current_split, epoch=epoch_start))


# Start and time training
start = time.time()
   
hist = model.fit(
    x_train, y_train,
    initial_epoch=epoch_start,
    validation_data=(x_test, y_test),
    **hyper.fit(callbacks=[cbcp])
)

stop = time.time()
print("Print Time for training: ", str(timedelta(seconds=stop - start)))

# Save history for this fold.
save_pickle_file(hist.history, os.path.join(filepath, f"history{postfix_file}_fold_{current_split}.pickle"))
save_pickle_file(str(timedelta(seconds=stop - start)),
                 os.path.join(filepath, f"time{postfix_file}_fold_{current_split}.pickle"))

# Plot prediction
predicted_y = model.predict(x_test, verbose=0)
true_y = y_test

plot_predict_true(np.array(predicted_y["energy"]), np.array(true_y["energy"]),
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict_energy{postfix_file}_fold_{splits_done}.png",
                  scaled_predictions=scaled_predictions)

num_forces = [len(x) for x in dataset_test.get("force")]
plot_predict_true(np.concatenate([np.array(f)[:l] for f,l in zip(predicted_y["force"], num_forces)], axis=0),
                  np.concatenate([np.array(f)[:l] for f,l in zip(true_y["force"], num_forces)], axis=0),
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict_force{postfix_file}_fold_{splits_done}.png",
                  scaled_predictions=scaled_predictions)

# rescale predicted_y["energy"], force=predicted_y["force"]
rescaled_predicted_y = scaler.inverse_transform(y=[predicted_y["energy"],predicted_y["force"]], X = dataset_test.get('atomic_number'))

rescaled_true_y = scaler.inverse_transform(y=[true_y["energy"],true_y["force"]], X = dataset_test.get('atomic_number'))

r2_E_test = add_plots.R_squared(np.array(rescaled_true_y[0]), np.array(rescaled_predicted_y[0])) 
r2_F_test = add_plots.R_squared(np.concatenate([np.array(f) for f in rescaled_true_y[1]], axis=0), np.concatenate([np.array(f) for f in rescaled_predicted_y[1]], axis=0)) 

r2_E_list.append(r2_E_test)
r2_F_list.append(r2_F_test)

# plot_en_err_size(rescaled_predicted_y, rescaled_true_y, fold_i, model_name: str = "", filepath: str = None, file_name: str = "", dataset_name: str = "", figsize: list = None, dpi: float = None, show_fig: bool = True):
add_plots.plot_en_err_size(rescaled_predicted_y, rescaled_true_y, x_test, current_split, filepath=filepath, model_name=hyper.model_name, dataset_name=hyper.dataset_class, dpi=300.0)

plot_predict_true(rescaled_predicted_y[0],rescaled_true_y[0],
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict_energy_unscaled_{postfix_file}_fold_{current_split}.png",
                  dpi=300.0)

plot_predict_true(np.concatenate([np.array(f) for f in rescaled_predicted_y[1]], axis=0),
                  np.concatenate([np.array(f) for f in rescaled_true_y[1]], axis=0),
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict_force_unscaled_{postfix_file}_fold_{current_split}.png",
                  dpi=300.0)

plot_predict_true(np.array(predicted_y["energy"]), np.array(true_y["energy"]),
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict_energy{postfix_file}_fold_{current_split}.png",
                  scaled_predictions=scaled_predictions, dpi=300.0)

plot_predict_true(np.concatenate([np.array(f) for f in predicted_y["force"]], axis=0),
                  np.concatenate([np.array(f) for f in true_y["force"]], axis=0),
                  filepath=filepath, data_unit=label_units,
                  model_name=hyper.model_name, dataset_name=hyper.dataset_class, target_names=label_names,
                  file_name=f"predict_force{postfix_file}_fold_{current_split}.png",
                  scaled_predictions=scaled_predictions, dpi=300.0)

# Save last keras-model to output-folder.
model.save(os.path.join(filepath, f"model{postfix_file}_fold_{current_split}.keras"))

# Save last keras-model to output-folder.
model.save_weights(os.path.join(filepath, f"model{postfix_file}_fold_{current_split}.weights.h5"))

# Get loss from history
#splits_done = splits_done + 1

# Save original data indices of the splits.
#np.savez(os.path.join(filepath, f"{hyper.model_name}_test_indices_{postfix_file}.npz"), *test_indices_all)
#np.savez(os.path.join(filepath, f"{hyper.model_name}_train_indices_{postfix_file}.npz"), *train_indices_all)

r2_dict = {'r2_E_test': np.array(r2_E_list), 'r2_F_test': np.array(r2_F_list)}
np.save('{}/r2_values.npy'.format(filepath), r2_dict, allow_pickle=True)

# Plot training- and test-loss vs epochs for all splits.
data_unit = hyper["data"]["data_unit"] if "data_unit" in hyper["data"] else ""
history_list = load_history_list(os.path.join(filepath, f"history{postfix_file}_fold_(i).pickle"), current_split + 1)
plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                     model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
                     filepath=filepath, file_name=f"loss{postfix_file}.png")

# Save hyperparameter again, which were used for this fit.
hyper.save(os.path.join(filepath, f"{hyper.model_name}_hyper{postfix_file}.json"))

# Save score of fit result for as text file.
time_list = load_time_list(os.path.join(filepath, f"time{postfix_file}_fold_(i).pickle"), current_split + 1)
save_history_score(
    history_list, loss_name=None, val_loss_name=None,
    model_name=hyper.model_name, data_unit=data_unit, dataset_name=hyper.dataset_class,
    model_class=hyper.model_class,
    multi_target_indices=hyper["training"]["multi_target_indices"] if "multi_target_indices" in hyper[
        "training"] else None,
    execute_folds=execute_folds, seed=args["seed"],
    filepath=filepath, file_name=f"score{postfix_file}.yaml",
    trajectory_name=(dataset.trajectory_name if hasattr(dataset, "trajectory_name") else None),
    time_list=time_list
)

## additional plots
#add_plots.plot_r_squared(r2_dict, model_name=hyper.model_name, 
#                         dataset_name=hyper.dataset_class, filepath=filepath, 
#                        file_name=f"r2{postfix_file}.png", dpi = 300.0)

add_plots.plot_train_test_loss(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"loss{postfix_file}.png", dpi = 300.0)

add_plots.plot_train_test_lr(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"lr{postfix_file}.png", dpi = 300.0)

add_plots.plot_train_test_mae_val(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit="eV", dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_val{postfix_file}.png", dpi = 300.0)

add_plots.plot_train_test_val_loss_all(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_val_loss_all{postfix_file}.png", dpi = 300.0)


'''
add_plots.plot_train_test_val_loss_eg(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_val_loss_eg{postfix_file}.png", dpi = 300.0)
'''
add_plots.plot_train_test_val_loss(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_val_loss{postfix_file}.png", dpi = 300.0)

add_plots.plot_train_test_train_loss_all(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_train_loss_all{postfix_file}.png", dpi = 300.0)
'''
add_plots.plot_train_test_train_loss_eg(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_train_loss_eg{postfix_file}.png", dpi = 300.0)
'''
add_plots.plot_train_test_loss_all(history_list, loss_name=None, val_loss_name=None,
                 model_name=hyper.model_name, data_unit=None, dataset_name=hyper.dataset_class,
                 filepath=filepath, file_name=f"mae_loss_all{postfix_file}.png", dpi = 300.0)
