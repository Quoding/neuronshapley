###############################################################################
import os
import time
import sys
import multiprocessing as mp
from multiprocessing import dummy as multiprocessing
import tensorflow as tf
import logging
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from inception_utils import *
import h5py

from custom import NetWrapper

slim = tf.contrib.slim


seed = sys.argv[1]
# key = sys.argv[2]  # Class name. Use 'all' for overall performance.
# model_scope = "InceptionV3"
metric = sys.argv[2]  # metric one of accuracy, binary, xe_loss.
# DATA_DIR = "./imagenet"
MEM_DIR = "./results"
CHECKPOINT = f"data/models/tf_model_{seed}.ckpt"
BATCH_SIZE = 128
TIME_START = time.time()
NUM_CLASSES = 1000
SAVE_FREQ = 1
GLOBAL_SAMPLE_COUNTER = 0


def remove_players(model, players):
    """Remove selected players (filters) in the Inception-v3 network."""
    if isinstance(players, str):
        players = [players]
    for player in players:
        # Get the variables part of the player set that need to be evaluated
        # It removes the last index because it is the integer representing the convolutional filter (the player)
        variables = layer_dic["_".join(player.split("_")[:-1])]

        # This updates the values of variables to their output when evaluating the input
        var_vals = model.sess.run(variables, feed_dict={model.input: model.X})

        # Set the variable values to 0.
        for var, var_val in zip(variables, var_vals):
            if "variance" in var.name:
                var_val[..., int(player.split("_")[-1])] = 1.0
            elif "beta" in var.name:
                pass
            else:
                var_val[..., int(player.split("_")[-1])] = 0.0

            var.load(var_val, model.sess)


def return_player_output(model, player):
    """The output of a filter."""
    layer = "_".join(player.split("_")[:-1])
    layer = "/".join(layer.split("/")[1:])
    unit = int(player.split("_")[-1])
    return model.ends[layer][..., unit]


def one_iteration(
    model,
    players,
    images,
    labels,
    chosen_players=None,
    c=None,
    metric="accuracy",
    truncation=None,
):
    """One iteration of Neuron-Shapley algoirhtm."""
    model.restore(CHECKPOINT)
    # Original performance of the model with all players present.
    init_val = value(model, images, labels, metric)
    if c is None:
        c = {i: np.array([i]) for i in range(len(players))}
    elif not isinstance(c, dict):
        c = {i: np.where(c == i)[0] for i in set(c)}
    if truncation is None:
        truncation = len(c.keys())
    if chosen_players is None:
        chosen_players = np.arange(len(c.keys()))

    # A random ordering of players
    idxs = np.random.permutation(len(c.keys()))
    # -1 default value for players that have already converged
    marginals = -np.ones(len(c.keys()))
    marginals[chosen_players] = 0.0
    t = time.time()
    truncation_counter = 0
    old_val = init_val.copy()

    # Chosen players == U
    # players is a list of filter names in the model (list of strings)
    for n, idx in enumerate(idxs[::-1]):
        if idx in chosen_players:
            if old_val is None:
                old_val = value(model, images, labels, metric)

            remove_players(
                model, players[c[idx]]
            )  # Set filters to their mean output (like zeroing out, but more accurate in this case)
            new_val = value(model, images, labels, metric)
            marginals[c[idx]] = (old_val - new_val) / len(c[idx])
            old_val = new_val
            # if isinstance(truncation, int):
            #     if n >= truncation:
            #         break
            # else:
            #     if n % 10 == 0:
            #         # print("icitte")
            #         print(n, time.time() - t, new_val)
            #     val_diff = new_val - base_value
            #     if metric == "accuracy" and val_diff <= truncation:
            #         truncation_counter += 1
            #     else:
            #         truncation_counter = 0
            #     if truncation_counter > 5:
            #         break
        else:
            old_val = None
    return idxs.reshape((1, -1)), marginals.reshape((1, -1))


def value(model, images, labels, metric="accuracy", batch_size=BATCH_SIZE):
    """The performance of the model on given image-label pairs."""
    global GLOBAL_SAMPLE_COUNTER
    GLOBAL_SAMPLE_COUNTER += 1
    val = 0.0
    if metric == "accuracy":
        val = model.sess.run(model.accuracy, feed_dict={model.input: model.X})
    else:
        raise ValueError("Invalid metric!")
    return val


seed = sys.argv[1]
metric = sys.argv[2]  # metric one of accuracy, binary, xe_loss.
bound = "Bernstein"
truncation = "notrunc"
max_sample_size = 128
# time.sleep(10 * np.random.random())
## Experiment Directory
experiment_dir = os.path.join(MEM_DIR, f"NShap/toy_model_{seed}/{metric}_new")
if not tf.gfile.Exists(experiment_dir):
    tf.gfile.MakeDirs(experiment_dir)
## CB directory
experiment_name = f"cb_{seed}_{truncation}"
cb_dir = os.path.join(experiment_dir, experiment_name)
if not tf.gfile.Exists(cb_dir):
    tf.gfile.MakeDirs(cb_dir)
## Load Model and find all convolutional filters
tf.reset_default_graph()
model = NetWrapper(seed)

graph = tf.get_default_graph()
writer = tf.summary.FileWriter(logdir="logdir", graph=graph)
writer.flush()
model_variables = tf.global_variables()
convs = ["layer_1", "layer_2"]
layer_dic = {
    conv: [var for var in model_variables if conv in var.name] for conv in convs
}
## Load the list of all players (filters) else save
if tf.gfile.Exists(os.path.join(experiment_dir, "players.txt")):
    players = open(os.path.join(experiment_dir, "players.txt")).read().split(",")
    players = np.array(players)
else:
    players = []
    var_dic = {var.name: var for var in model_variables}
    # print(var_dic)
    for conv in layer_dic.keys():
        players.append(
            ["{}_{}".format(conv, i) for i in range(var_dic[conv + ":0"].shape[-1])]
        )
    players = np.sort(np.concatenate(players))
    open(os.path.join(experiment_dir, "players.txt"), "w").write(",".join(players))


# pprint(model_variables)
# print("========")
# pprint(layer_dic)
# print("========")
# pprint(players)
print(players)
# input()
## Load metric's base value (random performance)
if metric == "accuracy":
    base_value = 1.0 / NUM_CLASSES
# elif metric == "xe_loss":
#     base_value = -np.log(NUM_CLASSES)
# elif metric == "binary":
#     base_value = 0.5
# elif metric == "logit":
#     base_value = 0
else:
    raise ValueError("Invalid metric!")

## Assign expriment number to this specific run of cb_run.py
results = [
    saved
    for saved in tf.gfile.ListDirectory(cb_dir)
    if "agg" not in saved and ".h5" in saved
]
experiment_number = 0
if len(results):
    results_experiment_numbers = [
        int(result.split(".")[-2].split("_")[-1][1:]) for result in results
    ]
    experiment_number += np.max(results_experiment_numbers) + 1
print(experiment_number)
save_dir = os.path.join(cb_dir, "{}.h5".format("0" + str(experiment_number).zfill(5)))
print(save_dir)
## Create placeholder for results in save ASAP to prevent having the
## same expriment_number with other parallel cb_run.py scripts
mem_tmc = np.zeros((0, len(players)))
idxs_tmc = np.zeros((0, len(players))).astype(int)
with h5py.File(save_dir, "w") as foo:
    foo.create_dataset("mem_tmc", data=mem_tmc, compression="gzip")
    foo.create_dataset("idxs_tmc", data=idxs_tmc, compression="gzip")

## Running CB-Shapley
c = None
if c is None:
    c = {i: np.array([i]) for i in range(len(players))}
elif not isinstance(c, dict):
    c = {i: np.where(np.array(c) == i)[0] for i in set(list(c))}

counter = 0
new_time = 0
old_time = 0
flag_stop = False
n_samples = []
# time.sleep(5)
while True:
    cnt = 0
    print("HERE")
    print(os.path.isfile(experiment_dir + "/go_run.lock"))
    while not os.path.isfile(experiment_dir + "/go_run.lock"):
        time.sleep(0.1)
        print("cb_run in while loop")
        # cnt += 1
        # # Unstuck because other script is probably over
        # if cnt > 100:
        #     with open(experiment_dir + "/go_run.lock", "w") as f:
        #         f.write("asd")
        #         f.flush()

    new_time = time.time()
    # print(f"one iteration: {new_time - old_time} seconds")
    old_time = new_time

    ## Load the list of players (filters) that are determined to be not confident enough
    ## by the cb_aggregate.py running in parallel to this script
    if tf.gfile.Exists(os.path.join(cb_dir, "chosen_players.txt")):
        chosen_players = open(os.path.join(cb_dir, "chosen_players.txt")).read()
        chosen_players = np.array(chosen_players.split(",")).astype(int)
        if len(chosen_players) == 1:
            flag_stop = True
        # if len(chosen_players) == 1:
        # with open(experiment_dir + "/go_agg.lock", "w") as f:
        #     f.write("asd")
    else:
        chosen_players = None
    # print("Outisde of if")
    # input("====")
    t_init = time.time()
    # iter_images, iter_labels = load_images_labels(
    #     key,
    #     num_images,
    #     max_sample_size,
    #     model,
    #     max_size=25000,
    # )

    # if metric == "binary":
    #     rnd_images, _ = load_images_labels(
    #         "rnd", len(iter_images), max_sample_size, model, max_size=25000
    #     )
    #     iter_images = np.concatenate([iter_images, rnd_images])
    #     iter_labels = np.concatenate(
    #         [iter_labels, -np.ones(len(rnd_images)).astype(int)]
    #     )

    idxs, vals = one_iteration(
        model=model,
        players=players,
        images=None,
        labels=None,
        chosen_players=chosen_players,
        c=c,
        metric=metric,
        truncation=truncation,
    )

    # print("here")
    # print(mem_tmc)
    # print(vals)
    mem_tmc = np.concatenate([mem_tmc, vals])
    # print(mem_tmc)
    # input()

    # print("idxs")
    # print(idxs_tmc)
    # print(idxs)
    idxs_tmc = np.concatenate([idxs_tmc, idxs])
    n_samples.append(GLOBAL_SAMPLE_COUNTER)
    # print(idxs_tmc)
    # input()
    ## Save results every SAVE_FREQ iterations
    if counter % SAVE_FREQ == SAVE_FREQ - 1 or GLOBAL_SAMPLE_COUNTER >= 20000:
        with h5py.File(save_dir, "w") as foo:
            foo.create_dataset("mem_tmc", data=mem_tmc, compression="gzip")
            foo.create_dataset("idxs_tmc", data=idxs_tmc, compression="gzip")
            foo.create_dataset(
                "n_samples", data=np.array(n_samples), compression="gzip"
            )
            print("Just wrote n_samples: ", n_samples[-1], GLOBAL_SAMPLE_COUNTER)

    counter += 1
    # print(time.time() - t_init, time.time() - TIME_START)
    # if not tf.test.is_gpu_available():
    #     print("No gpu!")
    #     print(time.time() - TIME_START)
    # else:
    #     print("There is a gpu!")
    #     print(time.time() - TIME_START)

    if GLOBAL_SAMPLE_COUNTER >= 20000:
        print("Exited with {} samples".format(GLOBAL_SAMPLE_COUNTER))
        with open(experiment_dir + "/go_agg.lock", "w") as f:
            f.write("asd")
        exit()

    os.remove(experiment_dir + "/go_run.lock")
    with open(experiment_dir + "/go_agg.lock", "w") as f:
        f.write("asd")

print(f"This run took {GLOBAL_SAMPLE_COUNTER} interactions")
