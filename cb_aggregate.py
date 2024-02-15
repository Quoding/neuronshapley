import time
import os
import sys
import numpy as np
import tensorflow as tf
import h5py


MEM_DIR = "./results"

keys = sys.argv[1]
metric = sys.argv[2]
num_images = int(sys.argv[3])
adversarials = sys.argv[4]
model_seed = sys.argv[5]

np.random.seed(int(model_seed))
total_samples = 0
time.sleep(5)
flag_stop = False
while True:
    for adv in adversarials.split(","):
        print(adv)
        adversarial = adv == "True"
        for key in keys.split(","):
            bound = "Bernstein"
            truncation = 0.2
            # if metric == "logit":
            #     truncation = 3443
            # max_sample_size = 128
            ## Experiment Directory
            experiment_dir = os.path.join(
                MEM_DIR, "NShap/toy_model_{}/{}_new".format(model_seed, metric)
            )
            if not tf.gfile.Exists(experiment_dir):
                tf.gfile.MakeDirs(experiment_dir)
            # if max_sample_size is None or max_sample_size > num_images:
            #     max_sample_size = num_images
            experiment_name = f"cb_{model_seed}_{truncation}"
            if adversarial:
                experiment_name = "ADV" + experiment_name
            cb_dir = os.path.join(experiment_dir, experiment_name)
            if not tf.gfile.Exists(cb_dir):
                tf.gfile.MakeDirs(cb_dir)
            ##
            if metric == "accuracy":
                R = 1.0
            # elif metric == "xe_loss":
            #     R = np.log(1000)
            # elif metric == "binary":
            # R = 1.0
            # elif metric == "logit":
            #     R = 10.0
            else:
                raise ValueError("Invalid metric!")
            top_k = 5
            delta = 0.2

            ## Start
            if not tf.gfile.Exists(os.path.join(experiment_dir, "players.txt")):
                print("Does not exist!")
                continue
            players = (
                open(os.path.join(experiment_dir, "players.txt")).read().split(",")
            )
            players = np.array(players)
            print(players)
            if not tf.gfile.Exists(os.path.join(cb_dir, "chosen_players.txt")):
                open(os.path.join(cb_dir, "chosen_players.txt"), "w").write(
                    ",".join(np.arange(len(players)).astype(str))
                )

            # Wait for CB_run
            with open(experiment_dir + "/go_run.lock", "w") as f:
                f.write("asd")
            cnt = 0
            while not os.path.isfile(experiment_dir + "/go_agg.lock"):
                time.sleep(0.1)
                cnt += 1
                print("stuck in while loop")
                # Unstick because cb_run probably already is over
                if cnt > 100:
                    with open(experiment_dir + "/go_agg.lock", "w") as f:
                        f.write("asd")
                    print("Wrote file")
            os.remove(experiment_dir + "/go_agg.lock")
            results = np.sort(
                [
                    saved
                    for saved in tf.gfile.ListDirectory(cb_dir)
                    if "agg" not in saved and ".h5" in saved
                ]
            )
            squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
            max_vals, min_vals = -np.ones(len(players)), np.ones(len(players))
            total_samples = 0
            for result in results:
                try:
                    with h5py.File(os.path.join(cb_dir, result), "r") as foo:
                        mem_tmc = foo["mem_tmc"][:]
                        n_samples = foo["n_samples"][-1]
                except:
                    continue
                if not len(mem_tmc):
                    continue
                sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
                squares += np.sum((mem_tmc != -1) * (mem_tmc**2), 0)
                counts += np.sum(mem_tmc != -1, 0)
                total_samples += n_samples
            print("Loaded {} samples".format(total_samples))
            # temp = mem_tmc * (mem_tmc != -1) - 1000 * (mem_tmc == -1)
            # max_vals = np.maximum(max_vals, np.max(temp, 0))
            # temp = mem_tmc * (mem_tmc != -1) + 1000 * (mem_tmc == -1)
            # min_vals = np.minimum(min_vals, np.min(temp, 0))
            counts = np.clip(counts, 1e-12, None)
            vals = sums / (counts + 1e-12)
            variances = R * np.ones_like(vals)
            variances[counts > 1] = squares[counts > 1]
            variances[counts > 1] -= (sums[counts > 1] ** 2) / counts[counts > 1]
            variances[counts > 1] /= counts[counts > 1] - 1
            if np.max(counts) == 0:
                os.remove(os.path.join(cb_dir, result))
            cbs = R * np.ones_like(vals)
            if bound == "Hoeffding":
                cbs[counts > 1] = R * np.sqrt(
                    np.log(2 / delta) / (2 * counts[counts > 1])
                )
            elif bound == "Bernstein":
                # From: http://arxiv.org/pdf/0907.3740.pdf
                cbs[counts > 1] = np.sqrt(
                    2 * variances[counts > 1] * np.log(2 / delta) / counts[counts > 1]
                ) + 7 / 3 * R * np.log(2 / delta) / (counts[counts > 1] - 1)

            thresh = (vals)[np.argsort(vals)[-top_k - 1]]
            chosen_players = np.where(
                ((vals - cbs) < thresh) * ((vals + cbs) > thresh)
            )[0]

            print("Statistics below: cb_dir, np.mean(counts), len(chosen_players)")
            print(cb_dir, np.mean(counts), len(chosen_players))

            open(os.path.join(cb_dir, "chosen_players.txt"), "w").write(
                ",".join(chosen_players.astype(str))
            )
            open(os.path.join(cb_dir, "variances.txt"), "w").write(
                ",".join(variances.astype(str))
            )
            open(os.path.join(cb_dir, "vals.txt"), "w").write(
                ",".join(vals.astype(str))
            )
            open(os.path.join(cb_dir, "counts.txt"), "w").write(
                ",".join(counts.astype(str))
            )

            open(os.path.join(cb_dir, "chosen_players_cumul.txt"), "a").write(
                ",".join(chosen_players.astype(str)) + "\n"
            )
            open(os.path.join(cb_dir, "variances_cumul.txt"), "a").write(
                ",".join(variances.astype(str)) + "\n"
            )
            open(os.path.join(cb_dir, "vals_cumul.txt"), "a").write(
                ",".join(vals.astype(str)) + "\n"
            )
            open(os.path.join(cb_dir, "counts_cumul.txt"), "a").write(
                ",".join(counts.astype(str)) + "\n"
            )
            open(os.path.join(cb_dir, "n_samples_cumul.txt"), "a").write(
                str(total_samples) + "\n"
            )

            if len(chosen_players) == 1:
                with open(experiment_dir + "/final_samples", "a") as f:
                    f.write(str(total_samples) + "\n")
                flag_stop = True

            if total_samples >= 20000:
                print("Final top_k")
                print(np.argsort(vals)[-top_k - 1 :])
                with open(experiment_dir + "/go_run.lock", "w") as f:
                    f.write("asd")
                exit()

            #     print("Final top_k")
            #     print(np.argsort(vals)[: -top_k - 1])
            #     exit()
        #         break
        # if len(chosen_players)== 1:
        #     break
