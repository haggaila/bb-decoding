# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import json
import numpy as np
import matplotlib.pyplot as plt
from bb_decoding.database_utils import query_simulations_by_name, generate_decoding_paths, load_simulation_data

fontsize = 14
b_save_figures = True

s_simulations_name = "no leaked init, no corr, medium errors"
n_simulations_cycles = 12

data_titles = ["(a) Ground state, Z checks", "(b) Ground state, X checks",
               "(c) Leaked state, Z checks", "(d) Leaked state, X checks"]
column_titles = [" (Z checks)", " (X checks)"]

fmts = ["-o", "--^"]
flip_legends = ["w/o flip", "w/ flip"]
s_x_label = "cycle"
cols = [0, 1]
n_cols = len(cols)
plt.rc("font", family="serif")
plt.rcParams["font.size"] = fontsize
if n_cols == 1:
    fig, axs = plt.subplots(2, n_cols, figsize=(9, 6))
else:
    fig, axs = plt.subplots(2, n_cols, figsize=(12, 6))

sims = query_simulations_by_name(s_simulations_name, "b_readout_flip")
if n_simulations_cycles is not None:
    sims = sims[sims.n_cycles == n_simulations_cycles]
if len(sims) != 2 or sims.b_readout_flip.iloc[0] == sims.b_readout_flip.iloc[1]:
    raise Exception("Failed to find exactly two simulations w/ and w/o flip.")

for i_flip in [0, 1]:
    n_cycles = sims.n_cycles.iloc[i_flip]
    n_shots = sims.n_shots.iloc[i_flip]
    n2 = int(int(sims.code_name.iloc[i_flip].split(".")[0]) / 2)
    sim_id = sims.unique_id.iloc[i_flip]
    sim_data = load_simulation_data(sim_id, n_up_dirs=1)
    mean_ground_z_x = sim_data["mean_ground_z_x"]
    mean_leaked_z_x = sim_data["mean_leaked_z_x"]
    data_arrays = [mean_ground_z_x, mean_leaked_z_x]
    i_data = 0
    for i_row in [0, 1]:
        for i_col in cols:
            data_ = np.reshape(data_arrays[i_row][:, i_col], (n_cycles, n2))
            data = np.mean(data_, 1)
            err = (data * (1. - data) / (n_shots * n2)) ** .5
            if n_cols == 1:
                ax = axs[i_row]
            else:
                ax = axs[i_row, i_col]
            ax.errorbar(
                range(1, n_cycles + 1),
                data,
                yerr=err,
                fmt=fmts[i_flip],
                label=flip_legends[i_flip],
                markersize=9,
                capsize=6,
                linewidth=3,
            )
            if i_row:
                ax.set_xlabel(s_x_label, fontsize=fontsize + 2)
                ax.legend(fontsize=fontsize)
            else:
                ax.legend(fontsize=fontsize, loc="right")

            ax.set_title(data_titles[i_data])
            i_data += 1

plt.tight_layout()
_, _, _, s_plot_path = generate_decoding_paths(1)
s_file_name = s_plot_path + "evolution"
if b_save_figures:
    plt.savefig(s_file_name + '.png')
    json.dump(sims.iloc[0].to_dict(), fp=open(s_file_name + '.txt', "w"), indent=4)
plt.show()
tmp = 2
