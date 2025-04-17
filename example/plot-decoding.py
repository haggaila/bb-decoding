# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import json
import matplotlib.pyplot as plt

from bb_decoding.database_utils import query_simulations_by_name, generate_decoding_paths

b_save_figures = False

i_fig = 7
match i_fig:
    case 2:
        s_parameter = "back"
        s_x_label = "backaction error"
        s_simulations_name = "leaked init, no corr, medium rates, idle increase, scan back"
    case 3:
        s_parameter = "leak"
        s_x_label = "leakage error"
        s_simulations_name = "leaked init, no corr, high seep and back, idle increase, scan leak"
    case 4:
        s_parameter = "seep"
        s_x_label = "seepage, long idle"
        s_simulations_name = "leaked init, no corr, medium errors, scan seep and id_l"
    case 6:
        s_parameter = "n_cycles"
        s_x_label = "number of cycles"
        s_simulations_name = "leaked init, no corr, medium errors, scan cycles"
    case 7:
        s_parameter = "corr"
        s_x_label = "correlated preparation error"
        s_simulations_name = "no leak, scan corr"
    case _:
        raise Exception("Unknown figure number.")

sims = query_simulations_by_name(s_simulations_name, s_parameter)
sims_flip_0 = sims.loc[~sims["b_readout_flip"]]
sims_flip_1 = sims.loc[sims["b_readout_flip"]]
leak_errs = [sims_flip_0[s_parameter].values,
             sims_flip_1[s_parameter].values]

bp_iterations = [sims_flip_0.bp_iterations.values, sims_flip_1.bp_iterations.values]
bp_converged = [sims_flip_0.bp_converged.values, sims_flip_1.bp_converged.values]
pre_readout_leaked = [sims_flip_0.mean_leaked.values, sims_flip_1.mean_leaked.values]
pre_readout_ground = [sims_flip_0.mean_ground.values, sims_flip_1.mean_ground.values]
cycle_err = [sims_flip_0.cycle_err.values, sims_flip_1.cycle_err.values]
shots_err = [sims_flip_0.shots_err.values, sims_flip_1.shots_err.values]

fmts = ["-o", "--^"]
flip_legends = ["w/o flip", "w/ flip"]
plt.rc("font", family="serif")

fontsize = 14
i_err = 1 if i_fig < 6 else 3
plt.rcParams["font.size"] = fontsize
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
if i_fig == 7:
    data_arrays = [bp_iterations, bp_converged, pre_readout_ground, cycle_err]
    data_titles = ["(a) BP iterations", "(b) BP converged fraction",
                   "(c) Pre-readout ground population", "(d) Logical error / cycle"]
else:
    data_arrays = [bp_iterations, bp_converged, pre_readout_leaked, cycle_err]
    data_titles = ["(a) BP iterations", "(b) BP converged fraction",
                   "(c) Pre-readout leaked population", "(d) Logical error / cycle"]

i_col = 0
for i_flip in [0, 1]:
    for i_data, data in enumerate(data_arrays):
        i_row = i_data // 2
        i_col = i_data % 2
        ax = axs[i_row, i_col]
        ax.errorbar(
            leak_errs[i_flip],
            data[i_flip],
            yerr=shots_err[i_flip] if i_data == i_err else None,
            fmt=fmts[i_flip],
            label=flip_legends[i_flip],
            markersize=9,
            capsize=6,
            linewidth=3,
        )
        ax.legend(fontsize=fontsize)
        if i_fig < 6:
            if i_fig == 4:
                ax.set_xticks([.2, .5, .7], ["0.2,\n0.005", "0.5,\n0.008", "0.7,\n0.01"])
            ax.set_xlabel(s_x_label, fontsize=fontsize + 2)
        else:
            if i_row == 1:
                ax.set_xlabel(s_x_label, fontsize=fontsize + 2)
        ax.set_title(data_titles[i_data], pad=12)

plt.tight_layout()
_, _, _, s_plot_path = generate_decoding_paths(1)
s_file_name = s_plot_path + s_parameter
if b_save_figures:
    plt.savefig(s_file_name + '.png')
    json.dump(sims_flip_1.iloc[0].to_dict(), fp=open(s_file_name + '.txt', "w"), indent=4)
plt.show()
tmp = 2
