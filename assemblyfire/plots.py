# -*- coding: utf-8 -*-
"""
Assembly detection related plots
author: András Ecker, last update: 10.2020
"""

import numpy as np
from copy import deepcopy
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


sns.set(style="ticks", context="notebook")


def _avg_rate(rate, bin_, t_start, t_end):
    """Helper function to bin rate for bar plots"""

    t1 = np.arange(t_start, t_end, bin_)
    t2 = t1 + bin_
    avg_rate = np.zeros_like(t1, dtype=np.float)
    t = np.linspace(t_start, t_end, len(rate))
    for i, (t1_, t2_) in enumerate(zip(t1, t2)):
        avg_ = np.mean(rate[np.where((t1_ <= t) & (t < t2_))])
        if avg_ != 0.:
            avg_rate[i] = avg_
    return avg_rate


def _gids_to_depth(gids, depths):
    """Converts unique gids to cortical depths"""
    return [depths[gid] for gid in gids]


def _spiking_gids_to_depth(spiking_gids, depths):
    """Converts array of gids to array of cortical depths"""

    spiking_depths = np.zeros_like(spiking_gids, dtype=np.float)
    for gid in np.unique(spiking_gids):
        idx = np.where(spiking_gids == gid)
        spiking_depths[idx] = depths[gid]
    return spiking_depths


def plot_raster(spike_timesE, spiking_gidsE, rateE, spike_timesI, spiking_gidsI, rateI,
                t_start, t_end, depths, ystuff, fig_name):
    """Plots raster"""

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(spike_timesI, _spiking_gids_to_depth(spiking_gidsI, depths), color="blue", marker='.', s=2, edgecolor="none")
    ax.scatter(spike_timesE, _spiking_gids_to_depth(spiking_gidsE, depths), color="red", marker='.', s=2, edgecolor="none")
    for h in ystuff["hlines"]:
        ax.axhline(h, color="gray", lw=0.5)
    ax2 = ax.twinx()
    bin_ = (t_end-t_start)/len(rateI)
    t_steps = np.arange(t_start, t_end, bin_) + bin_
    ax2.step(t_steps, rateI, where="pre", color="blue")
    ax2.step(t_steps, rateE, where="pre", color="red")
    ax.set_xlim([t_start, t_end])
    ax.set_xlabel("Time (ms)")
    ax.set_yticks(ystuff["yticks"])
    ax.set_yticklabels(ystuff["yticklabels"])
    ax.set_ylim([ystuff["hlines"][-1], ystuff["hlines"][0]])
    ax2.set_ylabel("Rate (spikes/(N*s))")
    ax2.set_ylim(bottom=0)

    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_rate(rate, rate_th, t_start, t_end, fig_name):
    """Plots thresholded rate (it's actually spike count)"""

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    ax.plot(np.linspace(t_start, t_end, len(rate)), rate, "k-")
    ax.axhline(np.mean(rate)+rate_th, color="gray", ls="--")
    ax.set_xlim([t_start, t_end])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (spikes/(N*s))")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _get_pattern_idx(t_bins):
    """maps stimulus times to column idx in the spike_matrix
    Note: doesn't guarantee that the time bin is gonna be *after* the stim presentation"""

    stim_times = np.arange(2000, 61001, 1000) # this is hard coded...
    return [np.abs(t_bins - t).argmin() for t in stim_times]


def plot_sim_matrix(sim_matrix, patterns, col_map, fig_name):
    """Plots similarity matrix"""

    t_idx = _get_pattern_idx(col_map)
    sim_mat = deepcopy(sim_matrix)
    np.fill_diagonal(sim_mat, np.nan)

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(sim_mat, cmap="coolwarm",
                  aspect="auto", interpolation="none")
    fig.colorbar(i)
    ax.set_xticks(t_idx); ax.set_xticklabels(patterns)
    ax.xaxis.tick_top()
    ax.set_xlabel("time bins")
    ax.set_yticks(t_idx); ax.set_yticklabels(patterns)
    ax.set_ylabel("time bins")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_transformed(transformed, patterns, t_bins, fig_name):
    """Plots time series in factor analysis/PCA space"""

    t_idx = _get_pattern_idx(t_bins)
    transformed_T = transformed.T

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(transformed_T, cmap="coolwarm", aspect="auto")
    ax.set_xlabel("time bins")
    ax.set_xticks(t_idx); ax.set_xticklabels(patterns)
    ax.xaxis.tick_top()
    ax.set_ylabel("components")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _reorder_gids(row_map, depths):
    """reorder components by gid depths"""
    tmp = np.asarray([depths[gid] for gid in row_map])
    return np.argsort(tmp)


def plot_components(components, row_map, depths, fig_name):
    """Plots components of factor analysis/PCA"""

    gid_idx = _reorder_gids(row_map, depths)[::-1]
    n = len(gid_idx)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(components[:, gid_idx], cmap="coolwarm", aspect="auto")
    ax.set_xlabel("gids (n=%i)"%n)
    ax.set_xticks([0, n])
    ax.set_xticklabels(["L2", "L6"])
    ax.xaxis.tick_top()
    #ax.set_xlim([0, xstuff["vlines"][-1]])
    ax.set_ylabel("components")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_rhos_deltas(rhos, deltas, tmp_idx, gammas, fit, lCI, uCI, centroid_idx, fig_name):
    """Plots rhos vs. deltas a la Rodriguez and Laio 2014"""

    tmp_centroid_idx = np.arange(len(centroid_idx))

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(rhos, deltas, c="black", marker='.', s=12)
    ax.scatter(rhos[centroid_idx], deltas[centroid_idx], c="red", marker='.', s=12)
    ax.set_xlabel(r"$\rho_i$")
    ax.set_xticks([np.min(rhos), np.max(rhos)])
    ax.set_xlim([np.min(rhos), np.max(rhos)])
    ax.set_ylabel(r"$\delta_i$")
    ax.set_ylim([np.min(deltas), np.max(deltas)])
    ax.set_yticks([np.min(deltas), np.max(deltas)])
    sns.despine(ax=ax, offset=5, trim=True)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(tmp_idx, gammas, c="black", marker='.', s=12)
    ax2.scatter(tmp_idx[tmp_centroid_idx], gammas[tmp_centroid_idx], c="red",
                marker='.', s=12, label="centroids")
    ax2.plot(tmp_idx, fit, color="gray")
    ax2.fill_between(tmp_idx, lCI, uCI, color="gray", alpha=0.5, label="99.9% CI")
    ax2.set_xlabel("n")
    ax2.set_xticks([tmp_idx[0], tmp_idx[-1]])
    ax2.set_xlim([tmp_idx[0], tmp_idx[-1]])
    ax2.set_ylabel(r"$\gamma_i = \rho_i \delta_i$")
    ax2.set_ylim([np.min(gammas), np.max(gammas)])
    ax2.set_yticks([np.min(gammas), np.max(gammas)])
    ax2.legend(frameon=False)
    sns.despine(ax=ax2, offset=5, trim=True)

    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_dendogram_silhouettes(clusters, linkage, silhouettes, fig_name):
    """Plots dendogram and silhouettes from Ward's hierarchical clustering"""

    n_clust = len(np.unique(clusters))
    ct = linkage[-(n_clust-1), 2]
    cmap = plt.cm.get_cmap("tab20", n_clust)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(2, 1, 1)
    set_link_color_palette([colors.to_hex(cmap(i)) for i in range(n_clust)])
    dendrogram(linkage, color_threshold=ct, above_threshold_color="gray",
               no_labels=True, ax=ax)
    ax.axhline(ct, color="red", ls="--", label="threshold: %.2f"%ct)
    ax.legend(frameon=False)

    ax2 = fig.add_subplot(2, 1, 2)
    sns.despine()
    x_lb = 0; xticks = []; xticklabels = []
    for i in range(n_clust):
        silhouettes_i = np.sort(silhouettes[clusters == i])
        x_ub = x_lb + silhouettes_i.shape[0]
        ax2.fill_between(np.arange(x_lb, x_ub), 0, silhouettes_i,
                         facecolor=cmap(i), edgecolor=cmap(i))
        xticks.append(x_lb + 0.5*silhouettes_i.shape[0]); xticklabels.append(i)
        x_lb = x_ub
    ax2.axhline(np.mean(silhouettes), color="gray", ls="--",
                label="avg. silhouettes score: %.2f"%np.mean(silhouettes))
    ax2.set_xticks(xticks); ax2.set_xticklabels(xticklabels)
    ax2.set_xlim([0, silhouettes.shape[0]])
    ax2.set_ylim([np.min(silhouettes), np.max(silhouettes)])
    ax2.set_yticks([np.min(silhouettes), np.max(silhouettes)])
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _group_by_patterns(clusters, t_bins, patterns):
    """reorders time series (of clusters) based on patterns
    returns a matrix for each pattern (max stim*max length)"""

    pattern_names, counts = np.unique(patterns, return_counts=True)
    max_count = np.max(counts)

    pattern_idx = _get_pattern_idx(t_bins)
    extended_pattern_idx = []
    for t_start, t_end in zip(pattern_idx[:-1], pattern_idx[1:]):
        extended_pattern_idx.append(np.arange(t_start, t_end))
    extended_pattern_idx.append(np.arange(pattern_idx[-1], len(t_bins)))
    max_len = np.max([len(idx) for idx in extended_pattern_idx])

    pattern_matrices = {pattern:np.full((max_count, max_len), np.nan) for pattern in pattern_names}
    row_idx = {pattern:0 for pattern in pattern_names}
    for pattern, idx in zip(patterns, extended_pattern_idx):
        clusters_slice = clusters[idx]
        pattern_matrices[pattern][row_idx[pattern], 0:len(clusters_slice)] = clusters_slice
        row_idx[pattern] += 1

    return pattern_idx, pattern_matrices, row_idx


def update(changed_image):
    """mpl hack to set 1 colorbar for multiple images"""
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


def plot_cluster_seqs(clusters, patterns, t_bins, fig_name):
    """plots sequence of time bins color coded by clusters"""

    cmap = plt.cm.get_cmap("tab20", len(np.unique(clusters)))
    images = []

    t_idx, pattern_matrices, row_idx = _group_by_patterns(clusters, t_bins, patterns)
    clusters = np.reshape(clusters, (1, len(clusters)))

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, 5, height_ratios=[1, 4, 4])
    ax = fig.add_subplot(gs[0, :])
    divider = make_axes_locatable(ax)
    i_base = ax.imshow(clusters, cmap=cmap, aspect="auto")
    images.append(i_base)
    cax = divider.new_vertical(size="50%", pad=0.1, pack_start=True)
    fig.add_axes(cax)
    ax.set_xticks(t_idx); ax.set_xticklabels(patterns)
    ax.xaxis.tick_top()
    ax.set_yticks([])

    for i, (name, matrix) in enumerate(pattern_matrices.items()):
        ax = fig.add_subplot(gs[1+np.floor_divide(i, 5), np.mod(i, 5)-5])
        i = ax.imshow(matrix, cmap=cmap, aspect="auto")
        images.append(i)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([0, row_idx[name]-1])

    # set one colorbar for all images
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(i_base, cax=cax, orientation="horizontal", ticks=np.unique(clusters))
    for im in images:
        im.callbacksSM.connect("changed", update)

    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_assemblies(core_cell_idx, assembly_idx, row_map, ystuff, depths, fig_name):
    """Plots depth profile of all assemblies"""

    cmap = plt.cm.get_cmap("tab20", core_cell_idx.shape[1])
    yrange = [ystuff["hlines"][-1], ystuff["hlines"][1]]

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(np.floor_divide(len(assembly_idx), 5)+1, 5)
    for i, assembly_id in enumerate(assembly_idx):
        gids = row_map[np.where(core_cell_idx[:, assembly_id]==1)[0]]
        assembly_depths = _gids_to_depth(gids, depths)

        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)-5])
        ax.hist(assembly_depths, bins=50, range=yrange, orientation="horizontal",
                color=cmap(assembly_id), edgecolor=cmap(assembly_id))
        for i in range(2, 6):
            ax.axhline(ystuff["hlines"][i], color="gray", ls="--")
        ax.set_title("Assembly %i (n=%i)" % (assembly_id, len(assembly_depths)))
        ax.set_xticks([])
        ax.set_yticks(ystuff["yticks"][1:])
        ax.set_ylim(yrange)
        ax.set_yticklabels([label[0:2] for label in ystuff["yticklabels"][1:]])
        sns.despine(bottom=True, offset=5)

    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_assembly_sim_matrix(sim_matrix, n_assemblies, fig_name):
    """Plots similarity matrix of assemblies"""

    sim_mat = deepcopy(sim_matrix)
    np.fill_diagonal(sim_mat, np.nan)
    n_assemblies_cum = [0] + np.cumsum(n_assemblies).tolist()

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(sim_mat, cmap="coolwarm",
                  aspect="auto", interpolation="none")
    fig.colorbar(i)
    ax.set_xticks(n_assemblies_cum)
    ax.set_yticks(n_assemblies_cum)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_single_cell_features(gids, r_spikes, mean_ts, std_ts, ystuff, depths, fig_name):
    """Plots spike time reliability and mean+/-std(spike time) in bin"""

    yrange = [ystuff["hlines"][-1], ystuff["hlines"][1]]
    gid_depths = _gids_to_depth(gids, depths)
    r_spikes[r_spikes == 0] = np.nan

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 2, 1)
    sns.despine(ax=ax, offset=5)
    ax.scatter(r_spikes, gid_depths,
               color="black", alpha=0.5, marker='.', s=5, edgecolor="none")
    for i in range(2, 6):
        ax.axhline(ystuff["hlines"][i], color="gray", ls="--")
    ax.set_xlabel("r_spike")
    ax.set_xlim([0, np.nanmax(r_spikes)])
    ax.set_yticks(ystuff["yticks"][1:])
    ax.set_ylim(yrange)
    ax.set_yticklabels([label[0:2] for label in ystuff["yticklabels"][1:]])
    ax2 = fig.add_subplot(1, 2, 2)
    sns.despine(ax=ax2, left=True, offset=5)
    ax2.errorbar(mean_ts, gid_depths, xerr=std_ts, color="black",
                 fmt="none", alpha=0.5, lw=0.1, errorevery=10)
    ax2.scatter(mean_ts, gid_depths,
               color="black", alpha=0.5, marker='.', s=5, edgecolor="none")
    for i in range(2, 6):
        ax2.axhline(ystuff["hlines"][i], color="gray", ls="--")
    ax2.set_xlabel("Spike time in bin (ms)")
    ax2.set_xlim([0, 10])
    ax2.set_ylim(yrange)
    ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_r_spike(consenus_r_spikes, r_spikes, fig_name):
    """Plots spike time reliability for consensus assemblies
    :param consenus_r_spikes: list of spike time reliabilities for all core cells in the consensus assemblies
    :param r_spikes: array of spikes time reliability (r_spike) for all cells"""

    n = len(consenus_r_spikes)
    cmap = plt.cm.get_cmap("tab20", n)
    max_len = np.max([assembly.shape[0] for assembly in consenus_r_spikes])
    widths = [assembly.shape[0]/max_len for assembly in consenus_r_spikes]

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 1])
    ax = fig.add_subplot(gs[0])
    parts = ax.violinplot(consenus_r_spikes, showextrema=False, widths=widths)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors.to_hex(cmap(i)))
        pc.set_edgecolor("black")
        pc.set_alpha(1)
    idx = np.arange(1, n+1)
    for i, consenus_rs in zip(idx, consenus_r_spikes):
        quartile1, median, quartile3 = np.percentile(consenus_rs, [25, 50, 75])
        ax.vlines(i, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.scatter(i, median, marker='o', color='white', s=30, edgecolor="none", zorder=3)
    ax.set_ylabel("r_spike")
    ax.set_xticks(idx)
    ax.set_xticklabels(["cons%s\n(n=%i)" % (i, consenus_r_spikes[i].shape[0])
                        for i in range(0, n)])
    ax.set_ylim([0, np.nanmax(r_spikes)])
    sns.despine(ax=ax, bottom=True, offset=5, trim=True)

    n_all = r_spikes.shape[0]
    r_spikes = r_spikes[~np.isnan(r_spikes)]
    ax2 = fig.add_subplot(gs[1])
    sns.despine(ax=ax2, left=True, bottom=True)
    parts = ax2.violinplot(r_spikes, showextrema=False, widths=[1])
    parts["bodies"][0].set_facecolor("gray")
    parts["bodies"][0].set_edgecolor("black")
    parts["bodies"][0].set_alpha(0.8)
    quartile1, median, quartile3 = np.percentile(r_spikes, [25, 50, 75])
    ax2.vlines(1, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax2.scatter(1, median, marker='o', color='white', s=30, edgecolor="none", zorder=3)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["all_gids\n(n=%i)" % n_all])
    ax2.set_ylim([0, np.nanmax(consenus_r_spikes[-1])])
    ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_t_in_bin(consensus_gids, all_gids, consenus_mean_ts, consenus_std_ts,
                            mean_ts, std_ts, ystuff, depths, fig_name):
    """Plots time in bin for consensus assemblies"""

    n = len(consenus_mean_ts)
    cmap = plt.cm.get_cmap("tab20", n)
    yrange = [ystuff["hlines"][-1], ystuff["hlines"][1]]

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, n+1)
    for i in range(n):
        ax = fig.add_subplot(gs[i])
        gid_depths = _gids_to_depth(consensus_gids[i], depths)
        color = colors.to_hex(cmap(i))
        errorevery = 10 if consensus_gids[i].shape[0] > 2000 else 1
        ax.errorbar(consenus_mean_ts[i], gid_depths, xerr=consenus_std_ts[i], color=color,
                    fmt="none", alpha=0.5, lw=0.1, errorevery=errorevery)
        ax.scatter(consenus_mean_ts[i], gid_depths,
                   color=color, alpha=0.5, marker='.', s=5, edgecolor="none")
        for j in range(2, 6):
            ax.axhline(ystuff["hlines"][j], color="gray", ls="--")
        ax.set_xlim([0, 10])
        ax.set_title("cons%s\n(n=%i)" % (i + 1, consensus_gids[i].shape[0]))
        ax.set_ylim(yrange)
        if i == 0:
            ax.set_yticks(ystuff["yticks"][1:])
            ax.set_yticklabels([label[0:2] for label in ystuff["yticklabels"][1:]])
            sns.despine(ax=ax, offset=5)
        else:
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, offset=5)

    ax = fig.add_subplot(gs[-1])
    gid_depths = _gids_to_depth(all_gids, depths)
    ax.errorbar(mean_ts, gid_depths, xerr=std_ts, color="black",
                fmt="none", alpha=0.5, lw=0.1, errorevery=10)
    ax.scatter(mean_ts, gid_depths,
               color="black", alpha=0.5, marker='.', s=5, edgecolor="none")
    for j in range(2, 6):
        ax.axhline(ystuff["hlines"][j], color="gray", ls="--")
    ax.set_xlim([0, 10])
    ax.set_title("all_gids\n(n=%i)" % all_gids.shape[0])
    ax.set_ylim(yrange)
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, offset=5)
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Spike time in bin (ms)")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)