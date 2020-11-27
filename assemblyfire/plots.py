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
    ax2.set_xlim([0, silh