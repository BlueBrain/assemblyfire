"""
Assembly detection related plots
author: András Ecker, last update: 01.2023
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from assemblyfire.utils import group_clusters_by_patterns, count_clusters_by_patterns_across_seeds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set(style="ticks", context="notebook")
PATTERN_COLORS = {"A": "#234091", "B": "#57B4D0", "C": "#C4A943", "D": "#7E1F19", "E": "#3F7AB3",
                  "F": "#8CAD8A", "G": "#A1632E", "H": "#66939D", "I": "#66939D", "J": "#665869"}
PROJ_COLORS = {"Thalamocortical_input_VPM": "tab:purple", "Thalamocortical_input_POM": "black"}
RED, BLUE = "#e32b14", "#3271b8"


def plot_rate(rate, rate_th, t_start, t_end, fig_name):
    """Plots thresholded rate"""
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    ax.plot(np.linspace(t_start, t_end, len(rate)), rate, "k-")
    ax.axhline(np.mean(rate) + rate_th, color="gray", ls="--")
    ax.set_xlim([t_start, t_end])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz)")
    # ax.set_ylim(bottom=0)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _get_pattern_idx(t_bins, stim_times, patterns):
    """Maps stimulus times to column idx in the spike_matrix"""
    t_idx, significant_patterns = [], []
    for pattern, t_start, t_end in zip(patterns, stim_times[:-1], stim_times[1:]):
        idx = np.where((t_start <= t_bins) & (t_bins < t_end))[0]
        if len(idx):
            t_idx.append(idx[t_bins[idx].argmin()])
            significant_patterns.append(pattern)
    return t_idx, significant_patterns


def plot_sim_matrix(sim_matrix, t_bins, stim_times, patterns, fig_name):
    """Plots similarity matrix"""
    t_idx, sign_patterns = _get_pattern_idx(t_bins, stim_times, patterns)
    np.fill_diagonal(sim_matrix, np.nan)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(sim_matrix, cmap="cividis", aspect="auto", interpolation="none")
    fig.colorbar(i)
    ax.set_xticks(t_idx); ax.set_xticklabels(sign_patterns)
    ax.xaxis.tick_top()
    ax.set_xlabel("time bins")
    ax.set_yticks(t_idx); ax.set_yticklabels(sign_patterns)
    ax.set_ylabel("time bins")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_sims_vs_rate_and_tdiff(sim_matrix, pw_avg_rate, t_offsets, mean_sims, fig_name):
    """Plots (pairwise) similarity vs. pairwise average firing rate and
    mean similarity (within time window) against increasing temporal offset"""
    t_offsets = t_offsets / 1000  # convert to sec
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t_offsets, mean_sims, color=BLUE)
    # ax.set_xlim([t_offsets[0], t_offsets[-1]])
    ax.set_xlabel("Minimum time difference (s)")
    ax.set_ylabel("Mean similarity (within time window)")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(pw_avg_rate, sim_matrix, c=BLUE, alpha=0.8, marker='.', s=10, edgecolor="none")
    ax2.set_xlabel("Pairwise avg. rate (Hz)")
    ax2.set_ylabel("Similarity")
    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_dendogram_silhouettes(clusters, linkage, silhouettes, fig_name):
    """Plots dendogram and silhouettes from Ward's hierarchical clustering"""
    n_clust = len(np.unique(clusters))
    ct = linkage[-(n_clust-1), 2]
    cmap = plt.cm.get_cmap("tab20", n_clust)

    if silhouettes is None:
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        x_lb = 0
        xticks, xticklabels = [], []
        for i in range(n_clust):
            silhouettes_i = np.sort(silhouettes[clusters == i])
            x_ub = x_lb + silhouettes_i.shape[0]
            ax2.fill_between(np.arange(x_lb, x_ub), 0, silhouettes_i,
                             facecolor=cmap(i), edgecolor=cmap(i))
            xticks.append(x_lb + 0.5*silhouettes_i.shape[0])
            xticklabels.append(i)
            x_lb = x_ub
        ax2.axhline(np.mean(silhouettes), color="gray", ls="--",
                    label="avg. silhouettes score: %.2f" % np.mean(silhouettes))
        ax2.set_xticks(xticks); ax2.set_xticklabels(xticklabels)
        ax2.set_xlim([0, silhouettes.shape[0]])
        ax2.set_ylim([np.min(silhouettes), np.max(silhouettes)])
        ax2.set_yticks([np.min(silhouettes), np.max(silhouettes)])
        ax2.legend(frameon=False)
        fig.tight_layout()

    set_link_color_palette([colors.to_hex(cmap(i)) for i in range(n_clust)])
    dendrogram(linkage, color_threshold=ct, above_threshold_color="gray",
               no_labels=True, ax=ax)
    ax.axhline(ct, color="red", ls="--", label="threshold: %.2f" % ct)
    ax.legend(frameon=False)
    sns.despine()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def update(changed_image):
    """mpl hack to set 1 colorbar for multiple images"""
    for im in images:
        if changed_image.get_cmap() != im.get_cmap() or changed_image.get_clim() != im.get_clim():
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


def plot_cluster_seqs(clusters, t_bins, stim_times, patterns, fig_name):
    """Plots sequence of time bins color coded by clusters"""
    cmap = plt.cm.get_cmap("tab20", len(np.unique(clusters)))
    images = []

    t_idx, sign_patterns = _get_pattern_idx(t_bins, stim_times, patterns)
    max_t, row_idx, pattern_matrices, _ = group_clusters_by_patterns(clusters, t_bins, stim_times, patterns)
    clusters = np.reshape(clusters, (1, len(clusters)))

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, 5, height_ratios=[1, 4, 4])
    ax = fig.add_subplot(gs[0, :])
    divider = make_axes_locatable(ax)
    i_base = ax.imshow(clusters, cmap=cmap, interpolation="nearest", aspect="auto")
    images.append(i_base)
    cax = divider.new_vertical(size="50%", pad=0.1, pack_start=True)
    fig.add_axes(cax)
    ax.set_xticks(t_idx); ax.set_xticklabels(sign_patterns)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for i, (pattern, matrix) in enumerate(pattern_matrices.items()):
        ax = fig.add_subplot(gs[1+np.floor_divide(i, 5), np.mod(i, 5)])
        im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", aspect="auto")
        images.append(im)
        ax.set_title(pattern)
        if np.floor_divide(i, 5) == 1:
            ax.set_xticks([0, matrix.shape[1] - 1])
            ax.set_xticklabels([0, max_t])
        else:
            ax.set_xticks([])
        ax.set_yticks([0, row_idx[pattern]])
    # set one colorbar for all images
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(i_base, cax=cax, orientation="horizontal", ticks=np.unique(clusters))
    for im in images:
        im.callbacks.connect("changed", update)
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Binned time (ms)")
    plt.ylabel("Pattern presentation")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_cons_cluster_seqs(clusters, t_bins, stim_times, patterns, n_clusters, fig_name):
    """Plots sequence of time bins color coded by consensus assemblies
    (+black if the orig cluster didn't become an assembly)"""
    cmap_tmp = plt.cm.get_cmap("tab20", n_clusters)
    cols = [(0.0, 0.0, 0.0, 0.8)] + [cmap_tmp(i) for i in range(n_clusters)]
    cmap = colors.ListedColormap(cols)
    bounds = np.arange(-1.5, n_clusters)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    t_idx, sign_patterns = _get_pattern_idx(t_bins, stim_times, patterns)
    max_t, row_idx, pattern_matrices, _ = group_clusters_by_patterns(clusters, t_bins, stim_times, patterns)
    clusters = np.reshape(clusters, (1, len(clusters)))

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, 5, height_ratios=[1, 4, 4])
    ax = fig.add_subplot(gs[0, :])
    divider = make_axes_locatable(ax)
    i_base = ax.imshow(clusters, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    cax = divider.new_vertical(size="50%", pad=0.1, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(i_base, cax=cax, orientation="horizontal", ticks=np.arange(-1, n_clusters))
    ax.set_xticks(t_idx); ax.set_xticklabels(sign_patterns)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for i, (pattern, matrix) in enumerate(pattern_matrices.items()):
        ax = fig.add_subplot(gs[1+np.floor_divide(i, 5), np.mod(i, 5)])
        ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        ax.set_title(pattern)
        if np.floor_divide(i, 5) == 1:
            ax.set_xticks([0, matrix.shape[1] - 1])
            ax.set_xticklabels([0, max_t])
        else:
            ax.set_xticks([])
        ax.set_yticks([0, row_idx[pattern]])
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Binned time (ms)")
    plt.ylabel("Pattern presentation")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_cons_cluster_seqs_all_seeds(clusters, t_bins, stim_times, patterns, n_clusters, fig_name):
    """Plots heatmaps of consensus assembly counts across seeds"""
    count_matrices, seeds, cons_assembly_idx, _ = count_clusters_by_patterns_across_seeds(clusters, t_bins,
                                                                                          stim_times, patterns, n_clusters)
    vmax = np.max([np.max(count_matrix) for _, count_matrix in count_matrices.items()])

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 5)
    for i, (pattern, matrix) in enumerate(count_matrices.items()):
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        im = sns.heatmap(matrix, cmap="inferno", mask=matrix == 0, ax=ax,
                         vmin=1, vmax=vmax, annot=True, cbar=False, xticklabels=cons_assembly_idx, yticklabels=seeds)
        for _, spine in im.spines.items():
            spine.set_visible(True)
        ax.set_title(pattern)
        if np.floor_divide(i, 5) == 0:
            ax.set_xticks([])
        if np.mod(i, 5) != 0:
            ax.set_yticks([])
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Consensus assembly id")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_pattern_clusters(clusters, t_bins, stim_times, patterns, fig_name):
    """Plots counts of clusters for every pattern"""
    n = len(np.unique(clusters))
    x = np.arange(n)
    cmap = plt.cm.get_cmap("tab20", n)
    cols = [colors.to_hex(cmap(i)) for i in range(n)]
    _, _, _, pattern_counts = group_clusters_by_patterns(clusters, t_bins, stim_times, patterns)

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 5)
    for i, (pattern_name, counts) in enumerate(pattern_counts.items()):
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        ax.bar(x, counts, width=0.5, align="center", color=cols)
        ax.set_title(pattern_name)
        ax.set_xticks(x)
        ax.set_xlim([-0.5, n-0.5])
    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_distance_corr(input_dist, output_dist, fig_name):
    """Plots input vs. output distances"""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    sns.regplot(x=input_dist, y=output_dist, color="black", marker='.', ax=ax)
    ax.set_xlabel("Input distance")
    ax.set_ylabel("Output distance")
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_db_scores(n, db_scores, fig_name):
    """Plots DB scores against n-clusters"""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(n, db_scores, color="black")
    ax.axvline(n[np.argmin(db_scores)], color="red", ls="--")
    ax.set_xlim([n[0], n[-1]])
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("D-B score")
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _get_depth_yticks(loc_df):
    """Gets mean depth for each layer (used for setting ticks for depth based plots)"""
    gb_locs = loc_df.groupby("layer")
    yticks = gb_locs["depth"].mean().to_numpy()
    yticklabels = ["L%i\n(%i)" % (l, n) for l, n in gb_locs.size().items()]
    return yticks, yticklabels


def plot_assemblies(core_cell_idx, assembly_idx, gids, loc_df, fig_name):
    """Plots depth profile of all assemblies"""
    cmap = plt.cm.get_cmap("tab20", core_cell_idx.shape[1])
    n = len(assembly_idx)
    yticks, yticklabels = _get_depth_yticks(loc_df)
    x_range = [loc_df["ss_flat_x"].min(), loc_df["ss_flat_x"].max()]
    y_range = [loc_df["ss_flat_y"].min(), loc_df["ss_flat_y"].max()]
    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    depth_range = [loc_df["depth"].min(), loc_df["depth"].max()]

    fig = plt.figure(figsize=(18, 10))
    n_rows = np.floor_divide(n, 5) + 1 if np.mod(n, 5) != 0 else int(n/5)
    gs = gridspec.GridSpec(2 * n_rows, 5)
    for i, assembly_id in enumerate(assembly_idx):
        assembly_gids = gids[core_cell_idx[:, assembly_id] == 1]
        ax = fig.add_subplot(gs[2 * np.floor_divide(i, 5), np.mod(i, 5)])
        ax.hexbin(loc_df.loc[assembly_gids, "ss_flat_x"].to_numpy(), loc_df.loc[assembly_gids, "ss_flat_y"].to_numpy(),
                  cmap=colors.LinearSegmentedColormap.from_list("assembly", [(1, 1, 1), cmap(i)], N=5),
                  gridsize=50, bins="log", extent=extent)
        ax.set_aspect("equal", "box")
        ax.set_title("Assembly %i (n=%i)" % (assembly_id, len(assembly_gids)))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(x_range); ax.set_ylim(y_range)
        ax2 = fig.add_subplot(gs[2 * np.floor_divide(i, 5) + 1, np.mod(i, 5)])
        ax2.hist(loc_df.loc[assembly_gids, "depth"].to_numpy(), bins=50, range=depth_range, orientation="horizontal",
                 color=cmap(assembly_id), edgecolor=cmap(assembly_id))
        ax2.set_xticks([])
        ax2.set_yticks(yticks)
        ax2.set_ylim(depth_range[::-1])
        ax2.set_yticklabels([label[0:2] for label in yticklabels])
        sns.despine(ax=ax, bottom=True, left=True)
        sns.despine(ax=ax2, bottom=True, offset=5)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_efficacy(efficacies, fig_name):
    """Plots efficacies (depressed and potentiated) for synapses within assemblies (within one seed)"""
    plt.rcParams["patch.edgecolor"] = "black"
    assembly_labels = np.sort(list(efficacies.keys()))
    n = len(assembly_labels)

    fig = plt.figure(figsize=(20, 8))
    n_rows = np.floor_divide(n, 5) + 1 if np.mod(n, 5) != 0 else int(n/5)
    gs = gridspec.GridSpec(n_rows, 5)
    for i, assembly_label in enumerate(assembly_labels):
        sizes = np.array([efficacies[assembly_label][0], efficacies[assembly_label][1]])
        ratios = 100 * sizes / np.sum(sizes)
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        ax.pie(sizes, labels=["%.2f%%" % ratio for ratio in ratios], colors=[BLUE, RED])
        ax.set_title("Assembly %i\n(nsyns=%.2fM)" % (assembly_label, (sizes[0]+sizes[1]) / 1e6))
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams["patch.edgecolor"] = "white"


def plot_tc_syn_properties(df, fig_name):
    """Plots synapse (to a given assembly) properties of projections"""
    df.loc[df["layer"].isin([2, 3]), "layer"] = 23
    order = [23, 4, 5, 6]
    hue_order = np.sort(df["name"].unique())
    palette = []
    for name in hue_order:
        color = PATTERN_COLORS[name] if name in PATTERN_COLORS else "black"
        palette.append(colors.to_rgb(color))

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(2, 1, 1)
    sns.violinplot(x="layer", y="g_syn", hue="name", order=order, hue_order=hue_order,
                   scale="count", bw="silverman", cut=1, inner=None, palette=palette, data=df, ax=ax)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc=2, ncol=3, fancybox=True)
    ax.set_xlabel('')
    ax.set_ylabel("G_syn (nS)")
    ax2 = fig.add_subplot(2, 1, 2)
    sns.violinplot(x="layer", y="path distance", hue="name", order=order, hue_order=hue_order,
                   scale="count", bw="silverman", cut=1, inner=None, palette=palette, data=df, ax=ax2)
    ax2.legend([], [], frameon=False)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Path distance (um)")
    sns.despine(bottom=True, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_in_degrees(in_degrees, in_degrees_control, fig_name, xlabel="In degree"):
    """Plots in degrees for assemblies (within one seed) and random controls"""
    assembly_labels = list(in_degrees.keys())
    n = len(assembly_labels)
    cmap = plt.cm.get_cmap("tab20", np.max([assembly_label[0] for assembly_label in assembly_labels])+1)

    fig = plt.figure(figsize=(20, 8))
    n_rows = np.floor_divide(n, 5) + 1 if np.mod(n, 5) != 0 else int(n/5)
    gs = gridspec.GridSpec(n_rows, 5)
    for i, assembly_label in enumerate(assembly_labels):
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        max_in_degree = np.max(in_degrees[assembly_label])
        ax.hist(in_degrees[assembly_label], bins=50, range=(0, max_in_degree),
                color=cmap(assembly_label[0]), edgecolor=cmap(assembly_label[0]), label="assembly")
        ax.hist(in_degrees_control["n"][assembly_label], bins=50, range=(0, max_in_degree),
                color="black", histtype="step", linestyle="dashed", label="ctrl. n neurons")
        ax.hist(in_degrees_control["depths"][assembly_label], bins=50, range=(0, max_in_degree),
                color="black", histtype="step", linestyle="dashdot", label="ctrl. depth profile")
        ax.hist(in_degrees_control["mtypes"][assembly_label], bins=50, range=(0, max_in_degree),
                color="black", histtype="step", label="ctrl. mtype comp.")
        ax.set_title("Assembly %i" % assembly_label[0])
        ax.set_xlim([0, max_in_degree])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, offset=5)
        if i == 0:
            ax.legend(frameon=False)
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_simplex_counts(simplex_counts, simplex_counts_control, fig_name):
    """Plots simplex counts for assemblies (within one seed) and random controls"""
    assembly_labels = list(simplex_counts.keys())
    n = len(assembly_labels)
    cmap = plt.cm.get_cmap("tab20", np.max([assembly_label[0] for assembly_label in assembly_labels]) + 1)

    fig = plt.figure(figsize=(20, 8))
    n_rows = np.floor_divide(n, 5) + 1 if np.mod(n, 5) != 0 else int(n/5)
    gs = gridspec.GridSpec(n_rows, 5)
    for i, assembly_label in enumerate(assembly_labels):
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        ax.plot(simplex_counts[assembly_label], color=cmap(assembly_label[0]), lw=3, label="assembly")
        ax.plot(simplex_counts_control["n"][assembly_label], color="black", lw=1, ls="--", label="ctrl. n neurons")
        ax.plot(simplex_counts_control["depths"][assembly_label], color="black", lw=1, ls="-.",
                label="ctrl. depth profile")
        ax.plot(simplex_counts_control["mtypes"][assembly_label], color="black", lw=1, label="ctrl. mtype comp.")
        ax.set_title("Assembly %i" % assembly_label[0])
        ax.set_yticks([])
        ax.set_xlim([0, 6])  # TODO not hard code this
        sns.despine(ax=ax, left=True, offset=5)
        if i == 0:
            ax.legend(frameon=False)
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Simplex dimension")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_assembly_prob_from(bin_centers, assembly_probs, assembly_probs_low, assembly_probs_high, chance_levels,
                            xlabel, palette, fig_name, logx=False):
    """Plots assembly membership probability vs. whatever data `xlabel` is"""
    keys = list(assembly_probs.keys())
    assembly_labels = list(assembly_probs[keys[0]].keys())
    n = len(assembly_labels)
    n_rows = np.floor_divide(n, 5) + 1 if np.mod(n, 5) != 0 else int(n / 5)
    if palette == "patterns":
        palette = PATTERN_COLORS
        ncol = 2
    elif palette == "projections":
        palette = PROJ_COLORS
        ncol = 1
    else:
        cmap = plt.cm.get_cmap("tab20", np.max([assembly_label for assembly_label in assembly_labels]) + 1)
        ncol = n_rows if "gray" not in list(palette.values()) else 2

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(n_rows, 5)
    for i, assembly_label in enumerate(assembly_labels):
        ax = fig.add_subplot(gs[i])
        ax.axhline(chance_levels[assembly_label], linestyle="--", color="lightgray", label="chance level")
        for j, key in enumerate(keys):
            if palette[key] == "assembly_color":
                color = cmap(i)
            elif palette[key] == "pre_assembly_color":
                color = cmap(j)
            else:
                color = palette[key]
            ax.plot(bin_centers[key][assembly_label], assembly_probs[key][assembly_label], color=color, label=key)
            ax.fill_between(bin_centers[key][assembly_label], assembly_probs_low[key][assembly_label],
                            assembly_probs_high[key][assembly_label], color=color, alpha=0.1)
            if i == 0:
                ax.legend(frameon=False, ncol=ncol)
        ax.set_title("Assembly %s" % assembly_label)
        if logx:
            ax.set_xscale("log")
        # else:
            # ax.set_xlim(left=0)
        ax.set_ylim([0, 1])
    sns.despine(trim=True, offset=2)
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel("Prob. of assembly membership")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_assembly_n_from(bin_centers, assembly_probs, assembly_probs_low, assembly_probs_high, xlabel, palette, fig_name):
    """Similar to the above, but it's a single plot (as it's not assembly specific) and the ylabel is different"""
    if palette == "projections":
        palette = PROJ_COLORS
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    for key, assembly_prob in assembly_probs.items():
        color = palette[key]
        ax.plot(bin_centers[key], assembly_prob, color=color, label=key)
        ax.fill_between(bin_centers[key], assembly_probs_low[key], assembly_probs_high[key], color=color, alpha=0.1)
    ax.legend(frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0)
    ax.set_ylabel("Nr. of assemblies a neuron is part of")
    ax.set_ylim(bottom=0)
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_frac_entropy_explained_by(mi_df, ylabel, fig_name):
    """Plots matrix of entropy explained by innervation (by patterns or internal connections)"""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    if np.any(mi_df.to_numpy() < 0):  # relative case
        abs_max = np.nanmax(mi_df.abs().to_numpy())
        i = ax.imshow(mi_df, cmap="coolwarm", aspect="auto", interpolation="none", vmin=-1*abs_max, vmax=abs_max)
        fig.colorbar(i, label="Relative loss in entropy")
    else:  # 'classical' case
        i = ax.imshow(mi_df, cmap="inferno", aspect="auto", interpolation="none")
        fig.colorbar(i, label="Loss in entropy")
    ax.set_xticks(np.arange(len(mi_df.columns)))
    ax.set_xticklabels(mi_df.columns.to_numpy())
    ax.set_xlabel("Assembly")
    ax.set_yticks(np.arange(len(mi_df.index)))
    ax.set_yticklabels(mi_df.index.to_numpy())
    ax.set_ylabel(ylabel)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")


def plot_joint_dists(vars_x, vars_y, xlabel, ylabel, fig_name):
    """Plots joint distibution of 2 variables (with hexbins and log count)"""
    idx = np.logical_and(~np.isnan(vars_x), ~np.isnan(vars_y))
    vars_x, vars_y = vars_x[idx], vars_y[idx]
    extent = (np.percentile(vars_x, 0.1), np.percentile(vars_x, 99.9),
              np.percentile(vars_y, 0.1), np.percentile(vars_y, 99.9))
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.hexbin(vars_x, vars_y, gridsize=25, bins="log", extent=extent, cmap="inferno")
    ax.set_xlim([extent[0], extent[1]])
    ax.set_xlabel(xlabel)
    ax.set_ylim([extent[2], extent[3]])
    ax.set_ylabel(ylabel)
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")


def plot_assembly_sim_matrix(sim_matrix, n_assemblies, fig_name):
    """Plots similarity matrix of assemblies"""
    np.fill_diagonal(sim_matrix, np.nan)
    n_assemblies_cum = [0] + np.cumsum(n_assemblies).tolist()

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(sim_matrix, cmap="cividis", aspect="auto", interpolation="none",
                  extent=(0, sim_matrix.shape[1], sim_matrix.shape[0], 0))
    fig.colorbar(i)
    ax.set_xticks(n_assemblies_cum)
    ax.set_yticks(n_assemblies_cum)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_mtypes(mtypes, core_mtypes, union_mtypes, fig_name):
    """Plots mtype composition of consensus assemblies' core and union"""
    n = len(core_mtypes)
    cmap = plt.cm.get_cmap("tab20", n)
    mtypes_lst = np.unique(mtypes)[::-1]
    mtypes_ypos = np.arange(len(mtypes_lst))

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, n+1)
    for i in range(n):
        ax = fig.add_subplot(gs[i])
        color = colors.to_hex(cmap(i))
        mtypes_plot = [np.where(core_mtypes[i] == mtype)[0].shape[0] for mtype in mtypes_lst]
        ax.barh(mtypes_ypos, mtypes_plot, color=color, edgecolor=color)
        mtypes_plot = [np.where(union_mtypes[i] == mtype)[0].shape[0] for mtype in mtypes_lst]
        ax.barh(mtypes_ypos, mtypes_plot, color="none", edgecolor="black")
        ax.set_title("cons%s\n(n=%i)" % (i, core_mtypes[i].shape[0]))
        # ax.set_xlim(left=5)
        if i == 0:
            ax.set_xticks([])
            ax.set_yticks(mtypes_ypos)
            ax.set_yticklabels(mtypes_lst)
            sns.despine(ax=ax, bottom=True, offset=5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)

    ax = fig.add_subplot(gs[-1])
    mtypes_plot = [np.where(mtypes == mtype)[0].shape[0] for mtype in mtypes_lst]
    ax.barh(mtypes_ypos, mtypes_plot, color="gray", edgecolor="gray")
    ax.set_title("all gids\n(n=%i)" % mtypes.shape[0])
    # ax.set_xlim(left=5)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_r_spikes(gids, r_spikes, loc_df, fig_name):
    """Plots spike time reliability (for all gids)"""
    yticks, yticklabels = _get_depth_yticks(loc_df)
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(r_spikes, loc_df.loc[gids, "depth"].to_numpy(),
               color="black", alpha=0.5, marker='.', s=5, edgecolor="none")
    ax.set_xlim([np.min(r_spikes), np.max(r_spikes)])
    ax.set_xlabel("r_spike")
    ax.set_ylim([loc_df["depth"].max(), loc_df["depth"].min()])
    ax.set_yticks(yticks)
    ax.set_yticklabels([label[0:2] for label in yticklabels])
    sns.despine(offset=2)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_r_spikes(df, fig_name):
    """Plots spike time reliability for consensus assemblies"""
    yrange = [df["r_spike"].min(), df["r_spike"].max()]
    df_cons_assembly = df.loc[df["consensus assembly id"] != "non assembly"]
    order = np.sort(df_cons_assembly["consensus assembly id"].unique().astype(int))
    cmap = plt.cm.get_cmap("tab20", len(order))
    palette = [colors.to_rgb(cmap(i)) for i in order]
    order = order.astype(str)

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[len(order), 1])
    ax = fig.add_subplot(gs[0])
    sns.violinplot(x="consensus assembly id", y="r_spike", order=order,
                   scale="count", bw="silverman", cut=1, inner=None, palette=palette, data=df_cons_assembly, ax=ax)
    ax.legend([], [], frameon=False)
    ax.set_xlabel('')
    ax.set_ylim(yrange)
    ax2 = fig.add_subplot(gs[1])
    sns.violinplot(x="consensus assembly id", y="r_spike", scale="count", bw="silverman", cut=1, inner=None,
                   color="black", data=df.loc[df["consensus assembly id"] == "non assembly"], ax=ax2)
    ax2.legend([], [], frameon=False)
    ax2.set_xlabel('')
    ax2.set_ylim(yrange)
    sns.despine(bottom=True, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_coreness_r_spike(r_spikes, coreness, fig_name):
    """Plots corenss vs. spike time reliability"""
    n = len(coreness)
    cmap = plt.cm.get_cmap("tab20", n)
    max_r = np.max([np.max(r_spikes_) for r_spikes_ in r_spikes])

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(np.floor_divide(n, 5)+1, 5)
    for i in range(n):
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        ax.scatter(coreness[i], r_spikes[i], color=cmap(i), marker='.', s=10, edgecolor="none")
        ax.set_title("union%i\n(n=%i)" % (i+1, coreness[i].shape[0]))
        ax.axvline(4., color="gray", ls="--")
        ax.set_xticks([0, 4, 5])
        ax.set_xlim([0, 5.1])
        ax.set_yticks([0, 0.3, 0.6])
        ax.set_ylim([0, max_r])
        sns.despine(ax=ax, offset=True, trim=True)
    fig.tight_layout()
    # fig.add_subplot(1, 1, 1, frameon=False)
    # plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    # plt.xlabel("Coreness")
    # plt.ylabel("r_spike")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_vs_average_assembly_composition(intersection_at_n, diff_at_n, fig_name):
    """TODO"""
    x = np.arange(len(intersection_at_n)) + 1
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, intersection_at_n, color="red", edgecolor="black", lw=0.5, label="consensus & average")
    ax.bar(x, diff_at_n, bottom=intersection_at_n, color="gray", edgecolor="black", lw=0.5, label="consensus \\ average")
    ax.legend(frameon=False)
    ax.set_xlabel("Times contained in consensus instantiations")
    ax.set_yscale("log")
    ax.set_ylim(bottom=100)
    ax.set_ylabel("Count")
    sns.despine(trim=True, offset=2, bottom=True)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_simplex_counts_consensus(simplex_counts, simplex_counts_control, fig_name):
    """Plots simplex counts for all instantiations of a consensus assembly
    and random control with the same size as the mean number of neurons in the instantiations"""
    n = len(simplex_counts)
    cmap = plt.cm.get_cmap("tab20", n)
    fig = plt.figure(figsize=(20, 8))
    n_rows = np.floor_divide(n, 5) + 1 if np.mod(n, 5) != 0 else int(n/5)
    gs = gridspec.GridSpec(n_rows, 5)
    # for i, (label, simplex_counts_cons) in enumerate(simplex_counts.items()):
    for i in range(n):
        label = "cluster%i" % i
        simplex_counts_cons = simplex_counts[label]
        ax = fig.add_subplot(gs[np.floor_divide(i, 5), np.mod(i, 5)])
        for simlex_count_inst in simplex_counts_cons:
            ax.plot(simlex_count_inst, color=cmap(i))
        for ctrl in simplex_counts_control[label]:
            ax.plot(ctrl, color="black", lw=0.5, ls="--")
        ax.set_title("cons%i\n(n=%s)" % (i, len(simplex_counts_cons)))
        ax.set_yticks([])
        ax.set_xlim([0, 6])  # TODO not hard code this
        sns.despine(ax=ax, left=True, offset=5)
    fig.add_subplot(1, 1, 1, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Simplex dimension")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_n_assemblies(stim_times, patterns, n_assemblies, t_chunks, fig_name):
    """Plots number of unique assemblies for every pattern presented through time"""
    stim_times /= 1000; t_chunks /= 1000  # ms to sec conversion
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    for pattern, color in PATTERN_COLORS.items():
        idx = np.where(patterns == pattern)[0]
        ax.scatter(stim_times[idx], n_assemblies[idx], color=color, s=20, edgecolor="none", label=pattern)
    for t_chunk in t_chunks:
        ax.axvline(t_chunk, color="gray", ls='--', alpha=0.5)
    ax.legend(title="Patterns", frameon=False)
    ax.set_xlim([t_chunks[0], t_chunks[-1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("#Unique assemblies")
    sns.despine()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_assembly_similarities(similarities, xlabel, ylabel, fig_name):
    """Plots similarity matrix of assemblies"""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(similarities, cmap="inferno", aspect="auto", interpolation="none",)
    fig.colorbar(i, label="Jaccard similarity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_across_conditions(df, y, fig_name):
    """Plots `y` features of spikes, across different condition (rerun in BGLibPy)"""
    order = np.sort(df["assembly"].unique())
    hue_order = ["passivedend", "baseline", "noNMDA"]
    palette = {"passivedend": "tab:blue", "baseline": "gray", "noNMDA": "tab:purple"}
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(x="assembly", y=y, hue="condition", order=order, hue_order=hue_order,
                palette=palette, fliersize=0, data=df, ax=ax)
    sns.stripplot(x="assembly", y=y, hue="condition", order=order, hue_order=hue_order,
                  dodge=True, palette=["black" for _ in range(len(hue_order))],
                  data=df.loc[df["member"] == 1], ax=ax, legend=False)
    sns.stripplot(x="assembly", y=y, hue="condition", order=order, hue_order=hue_order,
                  dodge=True, palette=["red" for _ in range(len(hue_order))],
                  data=df.loc[df["member"] == 0], ax=ax, legend=False)
    sns.despine(trim=True, offset=2, bottom=True)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_synapse_distance_dist(bin_edges, hist, cum, fit, fig_name):
    """Plots distribution of distances between synapses (on the same section)"""
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(bin_centers, hist, color="gray", edgecolor="black", lw=0.5)
    ax.set_xlim([bin_edges[0], bin_edges[-1]])
    ax.set_xlabel("Inter-synapse distance (um)")
    ax.set_ylabel("Count")
    ax2 = ax.twinx()
    ax2.step(bin_centers, cum, "k-", where="mid")
    ax2.plot(bin_centers, np.polyval(fit, bin_centers), "r-")
    ax2.set_ylabel("Cumulative distribution")
    sns.despine(right=False, offset=5, trim=True)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_synapse_clusters(morph, cluster_df, xyz, fig_name):
    """Plots neuron morphology and (10 biggest) detected synapse clusters on it"""
    from neurom import load_morphology, NeuriteType
    from neurom.view import matplotlib_impl

    cmap = plt.cm.get_cmap("tab20", 10)
    labels = [label for label in list(cluster_df.columns) if label not in xyz]
    n = len(labels)
    fig = plt.figure(figsize=(n*4, 6))
    for i, label in enumerate(labels):
        ax = fig.add_subplot(1, n, i+1)
        matplotlib_impl.plot_morph(neurite_type=(NeuriteType.apical_dendrite, NeuriteType.basal_dendrite),
                                   ax=ax, morph=load_morphology(morph), color="black")
        ax.set_title(label)
        ax.set_xlabel("x (um)")
        df = cluster_df.loc[cluster_df[label] >= 0]
        if len(df):
            clusters, counts = np.unique(df[label].to_numpy(), return_counts=True)
            clusters = clusters[np.argsort(counts)[::-1]]  # sort them based on size (decreasing order)
            clusters = clusters[:10] if len(clusters) > 10 else clusters  # take only 10 biggest
            for j, cluster in enumerate(clusters):
                ax.scatter(df.loc[df[label] == cluster, xyz[0]].to_numpy(),
                           df.loc[df[label] == cluster, xyz[1]].to_numpy(),
                           color=cmap(j), s=20, label="cluster%i" % cluster)
            ax.legend(frameon=False)
        if i == 0:
            ax.set_ylabel("y (um)")
            sns.despine(ax=ax, trim=True)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def get_michelson_contrast(cluster_dfs):
    """
    Groups efficacies (initial rhos) in sample neurons from assemblies based on
    2 criteria: 1) synapse is coming from assembly neuron vs. non-assembly neuron (fist +/-)
                2) synapse is part of a synapse cluster vs. not (second +/-)
    and gets Michelson contrasts (aka. visibility) defined as:
    P(pot/dep | condition) - P(pot/dep) / (P(pot/dep | condition) + P(pot/dep))
    """
    assembly_labels = np.sort(list(cluster_dfs.keys()))
    probs, pot_contrasts, dep_contrasts = {}, {}, {}
    for assembly_label in assembly_labels:
        df = cluster_dfs[assembly_label]
        uncond_probs = df["rho"].value_counts(normalize=True)
        probs[assembly_label] = uncond_probs.to_numpy()
        cond_probs, pot_contrast, dep_contrast = {}, np.zeros((2, 2)), np.zeros((2, 2))
        cond_probs["++"] = df.loc[df["assembly%i" % assembly_label] >= 0, "rho"].value_counts(normalize=True)
        cond_probs["+-"] = df.loc[df["assembly%i" % assembly_label] == -1, "rho"].value_counts(normalize=True)
        cond_probs["-+"] = df.loc[df["non_assembly"] >= 0, "rho"].value_counts(normalize=True)
        cond_probs["--"] = df.loc[df["non_assembly"] == -1, "rho"].value_counts(normalize=True)
        for i, assembly in enumerate(["+", "-"]):
            for j, clustered in enumerate(["+", "-"]):
                cond = assembly + clustered
                p_dep, p_pot = uncond_probs.loc[0], uncond_probs.loc[1]
                if len(cond_probs[cond]) == 2:
                    p_dep_cond, p_pot_cond = cond_probs[cond].loc[0], cond_probs[cond].loc[1]
                    pot_contrast[i, j] = (p_pot_cond - p_pot) / (p_pot_cond + p_pot)
                    dep_contrast[i, j] = (p_dep_cond - p_dep) / (p_dep_cond + p_dep)
                else:  # this could happen if no synapse clusters are found...
                    pot_contrast[i, j], dep_contrast[i, j] = np.nan, np.nan
        pot_contrasts[assembly_label], dep_contrasts[assembly_label] = pot_contrast, dep_contrast
    return probs, pot_contrasts, dep_contrasts


def plot_cond_rhos(cluster_dfs, fig_name):
    """For every assembly plots pie chart with initial rhos in sample neurons and 2 matrices
    with the cond. prob of being potentiated and depressed (in a 2x2 grid - see `get_michelson_contrast()` above)"""
    plt.rcParams["patch.edgecolor"] = "black"
    neg_colors = plt.cm.Greys_r(np.linspace(0, 1, 128))
    pot_colors = plt.cm.Reds(np.linspace(0, 1, 128))
    dep_colors = plt.cm.Blues(np.linspace(0, 1, 128))
    pot_cmap = colors.LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, pot_colors)))
    dep_cmap = colors.LinearSegmentedColormap.from_list("dep_cmap", np.vstack((neg_colors, dep_colors)))
    pot_cmap.set_bad(color="tab:pink")
    dep_cmap.set_bad(color="tab:pink")

    probs, pot_matrices, dep_matrices = get_michelson_contrast(cluster_dfs)
    assembly_labels = np.sort(list(probs.keys()))
    n = len(assembly_labels)
    pot_extr = np.max([np.nanmax(np.abs(pot_matrix)) for _, pot_matrix in pot_matrices.items()])
    dep_extr = np.max([np.nanmax(np.abs(dep_matrix)) for _, dep_matrix in dep_matrices.items()])

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, n+1, width_ratios=[10 for i in range(n)] + [1])
    for i, assembly_label in enumerate(assembly_labels):
        ax = fig.add_subplot(gs[0, i])
        ax.pie(probs[assembly_label], labels=["%.2f%%" % (prob * 100) for prob in probs[assembly_label]],
               colors=[BLUE, RED], normalize=True)
        ax.set_title("assembly %i" % assembly_label)
        ax2 = fig.add_subplot(gs[1, i])
        i_pot = ax2.imshow(pot_matrices[assembly_label], cmap=pot_cmap, aspect="auto", vmin=-pot_extr, vmax=pot_extr)
        ax3 = fig.add_subplot(gs[2, i])
        i_dep = ax3.imshow(dep_matrices[assembly_label], cmap=dep_cmap, aspect="auto", vmin=-dep_extr, vmax=dep_extr)
        if i == 0:
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["assembly", "non-assembly"])
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["assembly", "non-assembly"])
        else:
            ax2.set_yticks([])
            ax3.set_yticks([])
        ax2.set_xticks([])
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["clustered", "not clustered"], rotation=45)
    fig.colorbar(i_pot, cax=fig.add_subplot(gs[1, i+1]), label="P(pot|cond) - P(pot) /\n P(pot|cond) + P(pot)")
    fig.colorbar(i_dep, cax=fig.add_subplot(gs[2, i+1]), label="P(dep|cond) - P(pot) /\n P(dep|cond) + P(pot)")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)




