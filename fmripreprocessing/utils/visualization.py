import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import plotting
from pycirclize import Circos


def plot_glasser_cor_mat(
    correlation_mat,
    labels,
    vmin=None,
    vmax=None,
    symmetric=True,
    title="Scheaffer",
):
    np.fill(correlation_mat, 0)
    if symmetric:
        v = np.max(np.abs(correlation_mat))
        vmin, vmax = -v, v
    yeo_networks = np.hstack(
        (
            np.repeat("visual", 31),
            np.repeat("somatomotor", 37),
            np.repeat("dorsal attention", 23),
            np.repeat("ventral attention", 22),
            np.repeat("limbic", 13),
            np.repeat("frontoparietal", 22),
            np.repeat("default", 52),
            np.repeat("visual", 31),
            np.repeat("somatomotor", 37),
            np.repeat("dorsal attention", 23),
            np.repeat("ventral attention", 22),
            np.repeat("limbic", 13),
            np.repeat("frontoparietal", 22),
            np.repeat("default", 52),
        )
    )

    network_labels = pd.Series(yeo_networks)[labels]
    network_labels.index = np.arange(len(network_labels))
    network_labels.name = "networks"
    sns.set_palette(palette="Paired", n_colors=network_labels.unique().size)
    network_lut = dict(
        zip(
            map(str, network_labels.unique()),
            sns.color_palette(n_colors=network_labels.unique().size),
        )
    )
    network_colors = pd.Series(network_labels).map(network_lut)
    np.fill_diagonal(correlation_mat, 0)

    g = sns.clustermap(
        pd.DataFrame(correlation_mat),
        cmap="RdBu_r",
        # Turn off the clustering
        row_cluster=False,
        col_cluster=False,
        # Add colored class labels
        row_colors=network_colors,
        col_colors=network_colors,
        # Make the plot look better when many rows/cols
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
    )

    _, idx = np.unique(network_labels, return_index=True)
    legend = network_labels[np.sort(idx)]
    for label in network_labels.unique():
        g.ax_col_dendrogram.bar(
            -1, 0, color=network_lut[label], label=label, linewidth=0
        )

    g.ax_col_dendrogram.legend(loc="center", ncol=4)
    g.cax.set_position([1, 0.2, 0.03, 0.5])
    g.fig.suptitle(title)

    return g


def plot_HMAT_cor_mat(
    correlation_mat, vmin=None, vmax=None, symmetric=True, title="HMAT"
):
    np.fill_diagonal(correlation_mat, 0)

    if symmetric:
        v = np.max(np.abs(correlation_mat))
        vmin, vmax = -v, v
    HMAT_key = np.array(
        [
            "Right_M1",
            "Left_M1",
            "Right_S1",
            "Left_S1",
            "Right_SMA",
            "Left_SMA",
            "Right_preSMA",
            "Left_preSMA",
            "Right_PMd",
            "Left_PMd",
            "Right_PMv",
            "Left_PMv",
        ]
    )
    sort_index = np.argsort(HMAT_key)

    labels = np.hstack(
        (
            np.repeat("M1", 1),
            np.repeat("PMC", 2),
            np.repeat("S1", 1),
            np.repeat("SMA", 2),
            np.repeat("M1", 1),
            np.repeat("PMC", 2),
            np.repeat("S1", 1),
            np.repeat("SMA", 2),
        )
    )
    network_labels = pd.Series(labels)
    network_labels.index = np.arange(len(network_labels))
    network_labels.name = ""
    sns.set_palette(palette="Paired", n_colors=network_labels.unique().size)
    network_lut = dict(
        zip(
            map(str, network_labels.unique()),
            sns.color_palette(n_colors=network_labels.unique().size),
        )
    )
    network_colors = pd.Series(network_labels).map(network_lut)
    np.fill_diagonal(correlation_mat, 0)
    g = sns.clustermap(
        pd.DataFrame(correlation_mat[:, sort_index][sort_index]),
        cmap="RdBu_r",
        # Turn off the clustering
        row_cluster=False,
        col_cluster=False,
        # Add colored class labels
        row_colors=network_colors,
        col_colors=network_colors,
        # Make the plot look better when many rows/cols
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
    )

    _, idx = np.unique(network_labels, return_index=True)
    legend = network_labels[np.sort(idx)]
    for label in network_labels.unique():
        g.ax_col_dendrogram.bar(
            -1, 0, color=network_lut[label], label=label, linewidth=0
        )

    g.ax_col_dendrogram.legend(loc="center", ncol=4)
    g.cax.set_position([1, 0.2, 0.03, 0.5])
    idx2 = list(idx) + [6, 7, 9, 10]
    g.ax_row_colors.set_yticks(
        0.5
        * (np.array(idx2) + np.array(list(idx2[1:]) + [len(network_labels)]))
    )
    g.ax_row_colors.set_yticklabels(np.tile(legend, 2))
    g.ax_row_colors.yaxis.set_tick_params(size=0)  # make tick marks invisible
    g.ax_col_colors.set_xticks(
        0.5
        * (np.array(idx2) + np.array(list(idx2[1:]) + [len(network_labels)]))
    )
    g.ax_col_colors.set_xticklabels(np.tile(legend, 2))
    g.ax_col_colors.xaxis.set_tick_params(size=0)  # make tick marks invisible
    g.ax_col_colors.xaxis.tick_top()
    g.fig.suptitle(title)
    return g


def plot_difumo_cor_mat(
    correlation_mat, labels, vmin=-0.3, vmax=0.6, title="Difumo"
):
    yeo_networks = []

    network_labels = pd.Series(yeo_networks)[labels]
    network_labels.index = np.arange(len(network_labels))
    network_labels.name = "networks"
    sns.set_palette(palette="Paired", n_colors=network_labels.unique().size)
    network_lut = dict(
        zip(
            map(str, network_labels.unique()),
            sns.color_palette(n_colors=network_labels.unique().size),
        )
    )
    network_colors = pd.Series(network_labels).map(network_lut)
    np.fill_diagonal(correlation_mat, 0)

    g = sns.clustermap(
        pd.DataFrame(correlation_mat),
        cmap="RdBu_r",
        # Turn off the clustering
        row_cluster=False,
        col_cluster=False,
        # Add colored class labels
        row_colors=network_colors,
        col_colors=network_colors,
        # Make the plot look better when many rows/cols
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
    )

    _, idx = np.unique(network_labels, return_index=True)
    legend = network_labels[np.sort(idx)]
    for label in network_labels.unique():
        g.ax_col_dendrogram.bar(
            -1, 0, color=network_lut[label], label=label, linewidth=0
        )

    g.ax_col_dendrogram.legend(loc="center", ncol=4)
    g.cax.set_position([1, 0.2, 0.03, 0.5])
    g.fig.suptitle(title)

    return g


def get_labels_HMAT():
    HMAT_key = np.array(
        [
            "Right_M1",
            "Left_M1",
            "Right_S1",
            "Left_S1",
            "Right_SMA",
            "Left_SMA",
            "Right_preSMA",
            "Left_preSMA",
            "Right_PMd",
            "Left_PMd",
            "Right_PMv",
            "Left_PMv",
        ]
    )
    sort_index = np.argsort(HMAT_key)

    labels = np.hstack(
        (
            np.repeat("M1", 1),
            np.repeat("PMC", 2),
            np.repeat("S1", 1),
            np.repeat("SMA", 2),
            np.repeat("M1", 1),
            np.repeat("PMC", 2),
            np.repeat("S1", 1),
            np.repeat("SMA", 2),
        )
    )
    network_labels = pd.Series(labels)
    network_labels.index = np.arange(len(network_labels))
    network_labels.name = ""
    sns.set_palette(palette="Paired", n_colors=network_labels.unique().size)
    network_lut = dict(
        zip(
            map(str, network_labels.unique()),
            sns.color_palette(n_colors=network_labels.unique().size),
        )
    )
    return network_labels, network_lut, sort_index


def get_labels_glasser(labels):
    yeo_networks = np.hstack(
        (
            np.repeat("visual", 31),
            np.repeat("somatomotor", 37),
            np.repeat("dorsal attention", 23),
            np.repeat("ventral attention", 22),
            np.repeat("limbic", 13),
            np.repeat("frontoparietal", 22),
            np.repeat("default", 52),
            np.repeat("visual", 31),
            np.repeat("somatomotor", 37),
            np.repeat("dorsal attention", 23),
            np.repeat("ventral attention", 22),
            np.repeat("limbic", 13),
            np.repeat("frontoparietal", 22),
            np.repeat("default", 52),
        )
    )

    network_labels = pd.Series(yeo_networks)[labels]
    network_labels.index = np.arange(len(network_labels))
    network_labels.name = "networks"
    sns.set_palette(palette="Paired", n_colors=network_labels.unique().size)
    network_lut = dict(
        zip(
            map(str, network_labels.unique()),
            sns.color_palette(n_colors=network_labels.unique().size),
        )
    )
    return network_labels, network_lut


def _get_cmap(colormap, lut=None):
    from matplotlib import colors, rcParams

    try:
        from matplotlib import colormaps
    except Exception:
        from matplotlib.cm import get_cmap
    else:

        def get_cmap(cmap):
            return colormaps[cmap]

    if colormap is None:
        colormap = rcParams["image.cmap"]
    if not isinstance(colormap, colors.Colormap):
        colormap = get_cmap(colormap)
    if lut is not None:
        # triage method for MPL 3.6 ('resampled') or older ('_resample')
        if hasattr(colormap, "resampled"):
            resampled = colormap.resampled
        else:
            resampled = colormap._resample
        colormap = resampled(lut)
    return colormap


def circle_plot(
    con,
    labels=None,
    con_thresh=0,
    atlas_name="glasser",
    vmin=None,
    vmax=None,
    colormap="RdBu_r",
    symmetric=True,
    colorbar=True,
    write_names=True,
    space=1,
):
    if atlas_name == "glasser":
        net_labels, net_lut = get_labels_glasser(labels)
    elif atlas_name == "HMAT":
        net_labels, net_lut, sort_index = get_labels_HMAT()
        con = con[:, sort_index][sort_index]

    def find_pivot(labels):
        test = False
        for i, value in enumerate(labels):
            if value != labels.iloc[0]:
                test = True
            if test and value == labels.iloc[0]:
                index = i
                break
        return index

    # Build sectors for Right hemisphere
    rh_index = find_pivot(net_labels)
    sectors_sizes = net_labels.iloc[rh_index:].value_counts()
    keys = pd.unique(net_labels[rh_index:])
    sectors = {"R-" + key: sectors_sizes[key] for key in keys}
    name2color = {"R-" + key: color for key, color in net_lut.items()}

    # Update with Left hemisphere
    sectors_sizes = net_labels.iloc[:rh_index][::-1].value_counts()
    keys = pd.unique(net_labels[:rh_index][::-1])
    sectors.update({"L-" + key: sectors_sizes[key] for key in keys})
    name2color.update({"L-" + key: color for key, color in net_lut.items()})

    circle_plot = Circos(sectors, space=space)
    for sector in circle_plot.sectors:
        track = sector.add_track((95, 100))
        track.axis(fc=name2color[sector.name])
        if not write_names:
            track.text("")
        else:
            track.text(sector.name, color="white", size=11)

    indices = np.tril_indices(con.shape[0], -1)
    con = con[indices]

    colormap = _get_cmap(colormap)

    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    del con_abs
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if not symmetric:
        if vmin is None:
            vmin = np.min(con[np.abs(con) >= con_thresh])
        if vmax is None:
            vmax = np.max(con)
    else:
        v = np.max(np.abs(con))
        vmin, vmax = -v, v

    vrange = vmax - vmin

    con_val_scaled = (con - vmin) / vrange
    networks_hemispheres = [
        "L-" + lab if i < rh_index else "R-" + lab
        for i, lab in enumerate(net_labels)
    ]
    network_starts = {networks_hemispheres[0]: 0}
    network0 = networks_hemispheres[0]
    for i, network in enumerate(networks_hemispheres):
        if network == network0:
            continue
        else:
            network_starts.update({network: i})
            network0 = network

    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        network_i, network_j = networks_hemispheres[i], networks_hemispheres[j]
        index_network_i, index_network_j = (
            i - network_starts[network_i],
            j - network_starts[network_j],
        )
        color = colormap(con_val_scaled[pos])

        circle_plot.link(
            (network_i, index_network_i, index_network_i + 1),
            (network_j, index_network_j, index_network_j + 1),
            color=color,
            allow_twist=False,
        )

    fig = circle_plot.plotfig()
    if colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin, vmax)
        )
        sm.set_array(np.linspace(vmin, vmax))
        colorbar_kwargs = dict()
        colorbar_kwargs.update(shrink=0.6)
        cb = fig.colorbar(sm, **colorbar_kwargs)
        cb_yticks = plt.getp(cb.ax.axes, "yticklabels")
        cb.ax.tick_params()
        plt.setp(cb_yticks)
    return fig
