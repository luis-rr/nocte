"""
Code to plot grids of axes to explore multi-dimensional data.
Similar to pd.scatter_matrix but broken into pieces for adjustment
"""

import numpy as np

import matplotlib.pyplot as plt

from nocte import plot as splot


def make_grid(
        labels: list[str],
        *,
        triangle: str = 'bottom left',
        xlabel_side: str = None,
        ylabel_side: str = None,
        figsize: tuple[float, float] | None = None,
        label_aliases=None,
):
    n = len(labels)

    fig, axes_grid = plt.subplots(
        n, n,
        figsize=figsize or (0.75 * n, 0.75 * n),
        sharex=False,
        sharey=False,
    )

    def keep(_i, _j):
        if triangle == 'bottom left':
            return _i >= _j
        if triangle == 'top right':
            return _i <= _j
        if triangle == 'bottom right':
            return _i + _j >= n - 1
        if triangle == 'top left':
            return _i + _j <= n - 1
        raise ValueError(f'Unknown triangle: {triangle}')

    ax_pairs = {}
    ax_margs = {}

    xlabels = labels

    ylabels = labels if triangle in ['bottom left', 'top right'] else labels[::-1]

    for i, ylab in enumerate(xlabels):
        for j, xlab in enumerate(ylabels):
            ax = axes_grid[i, j]

            if not keep(i, j):
                ax.set_visible(False)
                continue

            if xlab != ylab:
                ax_pairs[(xlab, ylab)] = ax
            else:
                ax_margs[xlab] = ax

    top_bottom, left_right = triangle.split(' ')

    xlabel_side = xlabel_side or top_bottom
    ylabel_side = ylabel_side or left_right

    set_axis_sharing(
        axes_grid,
        exclude=list(ax_margs.values()),
    )

    set_axis_labels(
        axes_grid,
        xlabels=labels,
        ylabels=labels,
        xside=xlabel_side,
        yside=ylabel_side,
        aliases=label_aliases,
    )

    return fig, ax_pairs, ax_margs


def set_axis_labels(
        axes: np.ndarray,
        xlabels,
        ylabels,
        *,
        xside='bottom',
        yside='left',
        inner_spines=True,
        inner_ticks=False,
        inner_ticklabels=False,
        aliases=None
):
    # x axis: operate on columns (original grid)
    set_axis_label(
        axes,
        axis='x',
        labels=xlabels,
        side=xside,
        inner_spines=inner_spines,
        inner_ticks=inner_ticks,
        inner_ticklabels=inner_ticklabels,
        aliases=aliases,
    )

    # y axis: operate on rows -> transpose, but KEEP ylabels aligned to rows
    set_axis_label(
        axes.T,
        axis='y',
        labels=ylabels,
        side=yside,
        inner_spines=inner_spines,
        inner_ticks=inner_ticks,
        inner_ticklabels=inner_ticklabels,
        aliases=aliases,
    )


def set_axis_label(
        axes: np.ndarray,
        *,
        axis: str,  # 'x' or 'y'
        labels: list[str],
        side: str,
        inner_spines: bool = True,
        inner_ticks: bool = False,
        inner_ticklabels: bool = False,
        aliases=None,
):
    aliases = aliases or {}
    n_outer, n_inner = axes.shape

    other = {
        'bottom': 'top',
        'top': 'bottom',
        'left': 'right',
        'right': 'left',
    }[side]

    for i in range(n_outer):
        group = [
            axes.T[i, j]
            for j in range(n_inner)
            if axes.T[i, j].get_visible()
        ]
        if not group:
            continue

        target = (
            group[0]
            if side in ('top', 'left')
            else group[-1]
        )

        for ax in group:
            ax.tick_params(
                axis=axis,
                which='both',
                **{
                    side: inner_ticks,
                    other: False,
                    f'label{side}': inner_ticklabels,
                    f'label{other}': False,
                },
            )
            ax.spines[side].set_visible(inner_spines)
            ax.spines[other].set_visible(False)

        axis_obj = target.xaxis if axis == 'x' else target.yaxis
        axis_obj.set_ticks_position(side)
        axis_obj.set_label_position(side)
        target.spines[side].set_visible(True)

        label = labels[i]
        target.set(**{f'{axis}label': aliases.get(label, label)})


def set_axis_sharing(
        axes: np.ndarray,
        *,
        sharex: bool = True,
        sharey: bool = True,
        exclude=None
):
    exclude = exclude or []

    nrows, ncols = axes.shape

    # ---- share x within columns ----
    if sharex:
        for j in range(ncols):
            col_axes = [
                axes[i, j]
                for i in range(nrows)
                if axes[i, j] not in exclude and axes[i, j].get_visible()
            ]
            if len(col_axes) < 2:
                continue

            base = col_axes[0]
            for ax in col_axes[1:]:
                ax.sharex(base)

    # ---- share y within rows ----
    if sharey:
        for i in range(nrows):
            row_axes = [
                axes[i, j]
                for j in range(ncols)
                if axes[i, j] not in exclude and axes[i, j].get_visible()
            ]
            if len(row_axes) < 2:
                continue

            base = row_axes[0]
            for ax in row_axes[1:]:
                ax.sharey(base)


def plot_scatter_2d(axs_map, df, /, styles=None, **kwargs):
    defaults = dict(alpha=0.3, edgecolor='w', linewidth=0.3, facecolor='k', marker='.', s=20)
    styles = styles or {}

    for (x, y), ax in axs_map.items():
        style = styles.get((x, y), {})
        full_kwargs = {**defaults, **style, **kwargs}
        ax.scatter(df[x], df[y], **full_kwargs)


def plot_line_2d(axs_map, df, /, styles=None, **kwargs):
    defaults = dict(alpha=0.3, linewidth=0.3, color='k')
    styles = styles or {}

    for (x, y), ax in axs_map.items():
        style = styles.get((x, y), {})
        full_kwargs = {**defaults, **style, **kwargs}
        ax.plot(df[x], df[y], **full_kwargs)


def plot_hist_1d(axs_map, df, /, styles=None, **kwargs):
    styles = styles or {}
    defaults = dict(facecolor='k')

    for x, ax in axs_map.items():
        style = styles.get(x, {})
        full_kwargs = {**defaults, **style, **kwargs}
        ax.hist(df[x], **full_kwargs)


def plot_line_segmented_2d(axs_map, df, /, styles=None, **kwargs):

    defaults = dict(alpha=0.3, linewidth=0.3, color='k')
    styles = styles or {}

    for (x, y), ax in axs_map.items():
        style = styles.get((x, y), {})
        full_kwargs = {**defaults, **style, **kwargs}

        splot.plot_segmented_line(
            ax,
            df[x].values,
            df[y].values,
            **full_kwargs,
        )
