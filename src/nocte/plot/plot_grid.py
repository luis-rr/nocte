"""
Code to plot grids of axes to explore multi-dimensional data.
Similar to pd.scatter_matrix but broken into pieces for adjustment

Separates layout (Cell/Grid), labeling, and rendering logic.
Each cell explicitly represents its semantic meaning and edge position.
"""

import matplotlib.cm
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from nocte.plot import plot as splot


def _edges_from_centers(values):
    """Return bin edges from ordered bin centers."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError('values must be one-dimensional')

    n = len(values)

    if n == 0:
        raise ValueError('cannot infer edges from an empty axis')

    if n == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])

    mids = 0.5 * (values[:-1] + values[1:])

    first_edge = values[0] - (mids[0] - values[0])
    last_edge = values[-1] + (values[-1] - mids[-1])

    return np.concatenate([[first_edge], mids, [last_edge]])


def _axis_edges_and_ticks(index):
    """
    Infer pcolormesh edges and tick positions from a pandas Index.

    Numeric or datetime-like indices are interpreted as bin centers.
    Other indices are treated as categorical bins.
    """
    if pd.api.types.is_datetime64_any_dtype(index):
        centers = mdates.date2num(pd.to_datetime(index).to_pydatetime())
        edges = _edges_from_centers(centers)
        return edges, centers, index, 'datetime'

    if pd.api.types.is_numeric_dtype(index):
        centers = index.to_numpy(dtype=float)
        edges = _edges_from_centers(centers)
        return edges, centers, index, 'numeric'

    centers = np.arange(len(index), dtype=float)
    edges = np.arange(len(index) + 1, dtype=float) - 0.5
    return edges, centers, index, 'categorical'


def dataframe_heatmap(
    ax,
    df,
    *,
    cmap='viridis',
    shading='auto',
    colorbar=False,
    colorbar_kwargs=None,
    xtick_rotation=0,
    ytick_rotation=0,
    set_labels=True,
    **pcolormesh_kwargs,
):
    """
    Plot a pandas DataFrame as a heatmap on an existing matplotlib axis.

    Rows correspond to df.index.
    Columns correspond to df.columns.

    Numeric axes are interpreted as bin centers, and bin edges are inferred
    from midpoints, with symmetric extrapolation at the boundaries.

    Non-numeric axes are treated as categorical, evenly spaced bins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    df : pandas.DataFrame
        Data to plot.
    cmap : str or Colormap
        Colormap passed to pcolormesh.
    shading : str
        Shading passed to pcolormesh. Default is 'auto'.
    colorbar : bool
        Whether to add a colorbar.
    colorbar_kwargs : dict or None
        Keyword arguments passed to fig.colorbar.
    xtick_rotation, ytick_rotation : float
        Tick label rotation angles.
    set_labels : bool
        Whether to use df.index.name and df.columns.name as axis labels.
    **pcolormesh_kwargs
        Extra kwargs passed to ax.pcolormesh.

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        The heatmap artist.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    if df.empty:
        raise ValueError('df must not be empty')

    x_edges, x_centers, x_labels, x_kind = _axis_edges_and_ticks(df.columns)
    y_edges, y_centers, y_labels, y_kind = _axis_edges_and_ticks(df.index)

    z = df.to_numpy(dtype=float)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap=cmap,
        shading=shading,
        **pcolormesh_kwargs,
    )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)

    if x_kind == 'datetime':
        ax.xaxis_date()
        ax.figure.autofmt_xdate()
    else:
        ax.set_xticklabels(x_labels, rotation=xtick_rotation)

    if y_kind == 'datetime':
        ax.yaxis_date()
    else:
        ax.set_yticklabels(y_labels, rotation=ytick_rotation)

    if set_labels:
        if df.columns.name is not None:
            ax.set_xlabel(df.columns.name)
        if df.index.name is not None:
            ax.set_ylabel(df.index.name)

    if colorbar:
        colorbar_kwargs = {} if colorbar_kwargs is None else colorbar_kwargs
        ax.figure.colorbar(mesh, ax=ax, **colorbar_kwargs)

    return mesh


class Cell:
    """
    Represents a single cell in a grid.

    Attributes:
        ax: matplotlib Axes object
        variables: tuple of variable name(s) represented (1, 2, or 3 elements)
        x_edge: x-axis label/tick position ('top', 'bottom', or 'inner')
        y_edge: y-axis label/tick position ('left', 'right', or 'inner')
    """

    def __init__(self, ax, variables: tuple, x_edge: str, y_edge: str):
        self.ax = ax
        self.variables = tuple(variables)
        self.x_edge = x_edge
        self.y_edge = y_edge

    @property
    def is_1d(self):
        """True if cell represents a single variable (diagonal)."""
        return self.dim == 1

    @property
    def is_2d(self):
        """True if cell represents two variables."""
        return self.dim == 2

    @property
    def is_3d(self):
        """True if cell represents three variables."""
        return self.dim == 3

    @property
    def dim(self):
        """Number of variables represented."""
        return len(self.variables)

    def set_spine_visible(self, edge, inner):
        """Set spine visibility, with separate controls for edge and inner cells."""
        ax = self.ax

        # x-axis spines
        if self.x_edge == 'inner':
            ax.spines['bottom'].set_visible(inner)

        elif self.x_edge == 'bottom':
            ax.spines['bottom'].set_visible(edge)

        elif self.x_edge == 'top':
            ax.spines['bottom'].set_visible(inner)

        # y-axis spines
        if self.y_edge == 'inner':
            ax.spines['left'].set_visible(inner)

        elif self.y_edge == 'left':
            ax.spines['left'].set_visible(edge)

        elif self.y_edge == 'right':
            ax.spines['left'].set_visible(inner)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def set_ticks_visible(self, edge, inner):
        """Set tick visibility, with separate controls for edge and inner cells."""
        ax = self.ax

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # x-axis ticks
        if self.x_edge == 'inner':
            ax.tick_params(
                axis='x',
                which='both',
                labelbottom=inner,
                bottom=inner,
                labeltop=inner,
                top=inner,
            )
        elif self.x_edge == 'bottom':
            ax.tick_params(
                axis='x',
                which='both',
                labelbottom=edge,
                bottom=edge,
                labeltop=inner,
                top=inner,
            )

        elif self.x_edge == 'top':
            ax.tick_params(
                axis='x',
                which='both',
                labelbottom=inner,
                bottom=inner,
                labeltop=edge,
                top=edge,
            )

        # y-axis ticks
        if self.y_edge == 'inner':
            ax.tick_params(
                axis='y',
                which='both',
                labelleft=inner,
                left=inner,
                labelright=inner,
                right=inner,
            )

        elif self.y_edge == 'left':
            ax.tick_params(
                axis='y',
                which='both',
                labelleft=edge,
                left=edge,
                labelright=inner,
                right=inner,
            )
        elif self.y_edge == 'right':
            ax.tick_params(
                axis='y',
                which='both',
                labelleft=inner,
                left=inner,
                labelright=edge,
                right=edge,
            )

    def set_axis_labels(self, label_aliases, edge, inner):
        """Set axis label text visibility, with separate controls for edge and inner cells."""
        label_aliases = label_aliases or {}
        ax = self.ax

        show_x = edge if self.x_edge != 'inner' else inner
        show_y = edge if self.y_edge != 'inner' else inner

        if len(self.variables) >= 1 and show_x:
            x_var = self.variables[0]  # first variable is x
            ax.set_xlabel(label_aliases.get(x_var, x_var))
        else:
            ax.set_xlabel('')

        if len(self.variables) >= 1 and show_y:
            y_var = self.variables[-1]  # last variable is y
            ax.set_ylabel(label_aliases.get(y_var, y_var))
        else:
            ax.set_ylabel('')

        if self.x_edge == 'bottom':
            ax.xaxis.set_label_position('bottom')
        elif self.x_edge == 'top':
            ax.xaxis.set_label_position('top')

        if self.y_edge == 'left':
            ax.yaxis.set_label_position('left')
        elif self.y_edge == 'right':
            ax.yaxis.set_label_position('right')


class Grid:
    """
    Container for grid cells with minimal API.

    Attributes:
        fig: matplotlib figure
        cells: list of Cell objects
        nrows, ncols: grid dimensions
    """

    def __init__(self, fig, cells, nrows, ncols):
        self.fig = fig
        self.cells = cells
        self.nrows = nrows
        self.ncols = ncols

    def iter_cells(self, dim=None):
        """
        Iterate over cells, optionally filtered by dimensionality.

        Args:
            dim: Filter by dimension (1, 2, or 3). If None, iterate all.
        """
        for cell in self.cells:
            if dim is None or cell.dim == dim:
                yield cell

    def iter_1d(self):
        """Iterate over 1D cells (diagonal)."""
        return self.iter_cells(dim=1)

    def iter_2d(self):
        """Iterate over 2D cells (off-diagonal)."""
        return self.iter_cells(dim=2)

    def iter_3d(self):
        """Iterate over 3D cells."""
        return self.iter_cells(dim=3)

    @staticmethod
    def _keep_triangle(i, j, n, triangle):
        """Determine if cell (i,j) should be visible based on triangle."""

        if triangle == 'bottom left':
            return i >= j

        if triangle == 'top right':
            return i <= j

        if triangle == 'bottom right':
            return i + j >= n - 1

        if triangle == 'top left':
            return i + j <= n - 1

        raise ValueError(f'Unknown triangle: {triangle}')

    @staticmethod
    def _compute_edge_position(i, j, n, xlabel_side, ylabel_side):
        """Compute where axis labels appear for cell (i,j)."""
        x_edge = 'inner'
        y_edge = 'inner'

        # x-axis at top or bottom
        if xlabel_side == 'top' and i == 0:
            x_edge = 'top'
        elif xlabel_side == 'bottom' and i == n - 1:
            x_edge = 'bottom'

        # y-axis at left or right
        if ylabel_side == 'left' and j == 0:
            y_edge = 'left'

        elif ylabel_side == 'right' and j == n - 1:
            y_edge = 'right'

        return x_edge, y_edge

    @staticmethod
    def _get_default_sides(triangle):
        """Compute default label sides from triangle name."""
        top_bottom, left_right = triangle.split(' ')
        return top_bottom, left_right

    @staticmethod
    def _normalize_sides(triangle, xlabel_side, ylabel_side):
        """Fill in default sides if not specified."""
        top_bottom, left_right = Grid._get_default_sides(triangle)
        return xlabel_side or top_bottom, ylabel_side or left_right

    @staticmethod
    def _share_yaxes(cells, exclude_3d=True, exclude_1d=True):
        """Share y within rows"""
        for _, row in cells.items():
            if exclude_3d:
                row = [cell for cell in row if not cell.is_3d]

            if exclude_1d:
                row = [cell for cell in row if not cell.is_1d]

            if len(row) > 1:
                for cell in row[1:]:
                    cell.ax.sharey(row[0].ax)

    @staticmethod
    def _share_xaxes(cells, exclude_3d=True, exclude_1d=False):
        """Share x within columns"""
        for _, col in cells.items():
            if exclude_3d:
                col = [cell for cell in col if not cell.is_3d]

            if exclude_1d:
                col = [cell for cell in col if not cell.is_1d]

            if len(col) > 1:
                for cell in col[1:]:
                    cell.ax.sharex(col[0].ax)

    @classmethod
    def make_grid(
        cls,
        labels: list[str],
        *,
        triangle: str = 'bottom left',
        show_marginals: bool = True,
        xlabel_side: str | None = None,
        ylabel_side: str | None = None,
        figsize: tuple[float, float] | None = None,
        label_aliases: dict | None = None,
    ):
        """
        Create a grid of cells for pairwise exploration.

        Args:
            labels: variable names
            triangle: which cells to show ('bottom left', 'top right', 'bottom right', 'top left')
            show_marginals: include marginal 1D cells (default: True). If False,
                build a reduced (n-1)x(n-1) grid with only 2D pairwise cells.
            xlabel_side: x-axis label position (defaults based on triangle)
            ylabel_side: y-axis label position (defaults based on triangle)
            figsize: figure size (default: 0.75*n x 0.75*n)
            label_aliases: dict mapping label -> display name

        Returns:
            Grid object with 2D pairwise Cell objects
        """
        n = len(labels)
        grid_n = n if show_marginals else n - 1

        if grid_n <= 0:
            raise ValueError(
                'At least 2 labels are required when show_marginals=False.'
            )

        # Create figure
        fig, axes_grid = plt.subplots(
            grid_n,
            grid_n,
            figsize=figsize or (0.75 * grid_n, 0.75 * grid_n),
            sharex=False,
            sharey=False,
        )

        # Handle 1D case
        if grid_n == 1:
            axes_grid = np.array([[axes_grid]])

        # Compute defaults
        xlabel_side, ylabel_side = cls._normalize_sides(
            triangle, xlabel_side, ylabel_side
        )

        # Build cells with standard pairwise semantics
        cells = []
        cells_by_col = {j: [] for j in range(grid_n)}
        cells_by_row = {i: [] for i in range(grid_n)}

        for i in range(grid_n):
            for j in range(grid_n):
                ax = axes_grid[i, j]

                if not cls._keep_triangle(i, j, grid_n, triangle):
                    ax.set_visible(False)
                    continue

                # Standard semantics:
                # include_diagonal_1d=True  -> diagonal = 1D, off-diagonal = 2D
                # include_diagonal_1d=False -> all visible cells are 2D using
                #                             x from labels[:-1], y from labels[1:]
                if show_marginals and i == j:
                    variables = (labels[i],)
                else:
                    if show_marginals:
                        variables = (labels[j], labels[i])  # (x, y)
                    else:
                        variables = (labels[j], labels[i + 1])  # (x, y)

                # Compute edge positions
                x_edge, y_edge = cls._compute_edge_position(
                    i, j, grid_n, xlabel_side, ylabel_side
                )

                cell = Cell(ax, variables, x_edge=x_edge, y_edge=y_edge)
                cells.append(cell)

                cells_by_col[j].append(cell)
                cells_by_row[i].append(cell)

        # Share axes
        cls._share_xaxes(cells_by_col)
        cls._share_yaxes(cells_by_row)

        grid = cls(fig, cells, grid_n, grid_n)

        # Apply default spine, tick, and axis label settings.
        grid.set_ticks_visible()
        grid.set_spine_visible()
        grid.set_axis_labels(label_aliases)

        return grid

    @classmethod
    def make_grid_3d(
        cls,
        labels: list[str],
        *,
        triangle: str = 'bottom left',
        xlabel_side: str | None = None,
        ylabel_side: str | None = None,
        figsize: tuple[float, float] | None = None,
        label_aliases: dict | None = None,
        var_triples: dict | None = None,
    ):
        """
        Create a grid with 3D projections on diagonal cells.

        Diagonal cells show 3D scatter/line plots of consecutive variable triplets.
        Off-diagonal cells remain 2D pairwise plots.

        Args:
            labels: variable names (ideally >= 3 for 3D visualizations)
            triangle: which cells to show
            xlabel_side, ylabel_side: label positions (default: based on triangle)
            figsize: figure size (default: 0.75*n x 0.75*n)
            label_aliases: dict mapping label -> display name
            var_triples: dict mapping label -> (x, y, z) tuple for 3D cells.
                        If not provided, uses consecutive dimensions:
                        - labels[i] -> (labels[i], labels[i+1], labels[i+2])
                        - Wraps around at end

        Returns:
            Grid object with 3D Cell objects on diagonal, 2D off-diagonal
        """
        n = len(labels)

        # Create figure with GridSpec for flexible axes
        fig = plt.figure(figsize=figsize or (0.75 * n, 0.75 * n))
        gs = GridSpec(n, n, figure=fig)

        # Compute defaults
        xlabel_side, ylabel_side = cls._normalize_sides(
            triangle, xlabel_side, ylabel_side
        )

        # Build default var_triples: consecutive dimensions
        if var_triples is None:
            var_triples = {}
            for i in range(n):
                # Use 3 consecutive labels, wrapping around
                indices = [(i + k) % n for k in range(3)]
                var_triples[labels[i]] = tuple(labels[idx] for idx in indices)

        # Build cells
        cells = []
        cells_by_col = {j: [] for j in range(n)}
        cells_by_row = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(n):
                if not cls._keep_triangle(i, j, n, triangle):
                    continue

                # Create appropriate axes type
                if i == j:
                    # Diagonal: 3D axes (or 1D if var_triples says so)
                    variables = var_triples.get(labels[i], (labels[i],))
                    if len(variables) == 3:
                        ax = fig.add_subplot(gs[i, j], projection='3d')
                    else:
                        ax = fig.add_subplot(gs[i, j])
                else:
                    # Off-diagonal: standard 2D axes
                    ax = fig.add_subplot(gs[i, j])
                    variables = (labels[j], labels[i])  # (x, y)

                # Compute edge positions
                x_edge, y_edge = cls._compute_edge_position(
                    i, j, n, xlabel_side, ylabel_side
                )

                cell = Cell(ax, variables, x_edge=x_edge, y_edge=y_edge)
                cells.append(cell)

                cells_by_col[j].append(cell)
                cells_by_row[i].append(cell)

        # Share axes (exclude 3D axes from sharing)
        cls._share_xaxes(cells_by_col)
        cls._share_yaxes(cells_by_row)

        grid = cls(fig, cells, n, n)

        # Apply default spine, tick, and axis label settings.
        grid.set_ticks_visible()
        grid.set_spine_visible()
        grid.set_axis_labels(label_aliases)

        return grid

    def set_ticks_visible(self, edge=True, inner=False):
        """Set tick visibility, with separate controls for edge and inner cells."""
        for cell in self.cells:
            cell.set_ticks_visible(edge=edge, inner=inner)

    def set_spine_visible(self, edge=True, inner=True):
        """Set spine visibility, with separate controls for edge and inner cells."""
        for cell in self.cells:
            cell.set_spine_visible(edge=edge, inner=inner)

    def set_axis_labels(self, label_aliases=None, edge=True, inner=False):
        """Set axis labels, with separate controls for edge and inner cells."""
        for cell in self.cells:
            cell.set_axis_labels(label_aliases=label_aliases, edge=edge, inner=inner)

    def plot_2d_scatter(self, df, /, styles=None, **kwargs):
        """
        Scatter plot data in 2D cells of this grid.

        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)

        Default style: alpha=0.3, edgecolor='w', linewidth=0.3, facecolor='k', marker='.', s=20
        """
        defaults = dict(
            alpha=0.3, edgecolor='w', linewidth=0.3, facecolor='k', marker='.', s=20
        )
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], df[y], **full_kwargs)

    def plot_2d_scatter_cmap(self, df, c, /, styles=None, **kwargs):
        """
        Scatter plot with color mapping in 2D cells of this grid.

        Args:
            df: DataFrame with variables
            c: color variable (string column name or array)
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)

        Default style: alpha=0.3, edgecolor='none', linewidth=0.3, cmap='viridis', marker='.', s=20
        """
        defaults = dict(
            alpha=0.3, edgecolor='none', linewidth=0.3, cmap='viridis', marker='.', s=20
        )
        styles = styles or {}

        if isinstance(c, str):
            c = df[c]

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], df[y], c=c, **full_kwargs)

    def plot_2d_line(self, df, /, styles=None, **kwargs):
        """
        Line plot in 2D cells of this grid.

        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional plot kwargs (override defaults)

        Default style: alpha=0.3, linewidth=0.3, color='k'
        """
        defaults = dict(alpha=0.3, linewidth=0.3, color='k')
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.plot(df[x], df[y], **full_kwargs)

    def plot_2d_heatmap(self, df, /, which, func='mean', styles=None, **kwargs):
        """ """
        if isinstance(func, str):
            func = getattr(pd.api.typing.DataFrameGroupBy, func)

        all_heatmaps = {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables

            heatmap = func(df.groupby([x, y])[which])
            heatmap = heatmap.unstack(x)
            heatmap.sort_index(inplace=True, axis=0)
            heatmap.sort_index(inplace=True, axis=1)

            all_heatmaps[x, y] = heatmap

        vmin = min(np.nanmin(heatmap.values) for heatmap in all_heatmaps.values())
        vmax = max(np.nanmax(heatmap.values) for heatmap in all_heatmaps.values())

        defaults = dict(
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        )
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables

            heatmap = all_heatmaps[x, y]

            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}

            dataframe_heatmap(
                cell.ax,
                heatmap,
                **full_kwargs,
            )

    def plot_1d_scatter(self, df, /, which, styles=None, **kwargs):
        """
        Scatter plot in 1D cells (diagonal) of this grid.

        Args:
            df: DataFrame with variables
            which: column name to plot
            styles: dict mapping var_name -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)

        Default style: facecolor='k'
        """
        styles = styles or {}
        defaults = dict(facecolor='k')

        if isinstance(which, str):
            which = df[which]

        for cell in self.iter_cells(dim=1):
            x = cell.variables[0]
            style = styles.get(x, {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], which, **full_kwargs)

    def plot_1d_hist(self, df, /, styles=None, **kwargs):
        """
        Histogram in 1D cells (diagonal) of this grid.

        Args:
            df: DataFrame with variables
            styles: dict mapping var_name -> style kwargs (default: None)
            **kwargs: additional hist kwargs (override defaults)

        Default style: facecolor='k'
        """
        styles = styles or {}
        defaults = dict(facecolor='k')

        for cell in self.iter_cells(dim=1):
            x = cell.variables[0]
            style = styles.get(x, {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.hist(df[x], **full_kwargs)

    def plot_2d_line_segmented(self, df, /, styles=None, **kwargs):
        """
        Segmented line plot in 2D cells of this grid.

        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional plot kwargs (override defaults)

        Default style: alpha=0.3, linewidth=0.3, color='k'
        """
        defaults = dict(alpha=0.3, linewidth=0.3, color='k')
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}

            splot.plot_segmented_line(
                cell.ax,
                df[x].values,
                df[y].values,
                **full_kwargs,
            )

    def plot_2d_line_segmented_cmap(self, df, /, c, styles=None, **kwargs):
        """
        Segmented line plot in 2D cells of this grid.

        Args:
            df: DataFrame with variables
            c: column name for color mapping or array of color values
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional plot kwargs (override defaults)

        Default style: alpha=0.3, linewidth=0.3, color='k'
        """
        defaults = dict(alpha=0.3, linewidth=0.3)
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}

            splot.plot_segmented_line_cmap(
                cell.ax,
                df[x].values,
                df[y].values,
                c=df[c] if isinstance(c, str) else c,
                **full_kwargs,
            )

    def plot_2d_line_segmented_highlighted(self, df, /, wins, styles=None, **kwargs):
        """
        Segmented line plot in 2D cells of this grid.

        Args:
            df: DataFrame with variables
            wins: DataFrame with window definitions
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional plot kwargs (override defaults)

        Default style: alpha=0.3, linewidth=0.3, color='k'
        """
        defaults = dict(alpha=0.3, linewidth=0.3)
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}

            splot.plot_segmented_line_highlighted(
                cell.ax,
                df[x],
                df[y],
                wins=wins,
                styles=styles,
                **full_kwargs,
            )

    def plot_3d_scatter(self, df, /, styles=None, **kwargs):
        """
        3D scatter plot in 3D cells of this grid.

        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var, z_var) -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)

        Default style: alpha=0.3, edgecolor='w', linewidth=0.3, c='k', marker='.', s=20
        """
        defaults = dict(
            alpha=0.3, edgecolor='w', linewidth=0.3, c='k', marker='.', s=20
        )
        styles = styles or {}

        for cell in self.iter_cells(dim=3):
            x, y, z = cell.variables
            style = styles.get((x, y, z), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], df[y], df[z], **full_kwargs)

    def plot_3d_line_segmented(self, df, /, styles=None, **kwargs):
        """
        Segmented 3D line plot in 3D cells of this grid.

        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var, z_var) -> style kwargs (default: None)
            **kwargs: additional plot kwargs (override defaults)

        Default style: alpha=0.3, linewidth=0.3, color='k'
        """
        defaults = dict(alpha=0.3, linewidth=0.3, color='k')
        styles = styles or {}

        for cell in self.iter_cells(dim=3):
            x, y, z = cell.variables
            style = styles.get((x, y, z), {})
            full_kwargs = {**defaults, **style, **kwargs}

            # Simple line plot for 3D
            cell.ax.plot(df[x], df[y], df[z], **full_kwargs)

    def add_colorbar(
        self,
        cmap,
        norm,
        location='top right',
        orientation='horizontal',
        label=None,
        size=None,
        pad=0.04,
        margin=0.04,
        **colorbar_kwargs,
    ):
        """
        Add a standalone colorbar to a figure.

        Parameters
        ----------
        fig:
            Matplotlib figure.
        cmap:
            Colormap.
        norm:
            Matplotlib norm.
        location:
            One of:
            'top right', 'top left', 'bottom right', 'bottom left',
            'right top', 'right bottom', 'left top', 'left bottom'.
        orientation:
            'horizontal' or 'vertical'.
        label:
            Optional colorbar label.
        size:
            Optional tuple specifying the colorbar axes size as
            (width, height) in figure coordinates.
        pad:
            Padding from the figure edge, in figure coordinates.
        margin:
            Additional margin from the nearest corner, in figure coordinates.
        **colorbar_kwargs:
            Passed to fig.colorbar.

        Returns
        -------
        cbar:
            The created colorbar.
        """

        if orientation not in {'horizontal', 'vertical'}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        location = location.lower().strip()

        if size is None:
            if orientation == 'horizontal':
                width, height = 0.22, 0.025
            else:
                width, height = 0.025, 0.22
        else:
            width, height = size

        horizontal_locations = {
            'top right': (1 - margin - width, 1 - pad - height),
            'top left': (margin, 1 - pad - height),
            'bottom right': (1 - margin - width, pad),
            'bottom left': (margin, pad),
        }

        vertical_locations = {
            'right top': (1 - pad - width, 1 - margin - height),
            'right bottom': (1 - pad - width, margin),
            'left top': (pad, 1 - margin - height),
            'left bottom': (pad, margin),
        }

        if orientation == 'horizontal':
            valid_locations = horizontal_locations
        else:
            valid_locations = vertical_locations

        if location not in valid_locations:
            options = "', '".join(valid_locations)
            raise ValueError(f"location must be one of: '{options}'")

        left, bottom = valid_locations[location]

        cax = self.fig.add_axes([left, bottom, width, height])

        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = self.fig.colorbar(
            sm,
            cax=cax,
            orientation=orientation,
            **colorbar_kwargs,
        )

        if label is not None:
            cbar.set_label(label)

        return cbar
