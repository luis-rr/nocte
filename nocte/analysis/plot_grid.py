"""
Code to plot grids of axes to explore multi-dimensional data.
Similar to pd.scatter_matrix but broken into pieces for adjustment

Separates layout (Cell/Grid), labeling, and rendering logic.
Each cell explicitly represents its semantic meaning and edge position.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from nocte import plot as splot


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
                axis='x', which='both',
                labelbottom=inner, bottom=inner, 
                labeltop=inner, top=inner,

            )
        elif self.x_edge == 'bottom':
            ax.tick_params(
                axis='x', which='both',
                labelbottom=edge, bottom=edge,
                labeltop=inner, top=inner,
            )

        elif self.x_edge == 'top':
            ax.tick_params(
                axis='x', which='both',
                labelbottom=inner, bottom=inner,
                labeltop=edge, top=edge,
            )

        # y-axis ticks
        if self.y_edge == 'inner':
            ax.tick_params(
                axis='y', which='both',
                labelleft=inner, left=inner,
                labelright=inner, right=inner,
            )
    
        elif self.y_edge == 'left':
            ax.tick_params(
                axis='y', which='both',
                labelleft=edge, left=edge,
                labelright=inner, right=inner,
            )
        elif self.y_edge == 'right':
            ax.tick_params(
                axis='y', which='both',
                labelleft=inner, left=inner,
                labelright=edge, right=edge,
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
    def _share_xaxes(cells,  exclude_3d=True, exclude_1d=False):
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
            xlabel_side: x-axis label position (defaults based on triangle)
            ylabel_side: y-axis label position (defaults based on triangle)
            figsize: figure size (default: 0.75*n x 0.75*n)
            label_aliases: dict mapping label -> display name
        
        Returns:
            Grid object with 2D pairwise Cell objects
        """
        n = len(labels)
        
        # Create figure
        fig, axes_grid = plt.subplots(
            n, n,
            figsize=figsize or (0.75 * n, 0.75 * n),
            sharex=False,
            sharey=False,
        )
        
        # Handle 1D case
        if n == 1:
            axes_grid = np.array([[axes_grid]])
        
        # Compute defaults
        xlabel_side, ylabel_side = cls._normalize_sides(
            triangle, xlabel_side, ylabel_side
        )
        
        # Build cells with standard pairwise semantics
        cells = []
        cells_by_col = {j: [] for j in range(n)}
        cells_by_row = {i: [] for i in range(n)}
        
        for i in range(n):
            for j in range(n):
                ax = axes_grid[i, j]
                
                if not cls._keep_triangle(i, j, n, triangle):
                    ax.set_visible(False)
                    continue
                
                # Standard semantics: diagonal = 1D, off-diagonal = 2D
                if i == j:
                    variables = (labels[i],)
                else:
                    variables = (labels[j], labels[i])  # (x, y)
                
                # Compute edge positions
                x_edge, y_edge = cls._compute_edge_position(
                    i, j, n, xlabel_side, ylabel_side
                )
                
                cell = Cell(ax, variables, x_edge=x_edge, y_edge=y_edge)
                cells.append(cell)
                
                cells_by_col[j].append(cell)
                cells_by_row[i].append(cell)
        
        # Share axes
        cls._share_xaxes(cells_by_col)
        cls._share_yaxes(cells_by_row)
        
        grid = cls(fig, cells, n, n)

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

    def plot_scatter_2d(self, df, /, styles=None, **kwargs):
        """
        Scatter plot data in 2D cells of this grid.
        
        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)
        
        Default style: alpha=0.3, edgecolor='w', linewidth=0.3, facecolor='k', marker='.', s=20
        """
        defaults = dict(alpha=0.3, edgecolor='w', linewidth=0.3, facecolor='k', marker='.', s=20)
        styles = styles or {}

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], df[y], **full_kwargs)


    def plot_scatter_2d_cmap(self, df, c, /, styles=None, **kwargs):
        """
        Scatter plot with color mapping in 2D cells of this grid.
        
        Args:
            df: DataFrame with variables
            c: color variable (string column name or array)
            styles: dict mapping (x_var, y_var) -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)
        
        Default style: alpha=0.3, edgecolor='none', linewidth=0.3, cmap='viridis', marker='.', s=20
        """
        defaults = dict(alpha=0.3, edgecolor='none', linewidth=0.3, cmap='viridis', marker='.', s=20)
        styles = styles or {}

        if isinstance(c, str):
            c = df[c]

        for cell in self.iter_cells(dim=2):
            x, y = cell.variables
            style = styles.get((x, y), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], df[y], c=c, **full_kwargs)


    def plot_line_2d(self, df, /, styles=None, **kwargs):
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


    def plot_hist_1d(self, df, /, styles=None, **kwargs):
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


    def plot_line_segmented_2d(self, df, /, styles=None, **kwargs):
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


    def plot_scatter_3d(self, df, /, styles=None, **kwargs):
        """
        3D scatter plot in 3D cells of this grid.
        
        Args:
            df: DataFrame with variables
            styles: dict mapping (x_var, y_var, z_var) -> style kwargs (default: None)
            **kwargs: additional scatter kwargs (override defaults)
        
        Default style: alpha=0.3, edgecolor='w', linewidth=0.3, c='k', marker='.', s=20
        """
        defaults = dict(alpha=0.3, edgecolor='w', linewidth=0.3, c='k', marker='.', s=20)
        styles = styles or {}
        
        for cell in self.iter_cells(dim=3):
            x, y, z = cell.variables
            style = styles.get((x, y, z), {})
            full_kwargs = {**defaults, **style, **kwargs}
            cell.ax.scatter(df[x], df[y], df[z], **full_kwargs)


    def plot_line_segmented_3d(self, df, /, styles=None, **kwargs):
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
