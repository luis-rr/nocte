from nocte.timeslice import Win, ms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nocte import plot as splot


class ScrollablePlot:
    def __init__(self, plot_zoomed, zoom_window: Win = ms(minutes=10), figsize=(6, 3), nrows=1):
        """
        Parameters:
        - plot_zoomed: Function to plot the zoomed-in section.
          It should take (ax, center, zoom_window) as arguments.
        - zoom_window: The number of points (or time range) to show in the zoomed-in panel.
        """
        self.plot_zoomed = plot_zoomed

        if isinstance(zoom_window, (float, int)):
            zoom_window = Win.build_centered(0, zoom_window)

        self.zoom_window = zoom_window

        # Create the figure and axes
        self.fig, axs = plt.subplots(
            1 + nrows, 1,
            figsize=figsize,
            gridspec_kw={'height_ratios': [1] + [3] * nrows},
            squeeze=False,
        )

        axs = axs.ravel()
        self.overview_ax = axs[0]
        self.zoom_axs = axs[1:]

        for ax in self.zoom_axs:
            ax.sharex(self.zoom_axs[0])

        splot.drop_spines_grid(self.zoom_axs)

        # Placeholder for the highlighted region in the overview
        self.highlight = None

        # Connect click event
        # noinspection PyTypeChecker
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def set_view(self, center):
        """Set the view with a given center point."""
        win = self.zoom_window.shift(center)
        self.update_fill(win)
        self.update_zoom(win)

    def update_fill(self, win):
        """Update the fill area in the overview plot to indicate the zoomed-in region."""
        if self.highlight:
            self.highlight.remove()  # Remove the previous fill

        self.highlight = self.overview_ax.axvspan(*win, facecolor='xkcd:magenta', alpha=0.3, linewidth=0.25, edgecolor='k', zorder=100)

    def update_zoom(self, win):
        """Call the user-defined function to plot the zoomed-in region."""
        for ax in self.zoom_axs:
            ax.clear()
        self.plot_zoomed(self.zoom_axs, win)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handles clicking on the overview to update the zoomed-in view."""
        if event.inaxes == self.overview_ax:
            self.set_view(event.xdata)


    @classmethod
    def build_for_timeseries(
            cls, data,
            zoom_window=Win.build_centered(0, ms(minutes=10)),
            scale_overview='hours',
            scale_zoom='seconds',
            overview_subsample=1000,
            **kwargs,
    ):

        def plot_zoomed(axs, zoom_win):
            ax = axs[0]
            ax.plot(
                data.crop(zoom_win, reset=True).traces,
                **kwargs,
            )
            splot.set_time_ticks(ax, scale=scale_zoom)

        sp = cls(plot_zoomed, zoom_window=zoom_window)

        sp.overview_ax.plot(
            data.traces.iloc[::overview_subsample],
            **kwargs,
        )
        splot.set_time_ticks(sp.overview_ax, scale=scale_overview)

        sp.set_view(np.median(data.drop_missing().time))

        return sp
