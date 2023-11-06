from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLineEdit, QVBoxLayout, QFrame, QLabel
import numpy as np
import skimage.util as util
import tifffile
import skimage.data as data
import os
import time
import glob
import tifffile
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from matplotlib.widgets import LassoSelector, RectangleSelector, SpanSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

# Class below was based upon matplotlib lasso selection example:
# https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, parent, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect, button=1)
        self.ind = []
        self.ind_mask = []
        

    def onselect(self, verts):
        path = Path(verts)
        self.ind_mask = path.contains_points(self.xys)
        self.ind = np.nonzero(self.ind_mask)[0]

        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.selected_coordinates = self.xys[self.ind].data
        
        if self.parent.plot_update is not None:
            self.parent.plot_update(self.ind_mask)


    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


class PlotFigureCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=7, plot_update=None):
        self.fig = Figure(figsize=(width, height), constrained_layout=True)
        self.plot_update = plot_update

        self.axes = self.fig.add_subplot(111)
        self.histogram = None

        self.match_napari_layout()
        self.xylim = None
        self.last_xy_labels = None

        super().__init__(self.fig)
        self.mpl_connect("draw_event", self.on_draw)
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_fun)

        self.pts = self.axes.scatter([], [])
        self.selector = SelectFromCollection(self, self.axes, self.pts)
        self.rectangle_selector = RectangleSelector(
            self.axes,
            self.draw_rectangle,
            useblit=True,
            props=dict(edgecolor="white", fill=False),
            button=3,  # right button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.selected_colormap = "magma"

        self.reset()

    def reset_zoom(self):
        if self.xylim:
            self.axes.set_xlim(self.xylim[0])
            self.axes.set_ylim(self.xylim[1])

    def on_draw(self, event):
        print('Called draw')
        #self.last_xy_labels = (self.axes.get_xlabel(), self.axes.get_ylabel())
        #self.xylim = (self.axes.get_xlim(), self.axes.get_ylim())
    
    def zoom_fun(self, event):
       # get the current x and y limits
        base_scale = 1.5
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        self.axes.set_xlim([xdata - cur_xrange*scale_factor,
                           xdata + cur_xrange*scale_factor])
        self.axes.set_ylim([ydata - cur_yrange*scale_factor,
                           ydata + cur_yrange*scale_factor])
        self.draw()

    def draw_rectangle(self, eclick, erelease):
        """eclick and erelease are the press and release events"""
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        self.xys = self.pts.get_offsets()
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        self.rect_ind_mask = [
            min_x <= x <= max_x and min_y <= y <= max_y
            for x, y in zip(self.xys[:, 0], self.xys[:, 1])
        ]

    def reset(self):
        self.axes.clear()
        self.is_pressed = None

    def make_scatter_plot(
        self,
        #data_x: "numpy.typing.ArrayLike",
        #data_y: "numpy.typing.ArrayLike",
        #colors: "typing.List[str]",
        #sizes: "typing.List[float]",
        #alpha: "typing.List[float]",
        data_x,data_y,colors,
    ):
        self.pts = self.axes.scatter(
            data_x,
            data_y,
            c=colors,
            s=1,
        )
        self.selector.disconnect()
        self.selector = SelectFromCollection(
            self,
            self.axes,
            self.pts,
        )
    
    def match_napari_layout(self):
        """Change background and axes colors to match napari layout"""
        # changing color of axes background to napari main window color
        self.fig.patch.set_facecolor("#262930")
        # changing color of plot background to napari main window color
        self.axes.set_facecolor("#262930")

        # changing colors of all axes
        self.axes.spines["bottom"].set_color("white")
        self.axes.spines["top"].set_color("white")
        self.axes.spines["right"].set_color("white")
        self.axes.spines["left"].set_color("white")
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")

        # changing colors of axes ticks
        self.axes.tick_params(axis="x", colors="white", labelcolor="white")
        self.axes.tick_params(axis="y", colors="white", labelcolor="white")

        # changing colors of axes labels
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")
        self.fig.canvas.draw_idle()
