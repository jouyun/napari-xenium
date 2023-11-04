"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
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

if TYPE_CHECKING:
    import napari

def minimize_labels(labels):
    uniques = np.unique(labels)
    return util.map_array(labels, uniques, np.arange(len(uniques)))

def infer_model(imgs, diameter, model, epochs, viewer):
    #model = models.CellposeModel(gpu=True, pretrained_model=model_file)
    print('Starting inference')
    nucleii, flows, styles = model.eval([i for i in imgs], channels=[0,0], diameter = diameter)
    viewer.add_labels(np.array(nucleii), name='rslts_' + str(diameter) + '_' + str(epochs))

def infer_model_file(imgs, diameter, model):
    if len(imgs.shape) > 2:
        nucleii, flows, styles = model.eval([i for i in imgs], channels=[0,0], diameter = diameter)
    else:
        nucleii, flows, styles = model.eval([imgs], channels=[0,0], diameter = diameter)
    return np.array(nucleii)

def get_sub_images(imgs, rectangles):
    sub_imgs = []
    for rec in rectangles:
        rec = rec.astype(int)
        sub_imgs.append(imgs[rec[0,0], np.min(rec[:,1]):np.max(rec[:,1]), np.min(rec[:,2]):np.max(rec[:,2])])
    return sub_imgs

def get_sub_labels(labels, rectangles):
    sub_labels = []
    for rec in rectangles:
        rec = rec.astype(int)
        labs = minimize_labels(labels[rec[0,0], np.min(rec[:,1]):np.max(rec[:,1]), np.min(rec[:,2]):np.max(rec[:,2])])
        sub_labels.append(labs)
    return sub_labels

def get_most_recent_file(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out directories and get the modification time for each file
    file_times = [(f, os.path.getmtime(os.path.join(folder_path, f))) \
                         for f in files if os.path.isfile(
                             os.path.join(folder_path, f))]

    # Sort the files by modification time (most recent first)
    sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)
    print("sorted files", len(sorted_files))
    # Return the name of the most recent file
    return sorted_files[0][0] if sorted_files else None

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
        print(self.selected_coordinates)

        if self.parent.plot_update is not None:
            self.parent.plot_update(self.selected_coordinates)


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
        data_x,data_y,colors, sizes, alpha,
    ):
        self.pts = self.axes.scatter(
            data_x,
            data_y,
            c=colors,
            s=sizes,
            alpha=alpha,
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

class XeniumWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.root_directory = ''

        self.figure = Figure()

        # Create the buttons
        self.initialize_btn = QPushButton("Initialize")
        self.scratch_train_btn = QPushButton("Train from scratch")
        self.update_annotation_btn = QPushButton("Transfer temporary labels")
        self.retrain_btn = QPushButton("Retrain")
        self.infer_btn = QPushButton("Infer On Opened Image")
        self.infer_dir_btn = QPushButton("Infer On Directory")

        # Connect the buttons to their respective functions
        self.initialize_btn.clicked.connect(self._initialize_click)
        self.scratch_train_btn.clicked.connect(self._scratch_train_click)
        self.update_annotation_btn.clicked.connect(self._update_labels_click)
        self.retrain_btn.clicked.connect(self._retrain_click)
        self.infer_btn.clicked.connect(self._infer_click)
        self.infer_dir_btn.clicked.connect(self._infer_dir_click)

        # Create the numeric field
        self.num_field = QLineEdit()
        self.num_field.setText("70")
        self.num_field_label = QLabel("Diameter:")
        self.num_field_label.setBuddy(self.num_field)

        self.dir_field = QLineEdit()
        self.dir_field.setPlaceholderText("")
        self.dir_field_label = QLabel("Directory:")
        self.dir_field_label.setBuddy(self.dir_field)

        self.epochs_field = QLineEdit()
        self.epochs_field.setText("1000")
        self.epochs_field_label = QLabel("Training iterations:")
        self.epochs_field_label.setBuddy(self.epochs_field)

        self.infer_dir_field = QLineEdit()
        self.infer_dir_field.setPlaceholderText("")
        self.infer_dir_field_label = QLabel("Infer Directory:")
        self.infer_dir_field_label.setBuddy(self.dir_field)

 

        # Add the widgets to the layout
        layout = QVBoxLayout()
        layout.addWidget(self.num_field_label)
        layout.addWidget(self.num_field)
        layout.addWidget(self.dir_field_label)
        layout.addWidget(self.dir_field)
        layout.addWidget(self.initialize_btn)

        
        # Setup the Plot Widget

        # This function will get called every time there is an update to the graph's selection
        def plot_update(inside):
            inside = np.array(inside)  # leads to errors sometimes otherwise

            self.viewer.add_image(data.coins())
            self.viewer.add_points(inside, size=10, face_color='red', name='AnnotatedROIs')

        # Add a figure widget
        self.figure_widget = PlotFigureCanvas(self.figure, plot_update=plot_update)
        graph_container = QWidget()
        graph_container.setLayout(QVBoxLayout())
        graph_container.layout().addWidget(self.figure_widget)
        layout.addWidget(graph_container, alignment=Qt.AlignTop)

        #layout.addSpacing(10)  # Add 10 pixels of spacing
        layout.addWidget(QFrame())
        layout.addWidget(self.epochs_field_label)
        layout.addWidget(self.epochs_field)
        layout.addWidget(self.scratch_train_btn)
        layout.addWidget(QFrame())
        layout.addWidget(self.update_annotation_btn)
        layout.addWidget(QFrame())
        layout.addWidget(self.retrain_btn)

        layout.addSpacing(200)  # Add 10 pixels of spacing
        layout.addWidget(self.infer_btn)
        layout.addSpacing(80)  # Add 10 pixels of spacing
        layout.addWidget(self.infer_dir_field_label)
        layout.addWidget(self.infer_dir_field)
        layout.addWidget(self.infer_dir_btn)

        self.setLayout(layout)

        self.scratch_train_btn.setEnabled(False)
        self.update_annotation_btn.setEnabled(False)
        self.retrain_btn.setEnabled(False)


    def _initialize_click(self):
        sz = 12
        df = pd.DataFrame({'X':np.random.randint(0,100,sz), 'Y':np.random.randint(0,100,sz), 'color':['blue']*sz, 'size':[10]*sz, 'alpha':[1]*sz})
        self.figure_widget.make_scatter_plot(df['X'].to_numpy(), df['Y'].to_numpy(), df['color'].values, df['size'].values, df['alpha'].values)


    def _scratch_train_click(self):

        imgs = self.viewer.layers[0].data

        rectangles = self.viewer.layers['AnnotatedROIs'].data
        labels = self.viewer.layers['AnnotatedLabels'].data

        tifffile.imwrite(self.root_directory+'labels.tiff', labels)
        np.savez(self.root_directory+'rectangles.npz', *rectangles)

        sub_imgs = get_sub_images(imgs, rectangles)
        sub_labels = get_sub_labels(labels, rectangles)

        #for i in sub_imgs:
        #    self.viewer.add_image(i)
        #for i in sub_labels:
        #    self.viewer.add_labels(i)

        diameter = int(self.num_field.text())
        self.cpm = models.CellposeModel(gpu=True, diam_mean=diameter, pretrained_model='cyto')

        svm = self.cpm.train(sub_imgs, sub_labels, test_data=None, test_labels=None,
          channels=[0,0], save_path=self.root_directory, n_epochs=int(self.epochs_field.text()), rescale=True,
            normalize=True, batch_size=50)
        infer_model(imgs, diameter, self.cpm, self.epochs_field.text(), self.viewer)

        self.update_annotation_btn.setEnabled(True)
        self.retrain_btn.setEnabled(True)
        #self.scratch_train_btn.setEnabled(False)

    def _update_labels_click(self):
        rectangles = self.viewer.layers['AnnotatedROIs'].data
        labels = self.viewer.layers['AnnotatedLabels'].data

        imgs = self.viewer.layers[0].data
        sub_imgs = get_sub_images(imgs, rectangles)

        # This time for sub_labels we have to be tricky, we might be using a new rectangle that
        # does not have existing labels, and we want to use the inferred labels as a start point
        # for that label
        sub_labels = []
        for rec in rectangles:
            rec = rec.astype(int)
            sub_lab = labels[rec[0,0], np.min(rec[:,1]):np.max(rec[:,1]), np.min(rec[:,2]):np.max(rec[:,2])]
            if (np.max(sub_lab)==0):
                sub_new_lab = self.viewer.layers[-1].data[rec[0,0], np.min(rec[:,1]):np.max(rec[:,1]), np.min(rec[:,2]):np.max(rec[:,2])].copy()
                #mask = sub_new_lab > 0
                #sub_new_lab[mask] = sub_new_lab[mask] - np.min(sub_new_lab[mask]) + np.max(labels) + 1
                self.viewer.layers['AnnotatedLabels'].data[rec[0,0], np.min(rec[:,1]):np.max(rec[:,1]), np.min(rec[:,2]):np.max(rec[:,2])] = sub_new_lab
                sub_labels.append(minimize_labels(sub_new_lab))
            else:
                sub_labels.append(minimize_labels(sub_lab))

        self.viewer.layers['AnnotatedLabels'].visible = False
        time.sleep(1)
        self.viewer.layers['AnnotatedLabels'].visible = True

        self.retrain_btn.setEnabled(True)


    def _retrain_click(self):
        rectangles = self.viewer.layers['AnnotatedROIs'].data
        labels = self.viewer.layers['AnnotatedLabels'].data

        tifffile.imwrite(self.root_directory+'labels.tiff', labels)
        np.savez(self.root_directory+'rectangles.npz', *rectangles)

        imgs = self.viewer.layers[0].data
        sub_imgs = get_sub_images(imgs, rectangles)
        sub_labels = get_sub_labels(labels, rectangles)

        diameter = int(self.num_field.text())

        #  For some reason this works better than starting from the model already made, this is
        #  more than a bit alarming
        # self.cpm = models.CellposeModel(gpu=True, diam_mean=diameter, pretrained_model='cyto')

        svm = self.cpm.train(sub_imgs, sub_labels, test_data=None, test_labels=None,
          channels=[0,0], save_path=self.root_directory, n_epochs=500, rescale=True,
            normalize=True, batch_size=50)
        infer_model(imgs, diameter, self.cpm, 500, self.viewer)

    def _infer_click(self):
        imgs = self.viewer.layers[0].data
        self.root_directory = (self.dir_field.text()) + '/'
        diameter = int(self.num_field.text())
        if hasattr(self, 'cpm'):
            print('Using existing model')
        else:
            print('Creating new model')
            model_file = get_most_recent_file(self.root_directory + 'models/')
            if model_file is None:
                self.cpm = models.CellposeModel(gpu=True,
                                                model_type='cyto2')
            else:
                model_file = self.root_directory + 'models/' +  model_file
                self.cpm = models.CellposeModel(gpu=True,
                                                pretrained_model=model_file )
            print(model_file)
        #  Fix this later
        infer_model(imgs, diameter, self.cpm, self.epochs_field.text(), self.viewer)

    def _infer_dir_click(self):
        self.infer_directory = (self.infer_dir_field.text()) + '/'
        fnames = glob.glob(self.infer_directory + '*.tif')
        diameter = int(self.num_field.text())
        if hasattr(self, 'cpm'):
            print('Using existing model')
        else:
            print('Loading existing model')
            model_file = self.root_directory + 'models/' + get_most_recent_file(self.root_directory + 'models/')
            print(model_file)
            self.cpm = models.CellposeModel(gpu=True, pretrained_model=model_file )
        #  Fix this later
        for f in fnames:
            imgs = tifffile.imread(f)
            output = infer_model_file(imgs, diameter, self.cpm)
            tifffile.imwrite(f[:-4] + '_labels.tiff', output)

