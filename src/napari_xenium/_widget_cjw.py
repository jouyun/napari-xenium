"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLineEdit, QVBoxLayout, QFrame, QLabel
import cellpose.models as models
import numpy as np
import skimage.util as util
import tifffile
import skimage.data as data
import os
import time
import glob
import tifffile

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

class XeniumWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.root_directory = ''

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
        self.root_directory = (self.dir_field.text()) + '/'
        os.mkdir(self.root_directory + "/models")
        # Check to see if the viewer is empty, if it is, load the default image
        if len(self.viewer.layers) == 0:
            self.viewer.add_image(data.cells3d()[:,1])
        # Check to see if labels.tiff and rectangles.npz exist, if they do, load them
        # Otherwise create empty labels and rectangles
        if os.path.exists(self.root_directory + 'labels.tiff') and os.path.exists(self.root_directory + 'rectangles.npz'):
            labels = tifffile.imread(self.root_directory + 'labels.tiff').astype(np.uint16)
            npzfile = np.load(self.root_directory + 'rectangles.npz')
            rectangles = [npzfile[a] for a in npzfile.files]
        else:
            labels = np.zeros_like(self.viewer.layers[0].data).astype(np.uint16)
            rectangles = []

        # Add the rectangles and labels already found, this is a bug right now:  if the empty version goes in it fails
        # because it does not have the right dimensions (labels defaults to 2D)
        self.viewer.add_shapes(rectangles, ndim=3, shape_type='rectangle', edge_width=16, edge_color='red', face_color='white', opacity=0.11, name='AnnotatedROIs')
        self.viewer.add_labels(labels, opacity=0.5, name='AnnotatedLabels')

        self.scratch_train_btn.setEnabled(True)
        self.update_annotation_btn.setEnabled(True)
        self.retrain_btn.setEnabled(True)
        self.initialize_btn.setEnabled(False)

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

