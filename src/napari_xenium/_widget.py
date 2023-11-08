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
from ._visualize import SelectFromCollection
from ._visualize import PlotFigureCanvas
from ._xenium import load_xenium_data, show_subset, visualize_by_transcript, visualize_gene_by_cell_w_boundaries, visualize_gene_by_cell
from qtpy.QtWidgets import QComboBox, QCompleter
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QFileDialog

if TYPE_CHECKING:
    import napari

class XeniumWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.root_directory = ''

        self.figure = Figure()

        # Create the buttons
        self.initialize_btn = QPushButton("Initialize")
        self.initialize_btn.clicked.connect(self._initialize_click)

        # Create the numeric field
        self.num_field = QLineEdit()
        self.num_field.setText("1.5")
        self.num_field_label = QLabel("UMAP Resolution:")
        self.num_field_label.setBuddy(self.num_field)

        self.dir_field = QLineEdit()
        self.dir_field.setPlaceholderText("")
        self.dir_field_label = QLabel("Directory:")
        self.dir_field_label.setBuddy(self.dir_field)

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

            show_subset(self.adata, inside, self.cell_img, self.viewer)

        # Add a figure widget
        self.figure_widget = PlotFigureCanvas(self.figure, plot_update=plot_update, width=5, height=5)
        graph_container = QWidget()
        graph_container.setLayout(QVBoxLayout())
        graph_container.layout().addWidget(self.figure_widget)
        layout.addWidget(graph_container, alignment=Qt.AlignTop)



        list_of_genes = ['ABCD', 'EFGH']
         # Create a QListWidget
        self.gene_list_widget = QListWidget()
        # Add items to the QListWidget
        for gene in list_of_genes:  # replace list_of_genes with your list of gene names
            self.gene_list_widget.addItem(gene)
        # Add the QListWidget to the layout
        self.gene_list_widget.itemSelectionChanged.connect(self.on_gene_selected)
        layout.addWidget(self.gene_list_widget)


        # Finish the layout
        self.setLayout(layout)

        #self.scratch_train_btn.setEnabled(False)

    def _initialize_click(self):
        #data_directory = 'U:/smc/public/SMC/Xenium/pat/output-XETG00063__0010721__Region_1__20231011__183002/'
        data_directory = QFileDialog.getExistingDirectory(self, "Select Directory") + '/'
        self.dir_field.setText(data_directory)
        self.trans_df, self.orig_img, self.nuclear_img, self.cell_img, self.leiden_img, self.cluster_img, self.adata = load_xenium_data(data_directory, resolution=float(self.num_field.text()))
        self.viewer.add_image(self.orig_img, name='original')
        self.viewer.add_labels(self.nuclear_img, name='nucleus_boundaries')
        self.viewer.add_labels(self.cell_img, name='cell_boundaries')
        self.viewer.add_labels(self.leiden_img, name='leiden')
        self.viewer.add_labels(self.cluster_img, name='kmeans')

        sz = self.adata.obsm['X_umap'].shape[0]
        
        leiden_layer = self.viewer.layers['leiden']
        colors = leiden_layer.get_color(np.array(self.adata.obs['leiden'].astype(int).values + 1).tolist())
        self.figure_widget.make_scatter_plot(data_x=self.adata.obsm['X_umap'][:,0], data_y=self.adata.obsm['X_umap'][:,1], colors=colors)


        # Update the list of genes
        genes = self.adata.var_names.tolist()
        self.gene_list_widget.clear()
        # Add the new items to the QListWidget
        for gene in genes:
            self.gene_list_widget.addItem(gene)


    def on_gene_selected(self):
        # Get the selected item
        selected_gene = self.gene_list_widget.currentItem().text()

        # Do something with the selected gene
        visualize_gene_by_cell_w_boundaries(self.adata, selected_gene, self.cell_img, self.viewer)
        visualize_by_transcript(self.trans_df, selected_gene, self.viewer)
        print(f"Selected gene: {selected_gene}")
