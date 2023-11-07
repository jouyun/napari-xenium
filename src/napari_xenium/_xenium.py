import pandas as pd
import napari
import numpy as np
import tifffile
import scanpy as sc
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.util as util
import matplotlib
import os
import plotly.express as px
import os
import gzip
import shutil
import cv2

def visualize_gene_by_cell(adata, gene, viewer):
    """
    Visualizes gene expression by cell in a scatter plot.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    gene (str): The name of the gene to visualize.
    viewer (napari.Viewer): The Napari viewer.

    Returns:
    None
    """

    cmap = matplotlib.colormaps['plasma']
    genes = adata.var.index.values

    values = np.sqrt(adata.raw.X[:,adata.var_names==gene][:,0].toarray()[:,0])
    colors = cmap(values/np.max(values))
    viewer.add_points(adata.obs[['y_centroid', 'x_centroid']].values, name=gene, face_color=colors, size=30, edge_color='black')

def visualize_gene_by_cell_w_boundaries(adata, gene, label_img, viewer):
    """
    Visualizes gene expression by cell in an image plot with cell boundaries.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    gene (str): The name of the gene to visualize.
    label_img (numpy.ndarray): The label image with cell boundaries.
    viewer (napari.Viewer): The Napari viewer.

    Returns:
    None
    """
    values = np.sqrt(adata.raw.X[:,adata.var_names==gene][:,0].toarray()[:,0])
    count_img = util.map_array(label_img, adata.obs['number'].values, values)
    viewer.add_image(count_img, name=gene, colormap='inferno', contrast_limits=[0, np.max(values)])

def visualize_by_transcript(df, gene, viewer):
    """
    Visualizes gene expression by transcript in a scatter plot.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the transcript data.
    gene (str): The name of the gene to visualize.
    viewer (napari.Viewer): The Napari viewer.

    Returns:
    None
    """
    sub_df = df[df['feature_name']==gene]
    viewer.add_points(sub_df[['pix_y', 'pix_x']], size=5, face_color='yellow', name=gene)

def get_transcripts(directory):
    """
    Loads the transcript data from a parquet file.

    Parameters:
    directory (str): The directory containing the parquet file.

    Returns:
    pandas.DataFrame: The DataFrame containing the transcript data.
    """
    trans_df = pd.read_parquet(directory + 'transcripts.parquet')
    trans_df = trans_df[~trans_df['feature_name'].str.contains('Codeword')]
    trans_df = trans_df[~trans_df['feature_name'].str.contains('BLANK')]
    trans_df = trans_df[~trans_df['feature_name'].str.contains('NegControl')]

    trans_df['pix_x'] = (trans_df['x_location'] * 4.706).astype(int)
    trans_df['pix_y'] = (trans_df['y_location'] * 4.706).astype(int)

    return trans_df

def get_DAPI_img(directory):
    """
    Loads the DAPI image from a TIFF file.

    Parameters:
    directory (str): The directory containing the TIFF file.

    Returns:
    numpy.ndarray: The DAPI image.
    """
    full_img = tifffile.imread(directory + 'morphology.ome.tif')
    orig_img = full_img.max(axis=0)
    return orig_img

def get_cell_df(directory):
    """
    Loads the cell data from a CSV file.

    Parameters:
    directory (str): The directory containing the CSV file.

    Returns:
    pandas.DataFrame: The DataFrame containing the cell data.
    """
    cell_df = pd.read_csv(directory + 'cells.csv')
    cell_df['x_centroid'] = cell_df['x_centroid'] * 4.706
    cell_df['y_centroid'] = cell_df['y_centroid'] * 4.706
    cell_df.set_index('cell_id', inplace=True)
    return cell_df

def get_adata(directory, cell_df):
    """
    Loads the AnnData object from a 10X HDF5 file and merges it with the cell data.

    Parameters:
    directory (str): The directory containing the HDF5 file.
    cell_df (pandas.DataFrame): The DataFrame containing the cell data.

    Returns:
    anndata.AnnData: The merged AnnData object.
    """
    adata = sc.read_10x_h5(directory + 'cell_feature_matrix.h5')
    merged = adata.obs.merge(cell_df, left_index=True, right_index=True, how='left')
    adata.obs = merged

    # Get 10X assigned clusters
    kmeans_df = pd.read_csv(directory + 'analysis/clustering/gene_expression_graphclust/clusters.csv')
    kmeans_df.set_index('Barcode', inplace=True)
    merged = adata.obs.merge(kmeans_df, left_index=True, right_index=True, how='left').fillna(-1)
    adata.obs = merged
    adata.obs['Cluster'] = adata.obs['Cluster'].astype(int).astype('category')

    return adata

def fill_function(x, label_img):
    """
    Fills a polygon in a label image.

    Parameters:
    x (pandas.Series): The series containing the polygon vertices.
    label_img (numpy.ndarray): The label image.

    Returns:
    None
    """    
    vertices = x[['vertex_x', 'vertex_y']].to_numpy()
    cv2.fillPoly(label_img, [vertices], int(x.iloc[0]['number']))

def make_label_img(directory, orig_img, bounds_df, adata):
    """
    Creates a label image from cell boundaries.

    Parameters:
    directory (str): The directory containing the data files.
    orig_img (numpy.ndarray): The original image.
    bounds_df (pandas.DataFrame): The DataFrame containing the cell boundaries.
    adata (anndata.AnnData): The AnnData object.

    Returns:
    numpy.ndarray: The label image.
    """
    bounds_df['vertex_x'] = bounds_df['vertex_x'] * 4.706
    bounds_df['vertex_y'] = bounds_df['vertex_y'] * 4.706
    bounds_df = bounds_df.merge(adata.obs, left_on='cell_id', right_index=True)
    label_img = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype=np.int32)
    bdf = bounds_df[['cell_id', 'vertex_x', 'vertex_y', 'leiden', 'number', 'Cluster']].copy()
    bdf['vertex_x'] = bdf['vertex_x'].astype(int)
    bdf['vertex_y'] = bdf['vertex_y'].astype(int)
    bdf['leiden'] = bdf['leiden'].astype(int)
    bdf['Cluster'] = bdf['Cluster'].astype(int)
    bdf['number'] = bdf['number'].astype(int)
    grouped = bdf.groupby('cell_id')
    grouped.apply(fill_function, label_img)
    return label_img

def make_leiden(adata, resolution=1.5, min_counts=20):
    """
    Performs Leiden clustering on the AnnData object.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    resolution (float): The resolution parameter for the Leiden algorithm.
    min_counts (int): The minimum number of counts for a cell to be included in the analysis.

    Returns:
    anndata.AnnData: The annotated data matrix with Leiden clusters.
    """    
    sc.pp.filter_cells(adata, min_counts=min_counts)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)

    return adata

def get_label_masks(directory, orig_img, adata):
    """
    Gets the label masks for the nucleus and cell boundaries.

    Parameters:
    directory (str): The directory containing the data files.
    orig_img (numpy.ndarray): The original image.
    adata (anndata.AnnData): The AnnData object.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray]: The label masks for the nucleus and cell boundaries.
    """
    if not os.path.exists(directory + 'nucleus_boundaries.npy'):
        bounds_df = pd.read_parquet(directory + 'cell_boundaries.parquet')
        nuclear_img = make_label_img(directory, orig_img, bounds_df, adata)
        np.save(directory + 'nucleus_boundaries.npy', nuclear_img)
    else:
        nuclear_img = np.load(directory + 'nucleus_boundaries.npy')

    if not os.path.exists(directory + 'cell_boundaries.npy'):
        bounds_df = pd.read_parquet(directory + 'cell_boundaries.parquet')
        cell_img = make_label_img(directory, orig_img, bounds_df, adata)
        np.save(directory + 'cell_boundaries.npy', cell_img)
    else:
        cell_img = np.load(directory + 'cell_boundaries.npy')
    return nuclear_img, cell_img

def load_xenium_data(directory, resolution=1.5):
    """
    Loads the Xenium data, including the transcript data, DAPI image, AnnData object, and label masks.

    Parameters:
    directory (str): The directory containing the data files.
    resolution (float): The resolution parameter for the Leiden algorithm.

    Returns:
    Tuple[pandas.DataFrame, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, anndata.AnnData]: The transcript data, DAPI image, nucleus label image, cell label image, Leiden cluster label image, cluster label image, and AnnData object.
    """
    trans_df = get_transcripts(directory)
    orig_img = get_DAPI_img(directory)

    if not os.path.exists(directory + 'cell_feature_matrix_clustered.h5ad'):
        cell_df = get_cell_df(directory)
        adata = get_adata(directory, cell_df)
        adata = make_leiden(adata, resolution=resolution)
        adata.obs['number'] = np.arange(1, adata.obs.shape[0]+1)
        adata.write(directory + 'cell_feature_matrix_clustered.h5ad')
    else:
        adata = sc.read_h5ad(directory + 'cell_feature_matrix_clustered.h5ad')

    nuclear_img, cell_img = get_label_masks(directory, orig_img, adata)

    
    leiden_img = util.map_array(cell_img.astype(int), adata.obs['number'].values, 1+(adata.obs['leiden'].astype(int).values))
    cluster_img = util.map_array(cell_img.astype(int), adata.obs['number'].values, 1+(adata.obs['Cluster'].astype(int).values))

    return trans_df, orig_img, nuclear_img, cell_img, leiden_img, cluster_img, adata

def reset_data(directory):
    """
    Deletes the processed data files.

    Parameters:
    directory (str): The directory containing the data files.

    Returns:
    None
    """
    if os.path.exists(directory + 'cell_feature_matrix_clustered.h5ad'):
        os.remove(directory + 'cell_feature_matrix_clustered.h5ad')
    if os.path.exists(directory + 'nucleus_boundaries.npy'):
        os.remove(directory + 'nucleus_boundaries.npy')
    if os.path.exists(directory + 'cell_boundaries.npy'):
        os.remove(directory + 'cell_boundaries.npy')

def show_subset(adata, mask, cell_img, viewer):
    """
    Shows a subset of the data based on a mask.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    mask (numpy.ndarray): The mask for the subset of cells to show.
    cell_img (numpy.ndarray): The label image with cell boundaries.
    viewer (napari.Viewer): The Napari viewer.

    Returns:
    None
    """    
    leiden_img = util.map_array(cell_img.astype(int), adata.obs['number'].values, (1+(adata.obs['leiden'].astype(int).values))*mask.astype(int))
    viewer.add_labels(leiden_img, name='leiden')
