#import common libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import warnings
import logging
import pytest
import sklearn
import sklearn.cluster as skclust
from scipy.sparse import csr_matrix, lil_matrix
from bisect import bisect_right
from bisect import bisect_left
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import umap
import hdbscan
from sklearn.mixture import GaussianMixture
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib import use as mplt_use
from matplotlib.ticker import FixedLocator
from scipy.stats import pearsonr, spearmanr
import random

#import HiCExplorer
from hicmatrix import HiCMatrix as hm
from pybedtools import BedTool
from hicexplorer._version import __version__
from hicexplorer.utilities import obs_exp_matrix
from hicexplorer.utilities import convertNansToZeros, convertInfsToZeros

#get logger
mplt_use('Agg')
log = logging.getLogger(__name__)

def read_matrix_file(matrix_file, pChromosome):
    ''''reads a given cool file and returns its hiCMatrix and numpy representation'''
    
    log.debug('matrix_file {}'.format(matrix_file))
    log.debug('pChromosome {}'.format(pChromosome))
    
    # check if instance of string or file and load appropriate
    if isinstance(matrix_file, str):
        if pChromosome is not None:
            hic_ma = hm.hiCMatrix(matrix_file, pChrnameList=[pChromosome])
        else:
            hic_ma = hm.hiCMatrix(matrix_file)

    else:
        hic_ma = matrix_file

    return hic_ma

def obs_exp_normalization(hic_ma, pThreads=None):
    '''apply obs_exp normalization'''
    log.debug('obs/exp matrix computation...')

    trasf_matrix = lil_matrix(hic_ma.matrix.shape)

    # from hicTransformTADs
    def _obs_exp(pSubmatrix, pThreads=None):
        obs_exp_matrix_ = obs_exp_matrix(pSubmatrix)
        obs_exp_matrix_ = convertNansToZeros(
            csr_matrix(obs_exp_matrix_))
        obs_exp_matrix_ = convertInfsToZeros(
            csr_matrix(obs_exp_matrix_))
        # if len(obs_exp_matrix_.data) == 0:
        # return np.array([[]])
        return obs_exp_matrix_  # .todense()

    for chrname in hic_ma.getChrNames():
        chr_range = hic_ma.getChrBinRange(chrname)
        submatrix = hic_ma.matrix[chr_range[0]:chr_range[1], chr_range[0]:chr_range[1]]
        submatrix.astype(float)
        obs_exp = _obs_exp(submatrix, pThreads)
        if obs_exp.nnz != 0:
            trasf_matrix[chr_range[0]:chr_range[1], chr_range[0]:chr_range[1]] = lil_matrix(obs_exp)

    hic_ma.setMatrix(
        trasf_matrix.tocsr(),
        cut_intervals=hic_ma.cut_intervals)
    log.debug('obs/exp matrix computation... DONE')

    return hic_ma

def read_regions_bed(bed_file):
    '''read bed file containing interactions'''
    
    # read and sort
    bed_df = pd.read_csv(bed_file, sep="\t", header=None)

    assert (bed_df.shape[1] == 6 or bed_df.shape[1] == 5)
    
    if(bed_df.shape[1] == 6):
        bed_df.columns = get_regions_bed_col_names()
    else:
        bed_df.columns = get_regions_bed_col_names_5()
        bed_df['Strand'] = bed_df['UnknownCol2']
    
    bed_df = bed_df.sort_values(by=["Chrom", "Start"])

    if(bed_df.size < 1):
        raise ValueError('empty bed file passed')
        
    bed_df.set_index(np.arange(0,bed_df.shape[0]),inplace=True)
    bed_df['RegionIndex'] = np.arange(0,bed_df.shape[0])

    return bed_df

def get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1):
    """get valid interaction pairs within given distance constraints"""
    
    #get chromosomes
    chromosomes = regions['Chrom'].unique()
    pairs = []
    
    #collect valid pairs for each chromosome
    for c in chromosomes:
        
        #get regions for current chromosome
        chr_regions = regions.loc[regions['Chrom'] == c]
        chr_regions = chr_regions.sort_values(by=["Start"])
        
        for index,row in chr_regions.iterrows():
            
            #filter regions to those in given interval
            pos = row['Start'] * resolution
            current = chr_regions[chr_regions['Start'] * resolution >= pos + min_distance]
            current = current[current['Start'] * resolution <= pos + max_distance]
            
            #TODO: check if binary search is faster
            #pos = row['Start'] * resolution
            #le = find_le(chr_regions['Start'], (pos + min_distance) / resolution)
            #ge = find_ge(chr_regions['Start'], (pos + max_distance) / resolution) + 1
            #current = chr_regions[le:ge]
            
            #build pair dataframe
            #current = current.copy()
            current['pairChrom'] = row['Chrom']
            current['pairStart'] = row['Start']
            current['pairEnd'] = row['End']
            current['pairRegionIndex'] = row['RegionIndex']
            
            if 'TestLabel' in chr_regions.columns:
                current['pairTestLabel'] = row['TestLabel']
            
            pairs.append(current)
            
    pairs = pd.concat(pairs)
    pairs.set_index(np.arange(0,pairs.shape[0]),inplace=True)
    pairs['MatrixIndex'] = np.zeros((len(pairs)), dtype='int')
    pairs['pairMatrixIndex'] = np.zeros((len(pairs)), dtype='int')
    
    return pairs

def get_submatrices(matrix,regions,pairs,submatrix_size=9):
    """collect submatrices for every pair of regions"""
    
    #get submatrices for these pairs
    submatrices = []
    #pos_dict = build_position_index(get_positions(matrix))
    pos_dict = build_position_index_bisect(get_positions(matrix))
    chromosomes = pairs['Chrom'].unique()
    chr_pos = dict(zip(chromosomes,map(matrix.getChrBinRange,chromosomes)))
    submatrix_radius = math.floor(submatrix_size / 2)
    remove_rows = []
    
    for index,row in pairs.iterrows():
        
        #get index of positions
        #i = pos_dict[(row['Chrom'],row['Start'])]
        #j = pos_dict[(row['Chrom'],row['pairStart'])]
        
        try:
            pos_chr_list = pos_dict[row['Chrom']]
            i = find_le(pos_chr_list, row['Start']) + chr_pos[row['Chrom']][0]
            
            pos_chr_list = pos_dict[row['pairChrom']]
            j = find_le(pos_chr_list, row['pairStart']) + chr_pos[row['pairChrom']][0]
            
            log.debug('position: ' + str(i) + ' ' + str(j) + ' in matrix: ' + str(matrix.matrix.shape[0]))
            
            row['MatrixIndex'] = i
            row['pairMatrixIndex'] = j

            #normal cases inside matrix
            if(i >= submatrix_size and j >= submatrix_size and i < matrix.matrix.shape[0] - submatrix_size and j < matrix.matrix.shape[0] - submatrix_size):

                up_i = i + submatrix_radius + 1
                lo_i = i - submatrix_radius
                up_j = j + submatrix_radius + 1
                lo_j = j - submatrix_radius
                submatrices.append(matrix.matrix[lo_i:up_i,lo_j:up_j].toarray())

            #TODO: cases at the border of the matrix, for which the submatrix crosses over the edge of the matrix
            else:
                #submatrices.append(np.full((up_i-lo_i,up_j-lo_j),0.0))
                remove_rows.append(index)
                log.warn('position of interaction pair at the edge of matrix')
            
        except ValueError:
            log.warn('position of interaction pair not found in matrix')
            #row['Index'] = -1
            #row['pairIndex'] = -1
            #submatrices.append(np.full((up_i-lo_i,up_j-lo_j),0.0))
            remove_rows.append(index)
            
    #drop all regions which cannot be clustered, because they are not (fully) contained in matrix
    pairs = pairs.drop(index=remove_rows)
    regions = regions[regions['RegionIndex'].isin(pd.concat([pairs['RegionIndex'],pairs['pairRegionIndex']]))]
    
    regions.rename(columns = {'RegionIndex':'RegionIndexOld'}, inplace = True)
    regions['RegionIndex'] = np.arange(0,regions.shape[0])
    regions['pairRegionIndexOld'] = regions['RegionIndexOld']
    regions['pairRegionIndex'] = regions['RegionIndex']
    
    pairs.rename(columns = {'RegionIndex':'RegionIndexOld'}, inplace = True)
    pairs.rename(columns = {'pairRegionIndex':'pairRegionIndexOld'}, inplace = True)
    
    pairs = pd.merge(pairs, regions[['RegionIndexOld','RegionIndex']], how='inner', left_on=['RegionIndexOld'], right_on=['RegionIndexOld'])
    pairs = pd.merge(pairs, regions[['pairRegionIndexOld','pairRegionIndex']], how='inner', left_on=['pairRegionIndexOld'], right_on=['pairRegionIndexOld'])
    
    regions.drop(columns=['RegionIndexOld','pairRegionIndexOld','pairRegionIndex'])
    pairs.drop(columns=['RegionIndexOld','pairRegionIndexOld'])
    
    return regions,pairs,submatrices
    
def get_positions(matrix):
    '''get positions for matrix'''

    # get start and end position for every bin in the matrix
    indices = np.arange(0, matrix.matrix.shape[0])
    vec_bin_pos = np.vectorize(matrix.getBinPos)
    pos = vec_bin_pos(indices)

    # return it as an ordered array
    pos = np.transpose(np.array(pos)[0:3, :], (1, 0))
    pos_df = pd.DataFrame(data=pos, index=np.arange(0,pos.shape[0]), columns=["Chrom", "Start", "End"])
    dtype_dict = {'Start': int, 'End': int}
    pos_df = pos_df.astype(dtype_dict)
    return pos_df

def build_position_index(positions):
    '''get reverse index for positions on matrix'''
    
    #this can only work for data, where both the matrix and the region file are correctly binned
    #for general case, using binary search seems the better approach
    chrom = positions['Chrom'].to_numpy()
    start = positions['Start'].to_numpy()
    indices = np.arange(0,chrom.shape[0])
    keys = list(zip(chrom, start))
    pos_dict = dict(zip(keys,indices))
    return pos_dict

def build_position_index_bisect(positions):
    """get reverse index for positions on matrix for binary search"""

    #get chromosomes
    chromosomes = positions['Chrom'].unique()
    chr_dict = {}
    
    #build dictionary with ordered lists
    for c in chromosomes:
        positions = positions.loc[positions['Chrom'] == c]
        positions = positions.sort_values(by=["Start"])
        
        chr_dict[c] = positions['Start'].to_numpy()
        
    return chr_dict
    
def get_features(submatrices,center_size=0.2,corner_position=None,corner_size=2):
    '''select features from the given submatrices'''
    
    #compute center square of matrix
    submatrix_size = submatrices[0].shape[0]
    submatrix_radius = math.floor(submatrix_size / 2.0)
    center = math.ceil(submatrix_size / 2.0)

    center_abs_size = max(math.floor(submatrix_radius * center_size),1)
    
    lo_c = center - center_abs_size
    hi_c = center + center_abs_size + 1
    shape_ = (hi_c-lo_c)**2

    features = np.zeros((len(submatrices),shape_))
        
    #build features on list of submatrices
    def build_features(s):
        
        if(not s is None):
            f = s[lo_c:hi_c,lo_c:hi_c].flatten()

            if (not corner_position is None):
                if (corner_position == 'upper_left'):
                    corner = s[0:corner_size,0:corner_size]

                elif (corner_position == 'upper_right'):
                    corner = s[0:corner_size,-corner_size:]

                elif (corner_position == 'lower_right'):
                    corner = s[-corner_size:,-corner_size:]

                elif (corner_position == 'lower_left'):
                    corner = s[-corner_size:,0:corner_size]

                m = np.nanmean(corner)

                if(m > 0):
                    f = f / m

            return f
        
        else:
            return np.full(shape_,np.nan)
    
    for i in range(0,len(submatrices)):
        features[i,:] = build_features(submatrices[i])
        
    return features
    
#def get_indices_list(pairs):
#    '''get indices of regions'''

#    pair_positions = pairs[['pairChrom','pairStart','pairEnd']].rename(columns = {'pairChrom':'Chrom', 'pairStart': 'Start', 'pairEnd': 'End'}, inplace = False)
#    indices_list = pd.concat([pairs[['Chrom','Start','End']].copy(),pair_positions])
#    indices_list = indices_list.drop_duplicates(subset=['Chrom','Start']).sort_values(by=["Chrom","Start"])
#    indices_list['featureIndex'] = np.arange(len(indices_list['Chrom']))
#    indices_list.set_index(np.arange(0,len(indices_list)))
    
#    return indices_list

def get_feature_matrix(pairs,features,regions,dev_feature_type='per_region_flattened'):
    '''build features per region from features per interaction'''
    
    if(dev_feature_type=='per_region_flattened'):
        feature_matrix = np.zeros((regions.shape[0],regions.shape[0]*len(features[0])))

        for index,row in pairs.iterrows():

            i = row['RegionIndex']
            j = row['pairRegionIndex']
            i_f = row['RegionIndex'] * len(features[0])
            j_f = row['pairRegionIndex'] * len(features[0])
            i_f1 = (row['RegionIndex'] + 1) * len(features[0])
            j_f1 = (row['pairRegionIndex'] + 1) * len(features[0])

            feature_matrix[i,j_f:j_f1] = features[index]
            feature_matrix[j,i_f:i_f1] = features[index]
            
    else:
        feature_matrix = np.zeros((regions.shape[0],regions.shape[0]))

        for index,row in pairs.iterrows():

            i = row['RegionIndex']
            j = row['pairRegionIndex']

            feature_matrix[i,j] = np.nanmean(features[index])
            feature_matrix[i,j] = np.nanmean(features[index])
        
    return feature_matrix

def perform_clustering(features,k,cluster_algorithm='kmeans'):
    '''perform cluster algorithm on data'''
    
    cluster_labels = None
    
    if(cluster_algorithm == 'kmeans'):
        clustering = skclust.KMeans(n_clusters=k, random_state=0).fit(features)
        cluster_labels = clustering.labels_
    
    elif(cluster_algorithm == 'agglomerative_hierarchical'):
        clustering = skclust.AgglomerativeClustering(n_clusters=k).fit(features)
        cluster_labels = clustering.labels_
        
    elif(cluster_algorithm == 'gaussian_mixture'):
        clustering = GaussianMixture(n_components=k).fit(features)
        cluster_labels = clustering.predict(features)
        
    elif(cluster_algorithm == 'hdbscan'):
        clustering = hdbscan.HDBSCAN(min_cluster_size=10)
        cluster_labels = clustering.fit_predict(features)
        
    elif(cluster_algorithm == 'community_detection'):
        #reformulate as graph community detection problem using python igraph
        raise NotImplementedError
        
    else:
        raise ValueError

    return pd.Series(cluster_labels)

def get_regions_bed_col_names():
    '''get column names'''

    return ['Chrom', 'Start', 'End', 'UnknownCol1', 'UnknownCol2', 'Strand']

def get_regions_bed_col_names_5():
    '''get column names'''

    return ['Chrom', 'Start', 'End', 'UnknownCol1', 'UnknownCol2']

def output_results(out_file_contact_pairs,pairs):
    '''output results to bed file'''

    #TODO
    pairs['Strand'] = pairs['UnknownCol1']
    
    pairs_out = pairs[['Chrom','Start','End','UnknownCol1','UnknownCol2','Strand','pairChrom','pairStart','pairEnd','Cluster']]
    pairs_out.to_csv(out_file_contact_pairs, sep='\t', header=None, index=False)

def perform_plotting_preprocessing(features,n_components=2,preprocessing_type=None):
    '''perform preprocessing for plotting'''
    
    if(preprocessing_type is None):
        Sc = StandardScaler()
        scaled = Sc.fit_transform(features)
        pca = PCA(n_components)
        pca.fit(scaled)
        reduced = pca.transform(scaled)

        #um = umap.UMAP(n_components=n_components, init='random', random_state=0)
        #reduced = um.fit_transform(scaled)
        
    else:
        reduced = features[:,0:n_components]
    
    return reduced

def perform_clustering_preprocessing(features,preprocessing_type=None,n_components=20,umap_n_neighbours=20,umap_metric='braycurtis',umap_min_dist=0.5):
    '''perform preprocessing for clustering'''
    
    if(preprocessing_type == 'umap'):
        return umap.UMAP(n_components=n_components,metric=umap_metric,n_neighbors=umap_n_neighbours,min_dist=umap_min_dist).fit_transform(features)
            
    if(preprocessing_type == 'pca'):
        Sc = StandardScaler()
        scaled = Sc.fit_transform(features)
        return PCA(n_components=n_components).fit_transform(features)
    
    else:
        return features
    
def plot_results(features,clusters,out_file_fig,scatter_plot_type='3d',preprocessing_type=None):
    '''plot clustering results'''
    
    fig = plt.figure(dpi=150)
    
    ax = fig.add_subplot(projection=scatter_plot_type)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    
    if(scatter_plot_type == '3d'):
        components = perform_plotting_preprocessing(features,n_components=3,preprocessing_type=preprocessing_type)
        scatter = ax.scatter(components[:,0],components[:,1],components[:,2],c=clusters,cmap='Set3',alpha=0.8)
        ax.set_zlabel('component 3')
        
    else:
        components = perform_plotting_preprocessing(features,n_components=2,preprocessing_type=preprocessing_type)
        scatter = ax.scatter(components[:,0], components[:,1],c=clusters,cmap='Set3',alpha=0.8)        
    
    legend1 = ax.legend(*scatter.legend_elements(),loc="upper left", title="")
    ax.add_artist(legend1)
    
    #fig.show()
    fig.savefig(out_file_fig)
    plt.close()

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    
    #source: https://docs.python.org/3/library/bisect.html
    i = bisect_right(a, x)
    if i:
        return i-1
    raise ValueError
    
def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    
    #source: https://docs.python.org/3/library/bisect.html
    i = bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError
    
def plot_submatrices(submatrices, clusters, out_file_name,vmin=None,vmax=None,colormap='RdYlBu_r',plot_aggr_mode='mean'):
    '''plot mean submatrices per cluster and for all regions'''
    
    assert len(clusters) == len(submatrices)
    clusters = pd.Series(clusters)
    cluster_list = clusters.unique()
    cluster_list.sort()
    aggr_submatrices = []
    
    submatrices = np.array(submatrices)
    clusters = clusters.to_numpy()
    M_half = int((submatrices[0].shape[0] - 1) // 2)
    
    for c in cluster_list:
        cluster_indices = clusters == c
        cluster_submatrices = submatrices[cluster_indices]
        
        if(plot_aggr_mode == 'median'):
            aggr_submatrix = np.nanmedian(cluster_submatrices, axis=0)
        else:
            aggr_submatrix = np.nanmean(cluster_submatrices, axis=0)
            
        aggr_submatrices.append(aggr_submatrix)
    
    
    if(plot_aggr_mode == 'median'):
        aggr_submatrix_all = np.nanmedian(submatrices, axis=0)
    else:
        aggr_submatrix_all = np.nanmean(submatrices, axis=0)
    
    assert len(aggr_submatrices) == len(cluster_list)
    
    fig = plt.figure(figsize=(5.5 * (len(cluster_list) + 1), 5.5))
    gs = gridspec.GridSpec(1,(len(cluster_list) + 1),wspace=0.1, hspace=0.1)

    gs.update(wspace=0.01, hspace=0.2)

    for cluster_number in range(0,len(aggr_submatrices)):
            title = "cluster_{}".format(cluster_number)
            ax = plt.subplot(gs[0,cluster_number])
            ax.set_title(title)
            
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_vertical(size="5%", pad=0.3,pack_start=True)
            fig = ax.get_figure()
            fig.add_axes(ax_cb)
            img = ax.imshow(aggr_submatrices[cluster_number], aspect='equal',interpolation='nearest',extent=[-M_half, M_half + 1, -M_half, M_half + 1],cmap = colormap,vmin=vmin,vmax=vmax)

            mappableObject = plt.cm.ScalarMappable(cmap = colormap)
            mappableObject.set_array(aggr_submatrices[cluster_number])
            plt.colorbar(mappableObject, cax = ax_cb,orientation='horizontal')
            ax_cb.xaxis.tick_bottom()
            ax_cb.xaxis.set_tick_params(labelbottom=True)
            

    
    title = 'all'
    ax = plt.subplot(gs[0,len(cluster_list)])
    ax.set_title(title)
    
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.3,pack_start=True)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    
    img = ax.imshow(aggr_submatrix_all, aspect='equal',interpolation='nearest',extent=[-M_half, M_half + 1, -M_half, M_half + 1],cmap = colormap,vmin=vmin,vmax=vmax)
    mappableObject = plt.cm.ScalarMappable(cmap = colormap)
    mappableObject.set_array(aggr_submatrix_all)
    plt.colorbar(mappableObject, cax = ax_cb, orientation='horizontal')
    ax_cb.xaxis.tick_bottom()
    ax_cb.xaxis.set_tick_params(labelbottom=True)    
    plt.savefig(out_file_name, dpi=300)
    #plt.show()
    plt.close()
    
def region_to_pair_labels(pairs,labels,col_name):
    '''merge region labels to pairs'''
    
    cl_df = pd.DataFrame(columns=[col_name])
    cl_df[col_name] = labels
    cl_df['RegionIndex'] = np.arange(0,labels.shape[0])
    o = pd.merge(pairs[['RegionIndex']], cl_df, how='inner', left_on=['RegionIndex'], right_on=['RegionIndex']) 
    
    return o[col_name]

def outlier_cropping_and_transformation(X,min_value=None,max_value=None,transform=None):
    '''crop statistical outliers and transform '''
    
    assert min_value is None or max_value is None or min_value <= max_value
    
    def oc_t_np(x):
        if(not max_value is None):
            np.minimum(x,max_value,out=x)

        if(not min_value is None):
            np.maximum(x,min_value,out=x)

        if(transform == 'log1p'):
            np.log1p(x,out=X)

        return x
    
    if(isinstance(X,list)):
        X = [oc_t_np(x) for x in X]
    else:
        X = oc_t_np(X)
        
    return X

def plot_density(np_array,out_file_prefix):
    '''plot density and boxplot of features'''
    
    fig = plt.figure(figsize =(10, 7))
    #ax = fig.add_axes([0,0,1,1])
    fl = np_array.flatten()
    fl_size = fl.shape[0]
    fl_nz = fl[fl > 0]
    frac_zero = str(fl_nz.shape[0] / fl.shape[0])
    l = min(4,len(frac_zero))
    frac_zero = frac_zero[0:l]
    title = 'density of feature values'

    sns.set_style('whitegrid')
    sns_plot = sns.kdeplot(fl, bw=0.5, cut=0)
    sns_plot.set(title=title)

    fig = sns_plot.get_figure()
    fig.savefig(out_file_prefix + '_feature_density.png')
    plt.close()

    plt.boxplot(fl_nz)
    plt.title('boxplot of non-zero feature values; proportion of zeros: ' + frac_zero)
    plt.savefig(out_file_prefix + '_feature_boxplot.png', dpi=300)
    plt.close()

def plot_diagnostic_heatmaps(submatrices,clusters,out_file_prefix,vmin=None,vmax=None,colormap='RdYlBu_r',dpi=300,transform='log1p'):
    '''plot submatrices heatmap by cluster'''

    M_half = int((submatrices[0].shape[0] - 1) // 2)
    #num_chromosomes = len(chrom_diagonals)
    
    vmax_heat = vmax
    if vmax_heat is not None:
        vmax_heat *= 5

    vmin_heat = vmin
    if vmin_heat is not None:
        vmin_heat *= 5
    else:
        vmin_heat = 0
        
    norm = None
    
    if(transform == 'log1p'):
        norm = LogNorm()
        vmin_heat = None
        vmax_heat = None
        
    #num_plots = len(chrom_diagonals)
    num_plots = 1
    fig = plt.figure(figsize=(num_plots * 4, 20))

    gs0 = gridspec.GridSpec(2, num_plots + 1, width_ratios=[10] * num_plots + [0.5], height_ratios=[1, 5],
                            wspace=0.1, hspace=0.1)

    gs_list = []
    #for idx, (chrom_name, values) in enumerate(chrom_diagonals.items()):
        #try:
            #heatmap = np.asarray(np.vstack(values))
        #except ValueError:
            #log.error("Error computing diagnostic heatmap for chrom: {}".format(chrom_name))
            #continue

    # get size of each cluster for the given chrom
    #clust_len = [(len(v)) for v in cluster_ids[chrom_name]]
    
    diagonales = np.zeros((len(submatrices),np.diagonal(submatrices[0]).shape[0]))
    
    for i in range(0,len(submatrices)):
        diagonales[i,:] = np.diagonal(submatrices[i])

    # prepare layout
    assert len(clusters) == len(submatrices)
    clusters = pd.Series(clusters)
    cluster_list = clusters.unique()
    cluster_list.sort()
    clusters = clusters.to_numpy()
    idx = 0
    clus_len = [len(clusters[clusters == c]) for c in cluster_list]
    
    gs_list.append(gridspec.GridSpecFromSubplotSpec(len(cluster_list), 1,
                                                    subplot_spec=gs0[1, idx],
                                                    height_ratios=clus_len,
                                                    hspace=0.03))
    summary_plot_ax = plt.subplot(gs0[0, idx])
    summary_plot_ax.set_title('heatmaps')

    for c in cluster_list:
        cluster_indices = clusters == c
        heatmap_to_plot = diagonales[cluster_indices,:]
        title = "cluster_{}".format(c)
        
        # sort by the value at the center of the rows
        order = np.argsort(heatmap_to_plot[:, M_half])[::-1]
        heatmap_to_plot = heatmap_to_plot[order, :]
        
        if(transform == 'log1p'):
            heatmap_to_plot = heatmap_to_plot + 1

        # add line to summary plot ax
        y_values = heatmap_to_plot.mean(axis=0)
        x_values = np.arange(len(y_values)) - M_half
        cluster_label = "cluster_{}".format(c)
        summary_plot_ax.plot(x_values, y_values, label=c)
        ax = plt.subplot(gs_list[-1][c, 0])
        ax.set_yticks([])
        #if num_chromosomes > 1:
        #    ax.set_ylabel(c)

        #if c < num_chromosomes - 1:
        ax.set_xticks([])

#        heat_fig = ax.pcolormesh(heatmap_to_plot,
#                            vmax=vmax_heat, vmin=vmin_heat,
#                            cmap=colormap, norm=norm)
        
        heat_fig = ax.imshow(heatmap_to_plot, aspect='auto',
                             interpolation='nearest',
                             cmap=colormap,
                             origin='upper', norm=norm,
                             vmax=vmax_heat, vmin=vmin_heat,
                             extent=[-M_half, M_half + 1,
                                     0, heatmap_to_plot.shape[0]])

    summary_plot_ax.legend(ncol=1, frameon=False, markerscale=0.5)

    cbar_x = plt.subplot(gs0[1, -1])
    fig.colorbar(heat_fig, cax=cbar_x, orientation='vertical')

    file_name = out_file_prefix + '_heatmap.png'
    log.info('Heatmap file saved under: {}'.format(file_name))
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
    plt.close()
    
def get_random_regions(amount,region_start,region_end,chromosome,region_size=1):
    '''create a bed file with random regions'''
    
    samples = np.array(random.sample(range(region_start,region_end),amount))
    cols = {'Chrom': pd.Series([], dtype='str'),
            'Start': pd.Series([], dtype='int'),
            'End': pd.Series([], dtype='int'),
            'UnknownCol1': pd.Series([], dtype='str'),
            'UnknownCol2': pd.Series([], dtype='str'),
            'Strand': pd.Series([], dtype='str')}
    
    random_bed_file = pd.DataFrame(cols)
    random_bed_file['Chrom'] = [chromosome] * amount
    random_bed_file['Start'] = samples
    random_bed_file['End'] = samples + region_size
    random_bed_file['UnknownCol1'] = np.zeros(samples.shape)
    random_bed_file['UnknownCol2'] = np.zeros(samples.shape)
    random_bed_file['Strand'] = '.'
    random_bed_file['RegionIndex'] = np.arange(0,random_bed_file.shape[0])
    
    return random_bed_file
    
def mix_with_random_regions(regions,region_start=None,region_end=None):
    '''introduce random regions to an input bed file'''
    
    n_rows = int(regions.shape[0]*1.1)
    region_size = int(np.median(regions['End'].to_numpy() - regions['Start'].to_numpy()))
    chromosome = regions['Chrom'][0]
    regions.sort_values(by=["Chrom", "Start"],inplace=True)
    
    if(region_start is None):
        region_start = regions['Start'][0]
    
    if(region_end is None):
        region_end = regions['Start'].to_numpy()[-1]
        
    #print(region_start)
    #print(region_end)
    #print(n_rows)
    #print(chromosome)
    
    random_bed_file = get_random_regions(n_rows,region_start,region_end,chromosome,region_size=region_size)
    
    #print(random_bed_file)
    random_bed_file['TestLabel'] = 1
    regions['TestLabel'] = 0
    regions = regions.append(random_bed_file)
    regions.sort_values(by=["Chrom", "Start"],inplace=True)
    regions.drop_duplicates(subset=["Chrom", "Start"],inplace=True)
    regions.set_index(np.arange(0,regions.shape[0]),inplace=True)
    regions['RegionIndex'] = np.arange(0,regions.shape[0])
    cols = get_regions_bed_col_names()
    cols.append('RegionIndex')
    return regions[cols],regions['TestLabel']

def dev_print_infos(pairs,clusters,dev_feature_type,dev_n_pairs,dev_n_regions,dev_evaluation):
    '''print clustering output'''
    
    if(dev_evaluation is None):
        print(clusters)
        return
        
    #indices_list = indices_list.copy(deep=True)
    print(clusters)
    print(pairs['Cluster'])
    
    #print(np.maximum(pairs['TestLabel'].to_numpy(),pairs['TestLabel_pair'].to_numpy()))
    if(dev_evaluation == 'test_against_random'):
        pairs['OutputTestLabel'] = np.maximum(pairs['TestLabel'].to_numpy(),pairs['pairTestLabel'].to_numpy())
    else:
        pairs['OutputTestLabel'] = pairs['TestLabel']
    
    assert pairs.shape[0] == dev_n_pairs
    
    if(dev_feature_type == 'per_region_flattened' or dev_feature_type == 'per_region_aggregated'):    
        #conditions = [pairs['Cluster'] == pairs['Cluster_pair']]
        #choices = [True,False]
        #pairs['hasSameCluster'] = np.select(conditions,choices,default=False)
        score = dev_evaluation_function(test_labels,clusters)
    
    else:
        score = dev_evaluation_function(pairs['Cluster'].to_numpy(),pairs['TestLabel'].to_numpy())
        
    print(score)
    print(pairs[['RegionIndex','Cluster','TestLabel']])
    pairs[['RegionIndex','Cluster','TestLabel']].to_csv('evaluation_labels', sep=';', index=False)

def dev_evaluation_function(test_labels,cluster_labels):
    '''compute evaluation score for clustering'''
    
    score = metrics.rand_score(test_labels,cluster_labels)
    return score

def dev_read_test_labels(test_label_file):
    '''get test labels for evaluation'''
    
    test_label_df = pd.read_csv(test_label_file, sep="\t", header=None)
    test_label_df.columns = ['TestLabel']
    
    if(test_label_df.size < 1):
        raise ValueError('empty domain file passed')
        
    return test_label_df['TestLabel']

#define parser and description

def parse_arguments(args=None):
    """
    get command line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        conflict_handler='resolve',
        description="""
        TODO
        
$ hicClusterContacts
        """)

   
    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--matrix', '-m',
                                help='HiC-Matrix file or list of files for input',
                                required=True,
                               type=str)
    
    parserRequired.add_argument('--bed',
                                help='regions to be clustered',
                                required=True,
                                type=str)    

    parserRequired.add_argument('--outFilePrefix',
                                '-o',
                                help='directory and prefix of output files',
                                required=True,
                                type=str)

    parserOpt = parser.add_argument_group('Optional arguments')

    parserOpt.add_argument('--transform',
                           help='set normalization method. Default: obs_exp',
                           type=str,
                           choices=[
                               'obs_exp',
                               'liebermann_aiden',
                               'non-zero'
                           ],
                           default='obs_exp')
    
    parserOpt.add_argument('--mode',
                           help='mode for intra- and interchromosomal contacts',
                           type=str,
                           choices=[
                               'intra',
                               'inter',
                               'all'
                           ],
                           default='intra')

    parserOpt.add_argument('--minRange',
                           help='only contacts within the given range are considered',
                           type=int,
                           default=1000000)
    
    parserOpt.add_argument('--maxRange',
                           help='only contacts within the given range are considered',
                           type=int,
                           default=20000000)

    parserOpt.add_argument('--submatrixCenterSize',
                           help='size of central feature square',
                           type=int,
                           default=0.2)
    
    parserOpt.add_argument('--useCompareToBorder',
                           help='compare to border of submatrices',
                           type=bool,
                           default=False)
    
    parserOpt.add_argument('--numberOfOutputClusters','-k',
                           help='number of output clusters',
                           type=int,
                           default=3)
    
    parserOpt.add_argument('--submatrixSize',
                           help='size of submatrices',
                           type=int,
                           default=25)
    
    parserOpt.add_argument('--clusterAlgorithm',
                           choices=[
                               'kmeans',
                               'agglomerative_hierarchical',
                               'gaussian_mixture',
                               'community_detection',
                               'hdbscan'
                           ],
                           help='cluster algorithm to use for computation',
                           type=str,
                           default='kmeans')

    parserOpt.add_argument('--threads', '-t',
                           help='number of threads used',
                           default=4,
                           type=int)
    
    parserOpt.add_argument('--devFeatureType',
                           choices=[
                               'per_region_aggregated',
                               'per_pair_flattened',
                               'per_region_flattened'
                           ],
                           default='per_pair_flattened',
                           help='for development: use features per region or per pair')
    
    parserOpt.add_argument('--scatterPlotType',
                           choices=[
                               '2d',
                               '3d'
                           ],
                           default='3d',
                           help='choose 2D or 3D scatter plot')
    
    parserOpt.add_argument('--vmin',
                           default=None,
                           help='set minimum value for output plot colormap')
    
    parserOpt.add_argument('--vmax',
                           default=None,
                           help='set maximum value for output plot colormap')
    
    parserOpt.add_argument('--colormap',
                           default='RdYlBu_r',
                           type=str,
                           help='set output plot colormap')
    
    parserOpt.add_argument('--plotAggrMode',
                           choices=[
                               'mean',
                               'median'
                           ],
                           default='mean',
                           help='whether plotted submatrices use mean or median aggregation')
    
    parserOpt.add_argument('--devPreprocessingType',
                           choices=[
                               'pca',
                               'umap',
                               None
                           ],
                           default=None,
                           help='choose clustering preprocessing')
    
    parserOpt.add_argument('--devNComponents',
                           default=20,
                           type=int,
                           help='number of components for pre-processing')
    
    parserOpt.add_argument('--devUmapNNeighbours',
                           default=20,
                           type=int,
                           help='umap hyperparameter')
    
    parserOpt.add_argument('--devUmapMetric',
                           choices=[
                            'euclidean',
                            'manhattan',
                            'chebyshev',
                            'minkowski',
                            'canberra',
                            'braycurtis',
                            'haversine',
                            'mahalanobis',
                            'wminkowski',
                            'seuclidean',
                            'cosine',
                            'correlation'
                           ],
                           default='minkowski',
                           help='number of components for pre-processing')
    
    parserOpt.add_argument('--devUmapMinDist',
                           default=0.5,
                           type=float,
                           help='umap hyperparameter')
    
    parserOpt.add_argument('--devOutlierCroppingMin',
                           default=None,
                           type=int,
                           help='min value for outlier cropping')
    
    parserOpt.add_argument('--devOutlierCroppingMax',
                           default=None,
                           type=int,
                           help='max value for outlier cropping')
    
    parserOpt.add_argument('--devEvaluation',
                           default=None,
                           type=str,
                           choices=[
                               'test_against_random',
                               'provide_test_labels',
                               None],
                           help='provide evaluation for the clustering')
          
    parserOpt.add_argument('--devTestLabels',
                           default=None,
                           type=str,
                           help='path to test labels')    
    
    parserOpt.add_argument('--randomSeed',
                           default=None,
                           type=int,
                           help='set random seed')
    
    parserOpt.add_argument('--transform',
                           default='log1p',
                           choices=[
                            None,
                            'log1p'
                           ],
                           help='Chooses whether to transform the submatrices before clustering')    

    parserOpt.add_argument("--help", "-h", action="help",
                           help="show this help message and exit")

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser
    
def main(args=None):
    
    #parse arguments
    args = parse_arguments().parse_args(args)
    matrix_file = args.matrix
    threads = args.threads
    bed_file = args.bed
    submatrix_size = args.submatrixSize
    min_distance = args.minRange
    max_distance = args.maxRange
    center_size = args.submatrixCenterSize
    compare_to_border = args.useCompareToBorder
    cluster_algorithm = args.clusterAlgorithm
    k = args.numberOfOutputClusters
    dev_feature_type = args.devFeatureType
    preprocessing_type = args.devPreprocessingType
    outlier_min = args.devOutlierCroppingMin
    outlier_max = args.devOutlierCroppingMax    
    
    out_file_contact_pairs = args.outFilePrefix + 'contact_pairs.bed'
    out_file_fig = args.outFilePrefix + 'scatter.png'
    out_file_prefix = args.outFilePrefix
    
    resolution = 1
    corner_position = 'upper_left'
    corner_size = 2
    
    if(not args.randomSeed is None):
        random.seed(args.randomSeed)
    
    #either by input, range normalization or for each submatrix itself
    vmin = args.vmin
    vmax = args.vmax
    colormap = args.colormap
    plot_aggr_mode = args.plotAggrMode
    scatter_plot_type = args.scatterPlotType
    
    #check for faulty parameters
    
    #ingest matrix file(s)
    log.info('reading matrix file')
    pChromosome = None
    matrix = read_matrix_file(matrix_file, pChromosome)
    
    log.info('reading bed file')
    #ingest bed file
    regions = read_regions_bed(bed_file)
    
    dev_n_regions = regions.shape[0]
    log.info('regions in bed file: ' + str(dev_n_regions))
    
    if(args.devEvaluation == 'test_against_random'):
        regions,test_labels = mix_with_random_regions(regions,region_start=None,region_end=None)
        
        assert dev_n_regions*2 <= regions.shape[0] and dev_n_regions*2.1 >= regions.shape[0]
        dev_n_regions = regions.shape[0]        
        log.info('random and non-random regions: ' + str(dev_n_regions))
    
    elif(args.devEvaluation == 'provide_test_labels'):
        assert not args.devTestLabels is None
        test_labels = dev_read_test_labels(args.devTestLabels)
          
    else:
        test_labels = np.zeros(regions.shape[0])
    
    regions['TestLabel'] = test_labels
    
    log.info('normalizing matrix file')
    #normalize matrix file(s)
    #matrix = obs_exp_normalization(matrix, pThreads=threads)
    
    log.info('calculating valid interaction pairs')
    #get pairs
    pairs = get_pairs(regions,min_distance,max_distance,resolution)
    
    dev_n_pairs = pairs.shape[0]        
    log.info('number of pairs: ' + str(dev_n_pairs))
    
    log.info('cutting out submatrices for interaction pairs')
    #cut out submatrices
    regions, pairs, submatrices = get_submatrices(matrix,regions,pairs,submatrix_size)

    assert dev_n_pairs >= pairs.shape[0]
    dev_n_pairs = pairs.shape[0]
    assert dev_n_pairs == len(submatrices)
    
    submatrices = outlier_cropping_and_transformation(submatrices,min_value=outlier_min,max_value=outlier_max)
    
    assert len(submatrices) == dev_n_pairs
    
    log.info('aggregating features for clustering')
    #aggregate features from submatrices
    features = get_features(submatrices,center_size=center_size,corner_position=corner_position,corner_size=corner_size)
    plot_density(features,out_file_prefix)
    
    assert len(features) == dev_n_pairs
    
    if(dev_feature_type == 'per_region_flattened' or dev_feature_type == 'per_region_aggregated'):
        features = get_feature_matrix(pairs,features,regions,dev_feature_type = dev_feature_type)
        
        assert pairs.shape[0] == dev_n_pairs
        assert features.shape[0] == dev_n_regions

        log.info('clustering')
        #cluster submatrices
        features = perform_clustering_preprocessing(features,preprocessing_type=preprocessing_type)
        clusters = perform_clustering(features,k,cluster_algorithm=cluster_algorithm)
        pairs['Cluster'] = region_to_pair_labels(pairs,clusters,'Cluster')
        
        assert features.shape[0] == dev_n_regions
        assert clusters.shape[0] == dev_n_regions
        assert clusters_per_interactions.shape[0] == dev_n_pairs
        
        log.info('writing results to file')
        output_results(out_file_contact_pairs,pairs)
        
    else:
        log.info('clustering')
        #cluster submatrices
        features = perform_clustering_preprocessing(features,preprocessing_type=preprocessing_type)
        clusters = perform_clustering(features,k,cluster_algorithm=cluster_algorithm)
        pairs['Cluster'] = clusters
        
        assert len(clusters) == dev_n_pairs
        assert features.shape[0] == dev_n_pairs
      
        log.info('writing results to file')
        output_results(out_file_contact_pairs,pairs)
    
    log.info('plotting results in scatter plot')
    plot_results(features,clusters,out_file_fig,scatter_plot_type = scatter_plot_type,preprocessing_type=preprocessing_type)
    
    log.info('plot submatrices')
    plot_submatrices(submatrices,pairs['Cluster'].to_numpy(),args.outFilePrefix + 'mean_submatrices.png',vmin=vmin,vmax=vmax,colormap=colormap,plot_aggr_mode=plot_aggr_mode)
    plot_diagnostic_heatmaps(submatrices,pairs['Cluster'].to_numpy(),args.outFilePrefix)
    
    #corr_matrix = compute_correlation(submatrices,clusters_per_interactions,out_file_name=args.outFilePrefix + 'correlation_scatter.png')
    #plot_correlation(corr_matrix,clusters_per_interactions,args.outFilePrefix + 'correlation_heatmap.png', vmax=None,vmin=None, image_format=None)
    
    dev_print_infos(pairs,clusters,dev_feature_type,dev_n_pairs,dev_n_regions,args.devEvaluation)
    
    log.info('Done')
