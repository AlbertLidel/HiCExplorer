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
from scipy.sparse import csr_matrix, lil_matrix, vstack
from bisect import bisect_right
from bisect import bisect_left
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import umap
import umap.plot
import hdbscan
from sklearn.mixture import GaussianMixture
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib import use as mplt_use
from matplotlib.ticker import FixedLocator
from scipy.stats import pearsonr, spearmanr
import random
import hicClusterContactsAutoEncoder as hccae
from os.path import exists

#import HiCExplorer
from hicmatrix import HiCMatrix as hm
from pybedtools import BedTool
from hicexplorer._version import __version__
from hicexplorer.utilities import obs_exp_matrix
from hicexplorer.utilities import convertNansToZeros, convertInfsToZeros

#get logger
mplt_use('Agg')
log = logging.getLogger(__name__)

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

    parserOpt.add_argument('--normalization',
                           help='set normalization method. Default: obs_exp',
                           type=str,
                           choices=[
                               'obs_exp',
                               'liebermann_aiden',
                               'non-zero'
                           ],
                           default='obs_exp')

    parserOpt.add_argument('--minRange',
                           help='only contacts within the given range are considered',
                           type=int,
                           default=1000000)
    
    parserOpt.add_argument('--maxRange',
                           help='only contacts within the given range are considered',
                           type=int,
                           default=3000000)

    parserOpt.add_argument('--submatrixCenterSize',
                           help='size of central feature square',
                           type=float,
                           default=0.2)
    
    parserOpt.add_argument('--cornerPosition',
                           help='which corner position to use for submatrix normalization',
                           type=str,
                           choices=[
                               'upper_left',
                               'upper_right',
                               'lower_left',
                               'lower_right'
                           ],
                           default=None)
    
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
    
    parserOpt.add_argument('--regionPositionType',
                           default=None,
                           type=str,
                           choices=[
                               'Start',
                               'End',
                               'Center',
                               None],
                           help='determine positioning type for regions')
    
    parserOpt.add_argument('--devPreprocessingType',
                           choices=[
                               'pca',
                               'umap',
                               'autoencoder',
                               'pca_umap',
                               None
                           ],
                           default='pca',
                           help='choose clustering preprocessing')
    
    parserOpt.add_argument('--devNComponents',
                           default=5,
                           type=int,
                           help='number of components for pre-processing')
    
    parserOpt.add_argument('--devUmapNNeighbours',
                           default=15,
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
                           default='euclidean',
                           help='number of components for pre-processing')
    
    parserOpt.add_argument('--devUmapMinDist',
                           default=0.1,
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
    
    parserOpt.add_argument('--nRandomRegions',
                           default=None,
                           type=int,
                           help='number of random regions for evaluation')    
          
    parserOpt.add_argument('--devTestLabels',
                           default=None,
                           type=str,
                           help='path to test labels')    
    
    parserOpt.add_argument('--randomSeed',
                           default=None,
                           type=int,
                           help='set random seed')
    
    parserOpt.add_argument('--transform',
                           default=None,
                           choices=[
                            None,
                            'log1p'
                           ],
                           help='Chooses whether to transform the submatrices before clustering')
    
    parserOpt.add_argument('--devCorrelationPlotType',
                           default=None,
                           type=str,
                           choices=[
                               'pearson',
                               'spearman',
                               None],
                           help='output correlation plot')
    
    parserOpt.add_argument('--devFeatureOutfile',
                           default=None,
                           type=str,
                           help='output file containing features')
    
    parserOpt.add_argument('--devAutoEncoderFile',
                           default=None,
                           type=str,
                           help='output file containing features')
    
    parserOpt.add_argument('--testBetweenTestLabels',
                           default='True',
                           choices=[
                               'True',
                               'False'
                           ],
                           type=str,
                           help='test')
    
    

    parserOpt.add_argument("--help", "-h", action="help",
                           help="show this help message and exit")

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser

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

def get_region_position_type(regions,args_region_position_type=None):
    '''select from input information, how to determine the exact position on matrix of the given regions'''
    
    assert args_region_position_type in [None,'Start','End','Center']
    
    if(not args_region_position_type is None):
        return region_position_type
    
    strand_type = regions['Strand'].mode()[0]
    
    if(strand_type == '-' or strand_type == '-1'):
        region_position_type = 'End'

    else:
        region_position_type = 'Start'
    
    return region_position_type

def get_pairs(regions,min_distance=1000000,max_distance=3000000,resolution=1):
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
            
            #build pair dataframe
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

def get_submatrices(matrix,regions,pairs,submatrix_size=9,position_type='Start',testBetweenTestLabels='True'):
    """collect submatrices for every pair of regions"""
    
    assert not matrix is None
    #get submatrices for these pairs
    submatrices = []
    submatrix_radius = math.floor(submatrix_size / 2)
    remove_rows = []
    assert position_type in ['Start','End','Center']
    
    for index,row in pairs.iterrows():       
        
        try:             
            pos = matrix.getRegionBinRange(row['Chrom'], row['Start'], row['End'])

            if(pos is None):
                raise ValueError
            else:
                i_start, i_end = pos
                
            pos = matrix.getRegionBinRange(row['pairChrom'], row['pairStart'], row['pairEnd'])
            
            if(pos is None):
                raise ValueError
            else:
                j_start, j_end = pos
                
            if(position_type == 'Start'):
                i = i_start
                j = j_start
                
            elif(position_type == 'End'):
                i = i_end
                j = j_end
            
            else:
                i = int((i_start + i_end) / 2)
                j = int((j_start + j_end) / 2)
                
            log.debug('position: ' + str(i) + ' ' + str(j) + ' in matrix: ' + str(matrix.matrix.shape[0]))
            
            row['MatrixIndex'] = i
            row['pairMatrixIndex'] = j

            #normal cases inside matrix
            if(testBetweenTestLabels == 'False' and row['TestLabel'] == 2):
                remove_rows.append(index)
            
            elif(i >= submatrix_size and j >= submatrix_size and i < matrix.matrix.shape[0] - submatrix_size and j < matrix.matrix.shape[0] - submatrix_size):

                up_i = i + submatrix_radius + 1
                lo_i = i - submatrix_radius
                up_j = j + submatrix_radius + 1
                lo_j = j - submatrix_radius
                submatrices.append(matrix.matrix[lo_i:up_i,lo_j:up_j])

            #TODO: cases at the border of the matrix, for which the submatrix crosses over the edge of the matrix
            else:
                remove_rows.append(index)
                log.warn('position of interaction pair at the edge of matrix')
            
        except ValueError:
            log.warn('position of interaction pair not found in matrix')
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
    
    regions = regions.drop(columns=['RegionIndexOld','pairRegionIndexOld','pairRegionIndex'])
    pairs = pairs.drop(columns=['RegionIndexOld','pairRegionIndexOld'])
    
    regions = regions.set_index(np.arange(0,regions.shape[0]))
    pairs = pairs.set_index(np.arange(0,pairs.shape[0]))
    
    return regions,pairs,submatrices
    
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
            f = s[lo_c:hi_c,lo_c:hi_c].toarray().flatten()

            if (not corner_position is None):
                if (corner_position == 'upper_left'):
                    corner = s[0:corner_size,0:corner_size].toarray().flatten()

                elif (corner_position == 'upper_right'):
                    corner = s[0:corner_size,-corner_size:].toarray().flatten()

                elif (corner_position == 'lower_right'):
                    corner = s[-corner_size:,-corner_size:].toarray().flatten()

                elif (corner_position == 'lower_left'):
                    corner = s[-corner_size:,0:corner_size].toarray().flatten()

                m = np.nanmean(corner)

                if(m > 0):
                    f = f / m

            return f
        
        else:
            return np.full(shape_,np.nan)
    
    for i in range(0,len(submatrices)):
        features[i,:] = build_features(submatrices[i])
        
    log.info('feature set size: ' + str(features.shape[1]))
    return features

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
            feature_matrix[j,i] = np.nanmean(features[index])
        
    return feature_matrix

def perform_clustering(features,k,cluster_algorithm='kmeans',random_state=0):
    '''perform cluster algorithm on data'''
    
    cluster_labels = None
    
    if(cluster_algorithm == 'kmeans'):
        clustering = skclust.KMeans(n_clusters=k, random_state=random_state).fit(features)
        cluster_labels = clustering.labels_
    
    elif(cluster_algorithm == 'agglomerative_hierarchical'):
        clustering = skclust.AgglomerativeClustering(n_clusters=k).fit(features)
        cluster_labels = clustering.labels_
        
    elif(cluster_algorithm == 'gaussian_mixture'):
        clustering = GaussianMixture(n_components=k, random_state=random_state).fit(features)
        cluster_labels = clustering.predict(features)
        
    elif(cluster_algorithm == 'hdbscan'):
        clustering = hdbscan.HDBSCAN(min_cluster_size=10, random_state=random_state)
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

    pairs['Strand'] = pairs['UnknownCol1']
    
    pairs_out = pairs[['Chrom','Start','End','UnknownCol1','UnknownCol2','Strand','pairChrom','pairStart','pairEnd','Cluster']]
    pairs_out[['Chrom','Start','End','Cluster','pairChrom','Strand','pairStart','pairEnd']].to_csv(out_file_contact_pairs, sep='\t', header=None, index=False)
    
def output_results_regions(out_file_regions,regions,clusters):
    '''output results to bed file'''
    
    regions['score'] = clusters
    regions[['Chrom','Start','End','score','Strand','Start','End']].to_csv(out_file_regions, sep='\t', header=None, index=False)

def perform_plotting_preprocessing(features,features_raw,n_components=3,preprocessing_type=None,umap_args=None):
    '''perform preprocessing for plotting'''
    
    if(preprocessing_type is 'pca'):
        reduced = features[:,0:n_components]
        embedder = None

    elif(preprocessing_type in ['umap','pca_umap']):
        if(umap_args is None):
            um = umap.UMAP(n_components=n_components, init='random', random_state=42)
        else:
            um = umap.UMAP(n_components=n_components, init='random', random_state=umap_args['random_state'],metric=umap_args['metric'], n_neighbors=umap_args['n_neighbors'], min_dist=umap_args['min_dist'])
            
        Sc = StandardScaler()
        scaled = Sc.fit_transform(features_raw)
        embedder = um.fit(scaled)
        reduced = embedder.embedding_
        
    else:
        Sc = StandardScaler()
        scaled = Sc.fit_transform(features)
        pca = PCA(n_components)
        pca.fit(scaled)
        reduced = pca.transform(scaled)
        embedder = None
    
    return embedder,reduced

def perform_clustering_preprocessing(features,preprocessing_type=None,n_components=20,encoder_file = None,random_state=0,umap_args=None):
    '''perform preprocessing for clustering'''
    
    assert features.shape[1] >= n_components

    Sc = StandardScaler()
    features = Sc.fit_transform(features)
    
    def perform_clustering_preprocessing_int(features, preprocessing_type=None, n_components=20, encoder_file = None,random_state=0,umap_args=None):
    
        if(preprocessing_type == 'umap'):
            if(umap_args is None):
                return umap.UMAP(n_components=n_components).fit_transform(features)                
            else:
                return umap.UMAP(n_components=n_components, metric=umap_args['metric'], n_neighbors=umap_args['n_neighbors'], min_dist=umap_args['min_dist'],random_state=umap_args['random_state']).fit_transform(features)

        if(preprocessing_type == 'pca'):
            return PCA(n_components=n_components).fit_transform(features)

        if(preprocessing_type == 'autoencoder'):
            assert not encoder_file is None
            return hccae.dimensionality_reduction(features,encoder_file)

        else:
            return features
        
    if(preprocessing_type == 'pca_umap'):
        if (features.shape[1] >= 50):
            pca_c = features.shape[1]
        else:
            pca_c = 50
            
        n_components = int(np.min(pca_c,n_components))
        features = perform_clustering_preprocessing_int(features, preprocessing_type='pca', n_components=pca_c,random_state=random_state)
        features = perform_clustering_preprocessing_int(features, preprocessing_type='umap', n_components=n_components, umap_args=umap_args,random_state=random_state)
        
    else:
        features = perform_clustering_preprocessing_int(features, preprocessing_type=preprocessing_type, n_components=n_components, umap_args=umap_args, encoder_file = encoder_file,random_state=random_state)
        
    return features
    
def plot_results(features,features_raw,clusters,out_file_prefix,scatter_plot_type='3d',preprocessing_type=None,title=None,umap_args=None):
    '''plot clustering results'''
    
    fig = plt.figure(dpi=150)
    
    if(scatter_plot_type == '3d'):
        ax = fig.add_subplot(projection=scatter_plot_type)
    else:
        ax = fig.add_subplot()        
    
    if(preprocessing_type in ['umap','pca_umap']):
        ax.set_xlabel('umap 1')
        ax.set_ylabel('umap 2')
    
    else:
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
    
    if(scatter_plot_type == '3d'):
        embedder,components = perform_plotting_preprocessing(features,features_raw,n_components=3,preprocessing_type=preprocessing_type,umap_args=umap_args)
        scatter = ax.scatter(components[:,0],components[:,1],components[:,2],c=clusters,cmap='Set3',alpha=0.8)
        
        if(preprocessing_type in ['umap','pca_umap']):
            ax.set_zlabel('umap 3')
        else:
            ax.set_zlabel('component 3')
        
    else:
        embedder,components = perform_plotting_preprocessing(features,features_raw,n_components=2,preprocessing_type=preprocessing_type,umap_args=umap_args)
        scatter = ax.scatter(components[:,0], components[:,1],c=clusters,cmap='Set3',alpha=0.8)
    
    legend1 = ax.legend(*scatter.legend_elements(),loc="upper left", title="")
    ax.add_artist(legend1)
    
    if(not title is None):
        ax.set_title(title)
    
    #fig.show()
    fig.savefig(out_file_prefix + '_scatter_plot.png')
    plt.close()
    
    if(preprocessing_type in ['umap','pca_umap']):
        embedder,components = perform_plotting_preprocessing(features, features_raw, n_components=2, preprocessing_type=preprocessing_type,umap_args=umap_args)      
        
        ax = umap.plot.points(embedder,labels=clusters)
        ax.figure.savefig(out_file_prefix + '_umap_points.png',dpi=300)
        plt.close()
        
        ax = umap.plot.connectivity(embedder, show_points=True)
        ax.figure.savefig(out_file_prefix + '_umap_connectivity.png',dpi=300)
        plt.close()
        
        ax = umap.plot.diagnostic(embedder, diagnostic_type='pca')
        ax.figure.savefig(out_file_prefix + '_umap_diagnostic.png',dpi=300)
        plt.close()        
    
def submatrix_list_to_matrix(submatrices):
    '''reshape list of matrices to a big matrix. Each row corresponds to one '''
    
    y,z = submatrices[0].shape
    flattened_submatrices = [m.tolil().reshape((1, y*z)) for m in submatrices]
    return vstack(flattened_submatrices).tocsr()

def submatrix_list_to_diagonal_array(submatrices):
    '''reshape list of matrices to a big matrix. Each row corresponds to one '''
    
    y,z = submatrices[0].shape
    diagonales = np.vstack([m.diagonal() for m in submatrices])
    
    return diagonales
    
def plot_submatrices(submatrices, clusters, out_file_name,vmin=None,vmax=None,colormap='RdYlBu_r',plot_aggr_mode='mean',transform=None):
    '''plot mean submatrices per cluster and for all regions'''
    
    assert len(clusters) == len(submatrices)
    clusters = pd.Series(clusters)
    cluster_list = clusters.unique()
    cluster_list.sort()
    aggr_submatrices = []
    titles = []
    norm = None
    
    if(transform == 'log1p'):
        norm = LogNorm()
        vmin_heat = None
        vmax_heat = None
    
    clusters = clusters.to_numpy()
    M_half = int((submatrices[0].shape[0] - 1) // 2)
    submatrix_shape = submatrices[0].shape
    submatrices = submatrix_list_to_matrix(submatrices)
    submatrices = [m.toarray() for m in submatrices]
    submatrices = np.array(submatrices)
    
    for c in cluster_list:
        cluster_indices = clusters == c
        cluster_submatrices = submatrices[cluster_indices]
        
        aggr_submatrix = cluster_submatrices.mean(axis=0)
        #aggr_submatrix = np.nanmean(cluster_submatrices,axis=0)
        aggr_submatrix = aggr_submatrix.reshape(submatrix_shape)
        aggr_submatrices.append(aggr_submatrix)
        titles.append(cluster_submatrices.shape[0])
    
    
    aggr_submatrix_all = submatrices.mean(axis=0)
    aggr_submatrix_all = aggr_submatrix_all.reshape(submatrix_shape)
    #aggr_submatrix_all = np.nanmean(submatrices,axis=0)
    title_all = submatrices.shape[0]
    
    assert len(aggr_submatrices) == len(cluster_list)
    
    fig = plt.figure(figsize=(5.5 * (len(cluster_list) + 1), 5.5))
    gs = gridspec.GridSpec(1,(len(cluster_list) + 1),wspace=0.1, hspace=0.1)

    gs.update(wspace=0.01, hspace=0.2)

    for cluster_number in range(0,len(aggr_submatrices)):
            title = "cluster_{}; number of submatrices: {}".format(cluster_number,titles[cluster_number])
            ax = plt.subplot(gs[0,cluster_number])
            ax.set_title(title)
            
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_vertical(size="5%", pad=0.3,pack_start=True)
            fig = ax.get_figure()
            fig.add_axes(ax_cb)
            
            o = aggr_submatrices[cluster_number]
            if(transform == 'log1p'):
                o = o+1
                
            img = ax.imshow(o, aspect='equal',interpolation='nearest',extent=[-M_half, M_half + 1, -M_half, M_half + 1],cmap = colormap,vmin=vmin,vmax=vmax,norm=norm)

            mappableObject = plt.cm.ScalarMappable(cmap = colormap)
            mappableObject.set_array(aggr_submatrices[cluster_number])
            plt.colorbar(mappableObject, cax = ax_cb,orientation='horizontal')
            ax_cb.xaxis.tick_bottom()
            ax_cb.xaxis.set_tick_params(labelbottom=True)
            

    
    title = 'all; number of submatrices: {}'.format(title_all)
    ax = plt.subplot(gs[0,len(cluster_list)])
    ax.set_title(title)
    
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_vertical(size="5%", pad=0.3,pack_start=True)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    o = aggr_submatrix_all
    if(transform == 'log1p'):
        o = o+1
        
    img = ax.imshow(o, aspect='equal',interpolation='nearest',extent=[-M_half, M_half + 1, -M_half, M_half + 1],cmap = colormap,vmin=vmin,vmax=vmax,norm=norm)
    mappableObject = plt.cm.ScalarMappable(cmap = colormap)
    mappableObject.set_array(aggr_submatrix_all)
    plt.colorbar(mappableObject, cax = ax_cb, orientation='horizontal')
    ax_cb.xaxis.tick_bottom()
    ax_cb.xaxis.set_tick_params(labelbottom=True)
    plt.savefig(out_file_name, dpi=300)
    #plt.show()
    plt.close()

def outlier_cropping_and_transformation(X,min_value=None,max_value=None,transform=None):
    '''crop statistical outliers and transform '''
    
    assert min_value is None or max_value is None or min_value <= max_value
    
    def oc_t_np(x):
        if(not max_value is None):
            np.minimum(x,max_value,out=x)

        if(not min_value is None):
            np.maximum(x,min_value,out=x)

        if(transform == 'log1p'):
            np.log1p(x,out=x)

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
    
    diagonales = submatrix_list_to_diagonal_array(submatrices)

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
        ax.set_xticks([])

        
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
    
def mix_with_random_regions(regions,region_start=None,region_end=None,n_random_regions=None):
    '''introduce random regions to an input bed file'''
    
    n_rows = int(regions.shape[0]*1.1)
    region_size = int(np.median(regions['End'].to_numpy() - regions['Start'].to_numpy()))
    chromosome = regions['Chrom'][0]
    regions.sort_values(by=["Chrom", "Start"],inplace=True)
    
    if(region_start is None):
        region_start = regions['Start'][0]
    
    if(region_end is None):
        region_end = regions['Start'].to_numpy()[-1]
        
    if(n_random_regions is None):
        n = n_rows
    else:
        n = n_random_regions
        
    random_bed_file = get_random_regions(n,region_start,region_end,chromosome,region_size=region_size)
    
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

def print_binary_pair_labels(pairs,clusters,dev_evaluation,prefix,features, features_raw, scatter_plot_type = '3d',preprocessing_type=None,umap_args=None):
    '''print clustering output'''
    
    if(dev_evaluation is None):
        print(pairs[['RegionIndex','Cluster']])
        return
        
    out_file_fig_test_labels = prefix + '_test_labels'
    score = dev_evaluation_function(pairs['TestLabel'],clusters)
    title = 'rand_score: ' + str(score)
    log.info(title)
    
    #print(pairs[['RegionIndex','Cluster','TestLabel']])
    pairs[['Chrom','Start','End','pairChrom','pairStart','pairEnd','Cluster','TestLabel']].to_csv(prefix + 'cluster_test_label_evaluation.csv', sep=';', index=False)
    
    plot_results(features,features_raw,pairs['TestLabel'],out_file_fig_test_labels,scatter_plot_type = scatter_plot_type,preprocessing_type=preprocessing_type,title=None,umap_args=umap_args)
    
def print_infos_per_regions(regions,clusters,dev_evaluation,prefix,features,features_raw, scatter_plot_type = '3d',preprocessing_type=None,umap_args=None):
    '''print clustering output'''
    
    if(dev_evaluation is None):
        print(regions[['RegionIndex','Cluster']])
        return
        
    out_file_fig_test_labels = prefix + '_test_labels'
    score = dev_evaluation_function(regions['TestLabel'],clusters)
    title = 'rand_score: ' + str(score)
    
    print(regions[['RegionIndex','Cluster','TestLabel']])
    
    score_matrix = regions[['Cluster','TestLabel']].reset_index().groupby(['Cluster','TestLabel'],as_index=False).size().pivot('TestLabel','Cluster')
    
    tfile = open(prefix + '_score_matrix.txt', 'w')
    tfile.write(title + '\n')
    tfile.write(score_matrix.to_string())
    tfile.close()
    
    plot_results(features,features_raw,regions['TestLabel'],out_file_fig_test_labels,scatter_plot_type= scatter_plot_type, preprocessing_type= preprocessing_type,title=None,umap_args=umap_args)

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

def compute_correlation(submatrices,clusters_per_interaction,correlation_type='pearson',out_file_name=None):
    '''compute correlation coefficient between the given subrmatrices'''
    #code taken and adapted from hicCorrelate
    
    def get_vectors(mat1, mat2):
        """
        Uses sparse matrix tricks to convert
        into a vector the matrix values such
        that zero values that appear in only
        one of the matrices is kept. But
        zeros in two matrices are removed

        Requires two sparse matrices as input
        """
        assert mat1.shape == mat2.shape, "Matrices have different shapes. "\
            "Computation of correlation is not possible."

        # create a new matrix that is the sum of the two
        # matrices to compare. The goal is to have
        # a matrix that contains all the positions
        # that are non-zero in both matrices
        _mat = mat1 + mat2

        # add one to each element in the new matrix
        _mat.data += 1

        # get a vector of the values in mat1 from
        # _mat
        values1 = (_mat - mat1) - 1

        # get a vector of the values in mat2 from
        # _mat
        values2 = (_mat - mat2) - 1

        return values1, values2
        
    assert len(submatrices) == len(clusters_per_interaction)
    matrices = submatrices
    labels = clusters_per_interaction
    num_files = len(matrices)
    
    #map(lambda x: os.path.basename(x), args.matrices)
    # initialize results matrix
    results = np.zeros((num_files, num_files), dtype='float')

    rows, cols = np.triu_indices(num_files)
    correlation_opts = {'spearman': spearmanr,
                        'pearson': pearsonr}
    hic_mat_list = []
    max_value = None
    min_value = None
    all_mat = None
    all_nan = []

    for i, _mat in enumerate(matrices):

        np.diagflat(_mat,0)
        _mat = np.triu(_mat)
        
        if all_mat is None:
            all_mat = _mat
        else:
            all_mat = all_mat + _mat

        if max_value is None or max_value < np.max(_mat):
            max_value = np.max(_mat)
        if min_value is None or min_value > np.min(_mat):
            min_value = np.min(_mat)

        hic_mat_list.append(_mat)

    big_mat = None
    
    for mat in hic_mat_list:
        #mat = mat[rows_keep, :][:, cols_keep]
        sample_vector = mat.flatten()
        
        if big_mat is None:
            big_mat = sample_vector
        else:
            big_mat = np.vstack([big_mat, sample_vector])

    # take the transpose such that columns represent each of the samples
    big_mat = np.ma.masked_invalid(big_mat).T

    grids = gridspec.GridSpec(num_files, num_files)
    grids.update(wspace=0, hspace=0)
    fig = plt.figure(figsize=(2 * num_files, 2 * num_files))
    plt.rcParams['font.size'] = 8.0

    min_value = int(np.min(big_mat))
    max_value = int(np.max(big_mat))
    
    if (min_value % 2 == 0 and max_value % 2 == 0) or \
            (min_value % 1 == 0 and max_value % 2 == 1):
        # make one value odd and the other even
        max_value += 1

    for index in range(len(rows)):
        row = rows[index]
        col = cols[index]
        if row == col:
            results[row, col] = 1

            # add titles as
            # empty plot in the diagonal
            ax = fig.add_subplot(grids[row, col])
            ax.text(0.6, 0.6, 'scatter plot',
                    verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=10, fontweight='bold',
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        #log.debug("comparing {} and {}\n".format(args.matrices[row],args.matrices[col]))

        # remove cases in which both are zero or one is zero and
        # the other is one
        _mat = big_mat[:, [row, col]]
        _mat = _mat[_mat.sum(axis=1) > 1, :]
        vector1 = _mat[:, 0]
        vector2 = _mat[:, 1]

        results[row, col] = correlation_opts[correlation_type](vector1, vector2)[0]

        # scatter plots
        ax = fig.add_subplot(grids[row, col])

        ax.text(0.2, 0.8, "{}={:.2f}".format(correlation_type,
                                             results[row, col]),
                horizontalalignment='left',
                transform=ax.transAxes)
        ax.get_yaxis().set_tick_params(
            which='both',
            left='off',
            right='off',
            direction='out')

        ax.get_xaxis().set_tick_params(
            which='both',
            top='off',
            bottom='off',
            direction='out')

        if col != num_files - 1:
            ax.set_yticklabels([])
        else:
            ax.yaxis.tick_right()
            ax.get_yaxis().set_tick_params(
                which='both',
                left='off',
                right='on',
                direction='out')
        if col - row == 1:
            ax.xaxis.tick_bottom()
            ax.get_xaxis().set_tick_params(
                which='both',
                top='off',
                bottom='on',
                direction='out')
        else:
            ax.set_xticklabels([])

        ax.hist2d(vector1, vector2, bins=150, cmin=0.1)
        
    fig.tight_layout()
    
    if(not out_file_name is None):
        log.debug("saving {}".format(out_file_name))
        fig.savefig(out_file_name, bbox_inches='tight')
        
    plt.close()
    return results + np.triu(results, 1).T
    
def plot_correlation(corr_matrix, labels, out_file_name, vmax=None,
                     vmin=None, colormap='Reds', image_format='png'):
    '''plot the given correlation matrix'''
    
    #code taken and adapted from hicCorrelate
    import scipy.cluster.hierarchy as sch
    num_rows = corr_matrix.shape[0]

    # set the minimum and maximum values
    if vmax is None:
        vmax = 1
    if vmin is None:
        vmin = 0 if corr_matrix.min() >= 0 else -1

    # Compute and plot dendrogram.
    fig = plt.figure(figsize=(10.5, 9.5))
    axdendro = fig.add_axes([0.02, 0.1, 0.1, 0.7])
    axdendro.set_axis_off()
    y_var = sch.linkage(corr_matrix, method='complete')
    z_var = sch.dendrogram(y_var, orientation='left',
                           link_color_func=lambda k: 'black')
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    cmap = plt.get_cmap(colormap)

    # this line simply makes a new cmap, based on the original
    # colormap that goes from 0.0 to 0.9
    # This is done to avoid colors that
    # are too dark at the end of the range that do not offer
    # a good contrast between the correlation numbers that are
    # plotted on black.
    cmap = cmap.from_list(colormap + "clipped", cmap([0.0, 0.8]))
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.13, 0.1, 0.6, 0.7])
    index = z_var['leaves']
    corr_matrix = corr_matrix[index, :]
    corr_matrix = corr_matrix[:, index]
    img_mat = axmatrix.matshow(corr_matrix, aspect='equal', origin='lower',
                               cmap=cmap, extent=(0, num_rows, 0, num_rows),
                               vmax=vmax, vmin=vmin)
    axmatrix.yaxis.tick_right()
    axmatrix.set_yticks(np.arange(corr_matrix.shape[0]) + 0.5)
    axmatrix.set_yticklabels(np.array(labels).astype('str')[index],
                             fontsize=14)

    axmatrix.set_xticks(np.arange(corr_matrix.shape[0]) + 0.5)
    axmatrix.set_xticklabels(np.array(labels).astype('str')[index],
                             fontsize=14,
                             rotation=45,
                             ha='left')

#    axmatrix.set_xticks([])
    # Plot colorbar.
    axcolor = fig.add_axes([0.13, 0.065, 0.6, 0.02])
    plt.colorbar(img_mat, cax=axcolor, orientation='horizontal')
    for row in range(num_rows):
        for col in range(num_rows):
            axmatrix.text(row + 0.5, col + 0.5,
                          "{:.2f}".format(corr_matrix[row, col]),
                          ha='center', va='center')

    fig.savefig(plot_filename, format=image_format)
    plt.close()
    
def plot_feature_matrix(feature_matrix,out_file_prefix,evaluation_labels=None,transform='log1p'):
    '''plot sorted feature matrix'''
    
    evaluation_labels = evaluation_labels.to_numpy(copy=True)
    feature_matrix = feature_matrix[evaluation_labels.argsort(),:]
    feature_matrix = feature_matrix[:,evaluation_labels.argsort()]    
    
    fig = plt.figure(figsize =(10, 7))
    #ax = fig.add_axes([0,0,1,1])
    
    title = 'sorted feature matrix'
    
    if(transform == 'log1p'):
        feature_matrix = np.log1p(feature_matrix)
    
    plt.imshow(feature_matrix, cmap='RdYlBu_r')
    plt.title(title)
    plt.savefig(out_file_prefix + '_sorted_feature_matrix.png', dpi=300)
    plt.close()
    
def region_to_pair_labels_binary(pairs,labels):
    '''merge region labels to pairs'''
    
    cl_df = pd.DataFrame(columns=['RegionLabel'])
    cl_df['RegionLabel'] = labels
    cl_df['RegionIndex'] = np.arange(0,labels.shape[0])      
    o = pd.merge(pairs[['RegionIndex','pairRegionIndex']], cl_df, how='inner', left_on=['RegionIndex'], right_on=['RegionIndex'])
    
    cl_df['pairRegionIndex'] = cl_df['RegionIndex']
    cl_df['pairRegionLabel'] = cl_df['RegionLabel']
    cl_df = cl_df[['pairRegionIndex','pairRegionLabel']]
    o = pd.merge(o, cl_df, how='inner', left_on=['pairRegionIndex'], right_on=['pairRegionIndex'])
    o['label'] = 0
    
    def assign_labels(row):
        if(row['RegionLabel'] == 0 and row['pairRegionLabel'] == 0):
            row['label'] = 0
            
        elif(row['RegionLabel'] == 1 and row['pairRegionLabel'] == 1):
            row['label'] = 1
        
        else:
            row['label'] = 2
    
    o.apply(lambda row: assign_labels(row), axis=1)
    return o['label']

#def region_to_pair_labels_plots(pairs,labels):
#    '''merge region labels to pairs'''
    
#    cl_df = pd.DataFrame(columns=['RegionLabel'])
#    cl_df['RegionLabel'] = labels
#    cl_df['RegionIndex'] = np.arange(0,labels.shape[0])      
#    o = pd.merge(pairs[['RegionIndex']], cl_df, how='inner', left_on=['RegionIndex'], right_on=['RegionIndex'])
    
#    return o['RegionLabel'].to_numpy()

def omitInterLabelPairs(pairs,submatrices,features):
    
    valid_indices = pairs['TestLabel'].to_numpy() != 2    
    features = features[valid_indices,:]
    submatrices = [submatrices[i] for i in valid_indices]
    pairs = pairs.iloc[valid_indices]
    pairs = pairs.set_index(np.arange(0,pairs.shape[0]))
    
    return pairs,submatrices,features

def cluster_occurence_per_region(pairs,out_file_prefix,regions,dev_evaluation=None,random_state=0,umap_args=None):
    '''if dataset is clustered per pair, count occurences of clusters per region and build heatmap'''
    
    occurences = pairs[['Chrom','Start','End','Cluster']]
    occurences_add = pairs[['pairChrom','pairStart','pairEnd','Cluster']]
    occurences_add = occurences_add.rename(columns={'pairChrom':'Chrom','pairStart':'Start','pairEnd':'End','Cluster':'Cluster'})
    occurences = occurences.append(occurences_add)
    occurences = occurences.groupby(['Chrom','Start','End','Cluster'],as_index = False).size().pivot(['Chrom','Start','End'],'Cluster').fillna(0)
    occurences_array = occurences.to_numpy()
    
    occurences_array = occurences_array / np.sum(occurences_array,axis=0)[np.newaxis,:]
    print(occurences_array.shape)
    
    if(not dev_evaluation is None):
        rlabels = regions[['Chrom','Start','End','TestLabel']]
        #o_i = occurences.index.to_frame()[['Chrom','Start','End']]
        rlabels = pd.merge(occurences, rlabels, how='inner', left_index=True, right_on=['Chrom','Start','End'])
        rlabels_ = rlabels['TestLabel']
        label_dict = dict(zip(rlabels_.unique(), "rbg"))
        row_colors = rlabels_.map(label_dict)
        row_colors = row_colors.to_numpy()
        
        score_matrix = rlabels.drop(columns=['Chrom','Start','End']).groupby(['TestLabel'],as_index = False).sum()
        score_matrix.to_csv(out_file_prefix + '_occurences_count_per_test_label.txt')
        occurences = rlabels
        
    else:
        row_colors=None
        
    #occurences_array = np.rot90(occurences_array,k=3)
    #, left_on=[['Chrom','Start','End']]
    grid = sns.clustermap(occurences_array,row_colors=row_colors)
    #grid.set_title('regions vs cluster occurences')
    grid.savefig(out_file_prefix + '_cluster_occurences.png')
    plt.close()
    
    grid = sns.clustermap(occurences_array,row_colors=row_colors,row_cluster=False,col_cluster=False)
    grid.savefig(out_file_prefix + '_sorted_occurences.png')
    plt.close()

    Sc = StandardScaler()
    scaled = Sc.fit_transform(occurences_array)
    comp = PCA(n_components=2).fit_transform(scaled)
    
    clustering = skclust.AgglomerativeClustering(n_clusters=occurences_array.shape[1]).fit(comp)
    cluster_labels = clustering.labels_

    plot_results(occurences_array,occurences_array,cluster_labels,out_file_prefix + '_region_occurence_clustering',scatter_plot_type = '2d',preprocessing_type=None,umap_args=umap_args)
    regions['Cluster'] = cluster_labels
    output_results_regions(out_file_prefix + '_region_occurence_clustering.bed',regions,cluster_labels)
    
    print_infos_per_regions(regions,cluster_labels,dev_evaluation,out_file_prefix + '_region_occurence_clustering',occurences_array,occurences_array,scatter_plot_type = '2d',preprocessing_type=None,umap_args=umap_args)
    
    occurences['Cluster'] = cluster_labels
    occurences.to_csv(out_file_prefix + '_region_occurence_clustering.csv', sep=';', index=True)
    score_matrix = occurences.reset_index().groupby(['Cluster'],as_index = False).sum()
    score_matrix.reset_index().to_csv(out_file_prefix + '_occurences_count_per_region_occurence_cluster.txt',index=False,header=False)
    
def output_region_cluster_position_heatmap(clusters,out_file_prefix,test_labels=None):
    '''print region cluster position overview'''
    
    if(not test_labels is None):
        h = np.hstack((clusters.to_numpy()[:,np.newaxis],test_labels.to_numpy()[:,np.newaxis]))
    else:
        h = clusters.to_numpy()
    
    grid = sns.clustermap(h,row_cluster=False,col_cluster=False)
    grid.savefig(out_file_prefix + '_region_cluster_positions.png')
    plt.close()    
    
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
    cluster_algorithm = args.clusterAlgorithm
    k = args.numberOfOutputClusters
    dev_feature_type = args.devFeatureType
    preprocessing_type = args.devPreprocessingType
    outlier_min = args.devOutlierCroppingMin
    outlier_max = args.devOutlierCroppingMax
    corner_position = args.cornerPosition
    print(center_size)
    
    if(args.transform == None):
        plot_transform = 'log1p'
        
    resolution = 1
    corner_size = 5
    
    if(not args.randomSeed is None):
        random.seed(args.randomSeed)
    
    #either by input, range normalization or for each submatrix itself
    vmin = args.vmin
    vmax = args.vmax
    colormap = args.colormap
    plot_aggr_mode = args.plotAggrMode
    scatter_plot_type = args.scatterPlotType
    
    umap_args = {}
    umap_args['random_state'] = args.randomSeed
    umap_args['min_dist'] = args.devUmapMinDist
    umap_args['n_neighbors'] = args.devUmapNNeighbours
    umap_args['metric'] = args.devUmapMetric
    #check for faulty parameters
    
    #ingest matrix file(s)
    log.info('reading matrix file')
    pChromosome = None
    matrix = read_matrix_file(matrix_file, pChromosome)
    
    log.info('reading bed file')
    #ingest bed file
    all_regions = read_regions_bed(bed_file)
    chromosomes = pd.unique(all_regions['Chrom'])
    log.info('regions contain chromosomes: ' + str(chromosomes))
    assert len(chromosomes) > 0
    region_position_type = get_region_position_type(all_regions,args_region_position_type=args.regionPositionType)
    log.info('using positioning type: ' + region_position_type)
    
    dev_n_regions = all_regions.shape[0]
    log.info('regions in bed file: ' + str(dev_n_regions))
    
    if(args.devEvaluation == 'provide_test_labels'):
        assert not args.devTestLabels is None
        test_labels = dev_read_test_labels(args.devTestLabels)
          
    else:
        test_labels = np.zeros(all_regions.shape[0])
    
    all_regions['TestLabel'] = test_labels
    
    log.info('normalizing matrix file')
    #normalize matrix file(s)
    #matrix = obs_exp_normalization(matrix, pThreads=threads)
    
    for c in chromosomes:
        log.info('running for chromosome ' + str(c))
        regions = all_regions[all_regions['Chrom'] == c]
        log.info('numbers of regions on chromosome: ' + str(regions.shape[0]))
        
        if(args.devEvaluation == 'test_against_random'):
            regions,test_labels = mix_with_random_regions(regions, region_start=None, region_end=None, n_random_regions=args.nRandomRegions)

            log.info('random and non-random regions: ' + str(regions.shape[0]))         
            regions['TestLabel'] = test_labels
        
        out_file_contact_pairs = args.outFilePrefix + '_' + str(c) + '_contact_pairs.bed'
        out_file_fig = args.outFilePrefix + '_' + str(c)
        out_file_fig_test_labels = args.outFilePrefix + '_' + str(c)       
        out_file_prefix = args.outFilePrefix + '_' + str(c)
        
    
        log.info('calculating valid interaction pairs')
        #get pairs
        pairs = get_pairs(regions,min_distance,max_distance,resolution)

        log.info('number of pairs on chromosome: ' + str(pairs.shape[0]))

        log.info('cutting out submatrices for interaction pairs')
        pairs['TestLabel'] = region_to_pair_labels_binary(pairs,regions['TestLabel'])
        
        #cut out submatrices
        regions, pairs, submatrices = get_submatrices(matrix,regions,pairs,submatrix_size,position_type=region_position_type, testBetweenTestLabels=args.testBetweenTestLabels)

        log.info('valid submatrices: ' + str(len(submatrices)))
        log.info('aggregating features for clustering')
        #aggregate features from submatrices
        features = get_features(submatrices,center_size=center_size,corner_position=corner_position,corner_size=corner_size)
        features = outlier_cropping_and_transformation(features,min_value=outlier_min,max_value=outlier_max,transform=args.transform)
        
        if(not args.devFeatureOutfile is None):
            np.savetxt(args.devFeatureOutfile,features)
        
        plot_density(features,out_file_prefix)

        assert not preprocessing_type == 'autoencoder' or (not args.devAutoEncoderFile is None and dev_feature_type == 'per_pair_flattened')

        if(dev_feature_type == 'per_region_flattened' or dev_feature_type == 'per_region_aggregated'):
            features = get_feature_matrix(pairs,features,regions,dev_feature_type = dev_feature_type)
            feature_matrix = features
            features_raw = features
            
            log.info('clustering')
            #cluster submatrices
            features = perform_clustering_preprocessing(features,preprocessing_type=preprocessing_type, n_components= args.devNComponents,random_state=args.randomSeed,umap_args=umap_args)
            clusters = perform_clustering(features,k,cluster_algorithm=cluster_algorithm,random_state=args.randomSeed)
            regions['Cluster'] = clusters

            log.info('writing results to file')
            output_region_cluster_position_heatmap(regions['Cluster'],out_file_prefix,test_labels=regions['TestLabel'])   
            output_results_regions(out_file_contact_pairs,regions,clusters)
            print_infos_per_regions(regions, clusters, args.devEvaluation, out_file_prefix, features, features_raw, scatter_plot_type = scatter_plot_type, preprocessing_type=preprocessing_type,umap_args=umap_args)
            
            if('TestLabel' in regions.columns):
                regions[['Chrom','Start','End','Cluster','TestLabel']].to_csv(out_file_prefix + 'cluster_test_label_evaluation.csv',sep=';')
                
            if(len(regions['Cluster'].unique()) == 2):
                pairs['Cluster'] = region_to_pair_labels_binary(pairs,clusters)
                plot_submatrices(submatrices,pairs['Cluster'].to_numpy(),out_file_prefix + 'mean_submatrices.png',vmin=vmin,vmax=vmax,colormap=colormap,plot_aggr_mode=plot_aggr_mode,transform=plot_transform)
                plot_diagnostic_heatmaps(submatrices,pairs['Cluster'].to_numpy(),out_file_prefix,transform=plot_transform)
                
            if(dev_feature_type == 'per_region_aggregated' and 'TestLabel' in regions.columns):
                log.info('plotting feature matrix')
                plot_feature_matrix(feature_matrix, out_file_prefix, evaluation_labels=regions['TestLabel'], transform=plot_transform)
                
            log.info('plotting results in scatter plot')
            plot_results(features,features_raw,clusters,out_file_fig,scatter_plot_type = scatter_plot_type,preprocessing_type=preprocessing_type,umap_args=umap_args)             

        else:
            log.info('clustering')
            
            features_raw = features
            #cluster submatrices
            features = perform_clustering_preprocessing(features,preprocessing_type=preprocessing_type,encoder_file = args.devAutoEncoderFile,n_components= args.devNComponents,random_state=args.randomSeed,umap_args=umap_args)
            clusters = perform_clustering(features,k,cluster_algorithm=cluster_algorithm,random_state=args.randomSeed)
            pairs['Cluster'] = clusters

            log.info('writing results to file')
            output_results(out_file_contact_pairs,pairs)
            
            log.info('plot submatrices')
            plot_diagnostic_heatmaps(submatrices,pairs['Cluster'].to_numpy(),out_file_prefix,transform=plot_transform)
            plot_submatrices(submatrices,pairs['Cluster'].to_numpy(),out_file_prefix + 'mean_submatrices.png',vmin=vmin,vmax=vmax,colormap=colormap,plot_aggr_mode=plot_aggr_mode,transform=plot_transform)               
            log.info('plotting results in scatter plot')
            plot_results(features,features_raw,clusters,out_file_fig,scatter_plot_type = scatter_plot_type,preprocessing_type=preprocessing_type,umap_args=umap_args)
            
            cluster_occurence_per_region(pairs,out_file_prefix,regions,args.devEvaluation,random_state=args.randomSeed)
            print_binary_pair_labels(pairs, clusters, args.devEvaluation, out_file_prefix, features, features_raw, scatter_plot_type = scatter_plot_type, preprocessing_type=preprocessing_type,umap_args=umap_args)
     

        
#        if(not args.devCorrelationPlotType is None):
#            log.info('computing correlation matrix')
#            corr_matrix = compute_correlation(submatrices,pairs['Cluster'],out_file_name=args.outFilePrefix + 'correlation_scatter.png')
#            log.info('plotting correlation heatmap')
#            plot_correlation(corr_matrix,pairs['Cluster'],args.outFilePrefix + 'correlation_heatmap.png', vmax=None,vmin=None, image_format=None)
    
    log.info('Done')
