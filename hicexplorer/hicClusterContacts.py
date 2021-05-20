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
from hicexplorer.utilities import obs_exp_matrix
from hicexplorer.utilities import convertNansToZeros, convertInfsToZeros
from bisect import bisect_right
from bisect import bisect_left
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

#import HiCExplorer
from hicmatrix import HiCMatrix as hm
from pybedtools import BedTool
from hicexplorer._version import __version__

#get logger
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
    bed_df.columns = get_regions_bed_col_names()
    bed_df = bed_df.sort_values(by=["Chrom", "Start"])

    if(bed_df.size < 1):
        raise ValueError('empty domain file passed')
        
    bed_df.set_index(np.arange(0,bed_df.shape[0]),inplace=True)

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
            pairs.append(current)            
            
    pairs = pd.concat(pairs)
    
    return pairs

def get_submatrices(matrix,pairs,submatrix_size=9):
    """collect submatrices for every pair of regions"""
    
    #get submatrices for these pairs
    submatrices = []
    #pos_dict = build_position_index(get_positions(matrix))
    pos_dict = build_position_index_bisect(get_positions(matrix))
    chromosomes = pairs['Chrom'].unique()
    chr_pos = dict(zip(chromosomes,map(matrix.getChrBinRange,chromosomes)))
    submatrix_radius = math.floor(submatrix_size / 2)
    pairs['Index'] = np.zeros((len(pairs)), dtype='int')
    pairs['pairIndex'] = np.zeros((len(pairs)), dtype='int')
    
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
            
            row['Index'] = i
            row['pairIndex'] = j

            #normal cases inside matrix
            if(i >= submatrix_size and j >= submatrix_size and i < matrix.matrix.shape[0] - submatrix_size and j < matrix.matrix.shape[0] - submatrix_size):

                up_i = i + submatrix_radius + 1
                lo_i = i - submatrix_radius
                up_j = j + submatrix_radius + 1
                lo_j = j - submatrix_radius
                submatrices.append(matrix.matrix[lo_i:up_i,lo_j:up_j].toarray())

            #TODO: cases at the border of the matrix, for which the submatrix crosses over the edge of the matrix
            else:
                submatrices.append(None)
            
        except ValueError:
            warnings.warn('position of interaction pair not found in matrix')
            print('warn')
            row['Index'] = -1
            row['pairIndex'] = -1
            submatrices.append(None)
            
    return pairs,submatrices
        
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
            return np.zeros(shape_)
    
    for i in range(0,len(submatrices)):
        features[i,:] = build_features(submatrices[i])
        
    return features

def get_feature_matrix(pairs,features):
    
    
    pair_positions = pairs[['pairChrom','pairStart','pairEnd']].rename(columns = {'pairChrom':'Chrom', 'pairStart': 'Start', 'pairEnd': 'End'}, inplace = False)
    indices_list = pd.concat([pairs[['Chrom','Start','End']].copy(),pair_positions])
    indices_list = indices_list.drop_duplicates(subset=['Chrom','Start']).sort_values(by=["Chrom","Start"])
    indices_list['featureIndex'] = np.arange(len(indices_list['Chrom']))
    indices_list.set_index(np.arange(0,len(indices_list)))

    pairs = pd.merge(pairs, indices_list, how='inner', left_on=['Chrom','Start','End'], right_on=['Chrom','Start','End'])
    pairs = pd.merge(pairs, indices_list, how='inner', left_on=['pairChrom','pairStart','pairEnd'], right_on=['Chrom','Start','End'])
    
    pairs = pairs[['Chrom_x','Start_x','End_x','UnknownCol1','UnknownCol2','Strand','pairChrom','pairStart','pairEnd','Index','pairIndex','featureIndex_x','featureIndex_y']]
    pairs = pairs.rename(columns = {'Chrom_x':'Chrom', 'Start_x': 'Start', 'End_x': 'End', 'featureIndex_x': 'featureIndex', 'featureIndex_y': 'pairFeatureIndex'}, inplace = False)
    pairs.set_index(np.arange(0,len(pairs)))
    
    feature_matrix = np.zeros((len(indices_list),len(indices_list)*len(features[0])))
    
    for index,row in pairs.iterrows():
        
        i = row['featureIndex']
        j = row['pairFeatureIndex']
        i_f = row['featureIndex'] * len(features[0])
        j_f = row['pairFeatureIndex'] * len(features[0])
        i_f1 = (row['featureIndex'] + 1) * len(features[0])
        j_f1 = (row['pairFeatureIndex'] + 1) * len(features[0])
        
        feature_matrix[i,j_f:j_f1] = features[index]
        feature_matrix[j,i_f:i_f1] = features[index]
        
    return pairs, indices_list, feature_matrix    

def perform_clustering(features,k,cluster_algorithm=None):
    '''perform cluster algorithm on data'''
    
    clustering = skclust.KMeans(n_clusters=k, random_state=0).fit(features)
    cluster_labels = clustering.labels_
    return cluster_labels

def get_regions_bed_col_names():
    '''get column names'''

    return ['Chrom', 'Start', 'End', 'UnknownCol1', 'UnknownCol2', 'Strand']

def output_results(out_file_contact_pairs,pairs,indices_list,clusters):
    '''output results to bed file'''
    
    indices_list['Cluster'] = clusters
    pairs = pd.merge(pairs, indices_list, how='inner', left_on=['Chrom','Start','End'], right_on=['Chrom','Start','End'])
    pairs = pairs[['Chrom','Start','End','UnknownCol1','UnknownCol2','Strand','pairChrom','pairStart','pairEnd','Cluster']]
    pairs.to_csv(out_file_contact_pairs, sep='\t', header=None, index=False)
    
def output_results_alt(out_file_contact_pairs,pairs,clusters):
    '''output results to bed file'''

    pairs['Cluster'] = clusters
    pairs = pairs[['Chrom','Start','End','UnknownCol1','UnknownCol2','Strand','pairChrom','pairStart','pairEnd','Cluster']]
    pairs.to_csv(out_file_contact_pairs, sep='\t', header=None, index=False)

def perform_plotting_preprocessing(features,n_components=2):
    '''perform preprocessing for plotting'''
    
    Sc = StandardScaler()
    scaled = Sc.fit_transform(features)
    pca = PCA(n_components)
    pca.fit(scaled)
    return pca.transform(scaled)

def perform_clustering_preprocessing(features):
    '''perform preprocessing for clustering'''
    
    return features
    
def plot_results(features,clusters,out_file_fig):
    '''plot clustering results'''
    
    components = perform_plotting_preprocessing(features,n_components=2)
    fig,ax = plt.subplots()
    scatter = ax.scatter(components[:,0], components[:,1],c=clusters,cmap='Set3',alpha=0.8)
    legend1 = ax.legend(*scatter.legend_elements(),loc="upper left", title="")
    ax.add_artist(legend1)
    
    #fig.show()
    fig.savefig(out_file_fig)

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

    parserRequired.add_argument('--outFileContactPairs',
                                '-o',
                                help='file name for output bed file',
                                required=True,
                                type=str)
    
    parserRequired.add_argument('--outFileFig',
                                '-p',
                                help='file name for output plot',
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
                           default=9)
    
    parserOpt.add_argument('--clusterAlgorithm',
                           help='cluster algorithm to use for computation',
                           type=str,
                           default=None)

    parserOpt.add_argument('--threads', '-t',
                           help='number of threads used',
                           default=4,
                           type=int)

    parserOpt.add_argument("--help", "-h", action="help",
                           help="show this help message and exit")

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    
    parserOpt.add_argument('--devFeatureType',
                           choices=[
                               'per_region',
                               'per_pair'
                           ],
                           default='per_region',
                           help='for development: use features per region or per pair')

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
    out_file_contact_pairs = args.outFileContactPairs
    out_file_fig = args.outFileFig
    dev_feature_type = args.devFeatureType
    
    resolution = 1
    corner_position = 'upper_left'
    corner_size = 2
    
    #check for faulty parameters
    
    #ingest matrix file(s)
    print('reading matrix file')
    pChromosome = None
    matrix = read_matrix_file(matrix_file, pChromosome)
    
    print('reading bed file')
    #ingest bed file
    regions = read_regions_bed(bed_file)
    
    print('normalizing matrix file')
    #normalize matrix file(s)
    matrix = obs_exp_normalization(matrix, pThreads=threads)
    
    print('calculating valid interaction pairs')
    #get pairs
    pairs = get_pairs(regions,min_distance,max_distance,resolution)
    
    print('cutting out submatrices for interaction pairs')
    #cut out submatrices
    pairs, submatrices = get_submatrices(matrix,pairs,submatrix_size)
    
    print('aggregating features for clustering')
    #aggregate features from submatrices
    features = get_features(submatrices,center_size=center_size,corner_position=corner_position,corner_size=corner_size)
    
    if(dev_feature_type == 'per_region'):
        pairs, indices_list, features = get_feature_matrix(pairs,features)
    
    print('clustering')
    #cluster submatrices
    clusters = perform_clustering(features,k,cluster_algorithm=cluster_algorithm)
    
    print('writing results to file')
    #output results
    output_results(out_file_contact_pairs,pairs,indices_list,clusters)
    
    print('plotting results in scatter plot')
    plot_results(features,clusters,out_file_fig)
    print('Done')
