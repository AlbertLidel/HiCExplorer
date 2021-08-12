import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
from hicexplorer import hicClusterContacts as hcc
from hicmatrix import HiCMatrix as hm
from tempfile import mkdtemp
import shutil
import os
import numpy.testing as nt
import numpy as np
import pandas as pd
from hicexplorer.test.test_compute_function import compute

#TODO
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data/")
#ROOT = "test_data/"

def test_read_matrix_file():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    assert m.matrix.shape == (3158,3158)

def test_obs_exp_normalization():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    m = hcc.obs_exp_normalization(m)
    assert m.matrix.shape == (3158,3158)

def test_read_regions_bed():
    df = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    assert df.shape == (37, 7) and df.iloc[0, 0] == '2L'
    
def test_get_positions():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    pos = hcc.get_positions(m)
    assert pos['Start'].iloc[4] == 1829

def test_build_position_index():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    pos = hcc.get_positions(m)    
    index = hcc.build_position_index(pos)
    assert index[('2L',1829)] == 4
    
def test_get_pairs():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs['comparison'] = pairs['Start'] - pairs['pairStart']
    assert (pairs['comparison'] >= 1000000).all()
    assert (pairs['comparison'] <= 20000000).all()
    assert (pairs['Chrom'] == pairs['pairChrom']).all()
    assert pairs.shape == (170,14)
    
def test_get_submatrices():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    assert submatrices[0].shape == (9,9)
    assert len(submatrices) == 170
    
def test_get_features():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    assert np.nanmax(features) > 0
    
def test_get_feature_matrix():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    feature_matrix = hcc.get_feature_matrix(pairs,features,regions)
    assert np.nanmax(feature_matrix) > 0

def test_perform_clustering():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    feature_matrix = hcc.get_feature_matrix(pairs,features,regions)
    clusters = hcc.perform_clustering(feature_matrix,3)
    c = pd.Series(clusters)
    assert len(c.unique()) > 1
    
def test_output_results():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    feature_matrix = hcc.get_feature_matrix(pairs,features,regions)
    clusters = hcc.perform_clustering(feature_matrix,3)
    test_folder = mkdtemp(prefix="test_case_cluster_contacts")
    pairs['Cluster'] = hcc.region_to_pair_labels(pairs,clusters,'Cluster')
    hcc.output_results(test_folder + 'test_output.bed',pairs)
    bed_df = pd.read_csv(test_folder + 'test_output.bed', sep="\t", header=None)
    assert len(bed_df) == 170
    
def test_output_results_alt():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    clusters = hcc.perform_clustering(features,3)
    test_folder = mkdtemp(prefix="test_case_cluster_contacts")
    pairs['Cluster'] = clusters
    hcc.output_results(test_folder + 'test_output.bed',pairs)
    bed_df = pd.read_csv(test_folder + 'test_output.bed', sep="\t", header=None)
    assert len(bed_df) == 170

def test_plot_results():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    feature_matrix = hcc.get_feature_matrix(pairs,features,regions)
    clusters = hcc.perform_clustering(features,3)
    test_folder = mkdtemp(prefix="test_case_cluster_contacts")
    pairs['Cluster'] = hcc.region_to_pair_labels(pairs,clusters,'Cluster')
    hcc.plot_results(features,clusters,test_folder + 'test_output_fig.png')
    #plot_results(features,clusters,'test_output_fig.png')
    assert True

def test_plot_submatrices():
    m = hcc.read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = hcc.get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    regions,pairs,submatrices = hcc.get_submatrices(m,regions,pairs,submatrix_size=9)
    features = hcc.get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    clusters = hcc.perform_clustering(features,3)
    #min_value = np.min(m.matrix)
    #max_value = np.max(m.matrix)
    min_value=None
    max_value=None
    test_folder = mkdtemp(prefix="test_case_plot_submatrices")
    hcc.plot_submatrices(submatrices, clusters,'test_output_fig.png',vmin=min_value,vmax=max_value)
    assert True
    
def test_mix_with_random_regions():
    region_start = 1000
    region_end = 30000
    regions = hcc.read_regions_bed(ROOT + 'unittest_regions.bed')
    n_non_random = regions.shape[0]
    regions,is_random = hcc.mix_with_random_regions(regions,region_start=None,region_end=None)
    #print(regions)
    #print(is_random)
    assert regions.shape[0] == is_random.shape[0]
    assert regions.shape[0] >= 2*n_non_random
    assert regions.shape[0] <= int(2.1*n_non_random)
    
def test_get_random_regions():
    region_start = 1000
    region_end = 30000
    amount = 20
    chromosome = '2L'
    region_size = 10
    r = hcc.get_random_regions(amount,region_start,region_end,chromosome,region_size=region_size)
    assert r.shape[0] == amount
    assert r.shape[1] == 7
    assert r['Chrom'][0] == chromosome
    assert r['End'][0] - r['Start'][0] == region_size
    assert np.min(r['Start'].to_numpy()) >= region_start
    assert np.max(r['Start'].to_numpy()) <= region_end

