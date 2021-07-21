import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
from hicexplorer import hicFindTADs
from hicmatrix import HiCMatrix as hm
from tempfile import mkdtemp
import shutil
import os
import numpy.testing as nt
from hicexplorer.test.test_compute_function import compute


#ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data/")
ROOT = "test_data/"

def test_read_matrix_file():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    assert m.matrix.shape == (3158,3158)

def test_obs_exp_normalization():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    m = obs_exp_normalization(m)
    assert m.matrix.shape == (3158,3158)

def test_read_regions_bed():
    df = read_regions_bed(ROOT + 'unittest_regions.bed')
    assert df.shape == (37, 6) and df.iloc[0, 0] == '2L'
    
def test_get_positions():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    pos = get_positions(m)
    assert pos['Start'].iloc[4] == 1829

def test_build_position_index():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    pos = get_positions(m)    
    index = build_position_index(pos)
    assert index[('2L',1829)] == 4
    
def test_get_pairs():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs['comparison'] = pairs['Start'] - pairs['pairStart']
    assert (pairs['comparison'] >= 1000000).all()
    assert (pairs['comparison'] <= 20000000).all()
    assert (pairs['Chrom'] == pairs['pairChrom']).all()
    assert pairs.shape == (170,10)
    
def test_get_submatrices():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    indices,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    assert submatrices[0].shape == (9,9)
    assert len(submatrices) == 170
    
def test_get_features():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    assert np.nanmax(features) > 0
    
def test_get_feature_matrix():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    pairs, indices_list, feature_matrix = get_feature_matrix(pairs,features)
    assert np.nanmax(feature_matrix) > 0

def test_perform_clustering():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    pairs, indices_list, feature_matrix = get_feature_matrix(pairs,features)
    clusters = perform_clustering(feature_matrix,3)
    c = pd.Series(clusters)
    assert len(c.unique()) > 1
    
def test_output_results():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    pairs, indices_list, feature_matrix = get_feature_matrix(pairs,features)
    clusters = perform_clustering(feature_matrix,3)
    test_folder = mkdtemp(prefix="test_case_cluster_contacts")
    output_results(test_folder + 'test_output.bed',pairs,indices_list,clusters)
    bed_df = pd.read_csv(test_folder + 'test_output.bed', sep="\t", header=None)
    assert len(bed_df) == 170
    
def test_output_results_alt():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    clusters = perform_clustering(features,3)
    test_folder = mkdtemp(prefix="test_case_cluster_contacts")
    output_results_alt(test_folder + 'test_output.bed',pairs,clusters)
    bed_df = pd.read_csv(test_folder + 'test_output.bed', sep="\t", header=None)
    assert len(bed_df) == 170

def test_plot_results():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    pairs, indices_list, features = get_feature_matrix(pairs,features)
    clusters = perform_clustering(features,3)
    test_folder = mkdtemp(prefix="test_case_cluster_contacts")
    plot_results(features,clusters,test_folder + 'test_output_fig.png')
    #plot_results(features,clusters,'test_output_fig.png')
    assert True

def test_plot_submatrices():
    m = read_matrix_file(ROOT + 'unittest_matrix.h5', None)
    regions = read_regions_bed(ROOT + 'unittest_regions.bed')
    pairs = get_pairs(regions,min_distance=1000000,max_distance=20000000,resolution=1)
    pairs,submatrices = get_submatrices(m,pairs,submatrix_size=9)
    features = get_features(submatrices,center_size=0.25,corner_position='upper_left',corner_size=2)
    clusters = perform_clustering(features,3)
    #min_value = np.min(m.matrix)
    #max_value = np.max(m.matrix)
    min_value=None
    max_value=None
    test_folder = mkdtemp(prefix="test_case_plot_submatrices")
    plot_submatrices(submatrices, clusters,'test_output_fig.png',vmin=min_value,vmax=max_value)
    assert True
