import numpy as np
import scipy as sp
import pandas as pd
import anndata
from universal_divergence import estimate


def score_divergence(codes, labels=None, sources=None, k=50, **kwargs):
    """
    Measures how well sources are mixed (smaller: well-mixed)

    Function to calculate the divergence score as described in BERMUDA (Wang et al., 2019)
    Code was adapted form https://github.com/txWang/BERMUDA/blob/master/evaluate.py

    Estimates the avg pairwise symmetric divergence of p_src and q_tgt
    i.e. .5 * D(p_src || q_tgt) + .5 D(q_tgt || p_src) for each src tgt pair

    p and q eval with a non-parametric density estimate centered at x_i
    i.e weighthed by the distance to the kth-NN from x_i for each dataset


    inputs:
        codes: merged data matrix
        labels: labels of each item (e.g. cell-type), if None then all cells are used
        sources: index of each item's source (e.g tech; data or prior)
        k: k-NN used to estimate data density
        kwargs: see preprocess_code

    outputs:
        divergence score,  non-negative
    """
    num_datasets = np.unique(sources).size
    div_pq = list()
    div_qp = list()

    # pairs of datasets
    for d1, d2 in combinations(range(num_datasets), 2):
        idx1, idx2, _ = separate_shared_idx(labels, sources, d1=d1, d2=d2)
        if sum(idx1) < k or sum(idx2) < k:
            continue

        pq = estimate(codes[idx1, :], codes[idx2, :], k)
        div_pq.append(max(pq, 0))

        qp = estimate(codes[idx2, :], codes[idx1, :], k)
        div_qp.append(max(qp, 0))

    # average the scores across pairs of datasets
    try:
        div_score = (sum(div_pq) / len(div_pq) + sum(div_qp) / len(div_qp)) / 2
    except ZeroDivisionError:
        div_score = np.nan
    return div_score


def separate_shared_idx(labels=None, sources=None, d1=0, d2=1):
    """
    Function to split index into shared and distinct cell-types between a pair of sources
    (needed for calculation of divergence and entropy scores)
    inputs:
        labels: labels of each item (e.g. cell-type), if None then all cells are used
        sources: index of each item's source (e.g tech; data or prior)
    outputs:
        logical idx of shared cell-types (per data source), logical idx of distinct cell-types
    """
    src1 = sources == d1
    src2 = sources == d2
    if(labels is None):
        src1_mutual = src1
        src2_mutual = src2
        src_specific = []
    else:
        shared_labels = np.intersect1d(np.unique(labels[src1]), np.unique(labels[src2]))
        src1_mutual = np.logical_and(src1, np.isin(labels, shared_labels))
        src2_mutual = np.logical_and(src2, np.isin(labels, shared_labels))
        src_specific = np.logical_and(np.logical_or(src1, src2), np.logical_not(np.isin(labels, shared_labels)))
    
    return src1_mutual, src2_mutual, src_specific

def extract_matched_labels(labels_source, labels_target, row_idx, col_idx):
    """ Merge all the metainfo (labels) for the matches
    labels_*: pandas dataframe with all the metainfo, rows correspond to cells
    row_idx: numerical indices of the matches (source)
    col_idx: numerical indices of the matches (target)
    """
    
    # bcs anndata is mutable
    labels_source = labels_source.copy()
    labels_target = labels_target.copy()
    
    indices = pd.DataFrame({'source':row_idx, 'target':col_idx}).dropna()
    
    if 'source' not in labels_source.columns:
        labels_source.columns = [x+'_source' for x in labels_source.columns]
    labels_source = labels_source.iloc[indices['source'],:].reset_index(drop=True)
    
    if 'target' not in labels_target.columns:
        labels_target.columns = [x+'_target' for x in labels_target.columns]
    labels_target = labels_target.iloc[indices['target'],:].reset_index(drop=True) 
    
    labels_matched = pd.concat([labels_source, labels_target], axis=1, ignore_index=False)
    
    return labels_matched  

def get_accuracy(matches, colname_compare='cell_type', n_null=0, extended=False):
    """Compute accuracy as true positive fraction
    matches: pandas dataframe, output from extract_matched_labels()
    colname_compare: column name to use for accuracy calculation {colname_compare}_source,
                     {colname_compare}_target must be in matches.columns
    n_null: number of matches with the null node to account for in denominator
    extended: whether to return extended information {accuracy, n_tp, n_fp}
    """
    n_tp = np.sum(matches[colname_compare+'_source']==matches[colname_compare+'_target'])
    n_matches = matches.shape[0] + n_null
    accuracy = n_tp/n_matches
    if(extended):
        return accuracy, n_tp, n_matches-n_tp
    else:
        return accuracy


def get_confusion_matrix(matches, colname_compare='cell_type'):
    """ Get the confusion matrix
    matches: pandas dataframe, output from extract_matched_labels()
    colname_compare: column name to use for accuracy calculation {colname_compare}_source,
                     {colname_compare}_target must be in matches.columns
    """
    y_source = pd.Series(matches[colname_compare+'_source'], name='Source')
    y_target = pd.Series(matches[colname_compare+'_target'], name='Target')
    
    return pd.crosstab(y_source, y_target)