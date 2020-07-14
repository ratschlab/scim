import numpy as np
import scipy as sp
import pandas as pd
import anndata
from universal_divergence import estimate
from itertools import combinations


def score_divergence(codes, sources, k=50, **kwargs):
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
        sources: index of each item's source (e.g tech; data or prior)
        k: k-NN used to estimate data density
        kwargs: see preprocess_code

    outputs:
        divergence score (non-negative)
    """
    num_datasets = np.unique(sources).size
    div_pq = list()
    div_qp = list()

    # calculate divergece score for each pair of datasets
    for d1, d2 in combinations(range(num_datasets), 2):
        idx1 = sources == d1
        idx2 = sources == d2
        
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

def extract_matched_labels(source, target, row_idx, col_idx, keep_cols=None):
    """ Merge all the metainfo (labels) for the matches
    source, target: anndata with ".obs" containing all the metainfo, rows correspond to cells
    row_idx: numerical indices of the matches (source)
    col_idx: numerical indices of the matches (target)
    keep_cols: which columns to keep (if None, all columns are kept)
    """
    
    labels_source = source.obs.copy()
    labels_target = target.obs.copy()
    
    indices = pd.DataFrame({'source':row_idx, 'target':col_idx}).dropna()
    indices['target'] = [int(x) for x in indices['target']]
    
    if 'source' not in labels_source.columns:
        labels_source.columns = [x+'_source' for x in labels_source.columns]
    labels_source = labels_source.iloc[indices['source'].values,:].reset_index(drop=True)
    
    if 'target' not in labels_target.columns:
        labels_target.columns = [x+'_target' for x in labels_target.columns]
    labels_target = labels_target.iloc[indices['target'].values,:].reset_index(drop=True) 
    
    labels_matched = pd.concat([labels_source, labels_target], axis=1, ignore_index=False)
    labels_matched['index_source'] = source.obs_names[indices['source'].values]
    labels_matched['index_target'] = target.obs_names[indices['target'].values]
    
    if(keep_cols is not None):
        keep_cols.append('index')
        keep_colnames = [x+'_source' for x in keep_cols]
        keep_colnames.extend([x+'_target' for x in keep_cols])
        labels_matched = labels_matched.loc[:,keep_colnames]
    
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
    if(np.sum(matches.columns.isin([colname_compare+'_source', colname_compare+'_target']))!=2):
        print('The input dataframe does not include {colname_compare} information!')
    else:
        y_source = pd.Series(matches[colname_compare+'_source'], name='Source')
        y_target = pd.Series(matches[colname_compare+'_target'], name='Target')
    
        return pd.crosstab(y_source, y_target)

def get_correlation(matches, colname_compare='pt', round_to=2):
    """ Get the correlation of pseudotime
    matches: pandas dataframe, output from extract_matched_labels()
    colname_compare: column name to use for correlation calculation {colname_compare}_source,
                     {colname_compare}_target must be in matches.columns
    round_to: round to which decimal place
    """
    if(np.sum(matches.columns.isin([colname_compare+'_source', colname_compare+'_target']))!=2):
        print('The input dataframe does not include {colname_compare} information!')
        cor_sp = np.nan
        cor_p = np.nan
    else:
        y_source = pd.Series(matches[colname_compare+'_source'], name='Source')
        y_target = pd.Series(matches[colname_compare+'_target'], name='Target')
        cor_sp = round(sp.stats.spearmanr(y_source, y_target)[0], round_to)
        cor_p = round(sp.stats.pearsonr(y_source, y_target)[0], round_to)
        
    return cor_sp, cor_p