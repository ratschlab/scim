import numpy as np
from universal_divergence import estimate


def score_divergence(codes, labels, sources, k=50, **kwargs):
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
        labels: labels of each item (e.g. cell-type)
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
    for d1 in range(num_datasets):
        for d2 in range(d1+1, num_datasets):
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


def separate_shared_idx(labels, sources, d1=0, d2=1):
    """
    Function to split index into shared and distinct cell-types between a pair of sources
    (needed for calculation of divergence and entropy scores)
    inputs:
        labels: labels of each item (e.g. cell-type)
        sources: index of each item's source (e.g tech; data or prior)
    outputs:
        logical idx of shared cell-types (per data source), logical idx of distinct cell-types
    """
    src1 = sources == d1
    src2 = sources == d2
    shared_labels = np.intersect1d(np.unique(labels[src1]), np.unique(labels[src2]))
    src1_mutual = np.logical_and(src1, np.isin(labels, shared_labels))
    src2_mutual = np.logical_and(src2, np.isin(labels, shared_labels))
    src_specific = np.logical_and(np.logical_or(src1, src2), np.logical_not(np.isin(labels, shared_labels)))
    return src1_mutual, src2_mutual, src_specific
