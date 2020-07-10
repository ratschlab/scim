import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData

from prosstt.tree import Tree
from prosstt import simulation
from prosstt import count_model


def df_from_rank_gene_groups(adata, drop_duplicates=True):
    '''Wraps results of sc.tl.rank_genes_groups in a df
    '''

    assert 'rank_genes_groups' in adata.uns
    columns = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']
    df_data = list()
    for col in columns:
        ser = pd.DataFrame(adata.uns['rank_genes_groups'][col]).unstack()
        ser.name = col
        df_data.append(ser)

    df = pd.DataFrame(df_data).T
    df = df.reset_index(level=1, drop=True)  # drop RangeIndex
    df.index.name = adata.uns['rank_genes_groups']['params']['groupby']
    df = df.reset_index().set_index('names')
    df = df.sort_values('scores', ascending=False)

    if drop_duplicates:
        df = df[~df.index.duplicated(keep='first')]

    return df


def simulate(tree, ngenes, ncells, ntest=2000):
    '''Simulates data using PROSSTT

    Data is generated and wrapped within an AnnData instance.
    Count data is sampled from a negative binomial distribution,
    normalized using sc.pp.normalize,
    and logged using sc.pp.log1p

    Labels are stored in obs: obs['branch'], pseudotime in obs['pt']
    Marker genes for branches are stored in var['marker']

    Args:
        tree: newick string defining branching process (e.g. '(B:50,C:50)A:50')
        ngenes: number of genes (features) to parameterize
        ncells: number of cells to sample
        ntest: number of cells to label as test data

    Returns:
        data: AnnData with labels, marker genes

    '''
    assert ntest < ncells
    lineage = Tree.from_newick(tree, genes=ngenes)
    alpha, beta = count_model.generate_negbin_params(lineage)
    lineage.default_gene_expression()
    X, pt, branches, _ = simulation.sample_density(lineage, ncells,
                                                   alpha=alpha, beta=beta,
                                                   scale=False)

    data = AnnData(X, obs={'pt': pt, 'branch': branches})
    data.obs['branch'] = data.obs['branch'].astype('category')

    sc.pp.normalize_total(data)
    sc.pp.log1p(data)

    # find good markers
    sc.tl.rank_genes_groups(data, groupby='branch')
    df = df_from_rank_gene_groups(data)
    markers = df.groupby('branch').head(5).index
    data.var['marker'] = False
    data.var.loc[markers, 'marker'] = True

    sc.tl.pca(data)

    test_idxs = np.random.choice(data.n_obs, size=ntest, replace=False)
    is_train = np.ones(data.n_obs, dtype=bool)
    is_train[test_idxs] = False
    data.obs['is_train'] = is_train

    # TODO: update anndata to save this (bug reading/writing)
    data.uns.pop('rank_genes_groups')

    return data
