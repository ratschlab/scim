import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import itertools
import time
from scipy.spatial import cKDTree


def get_null_cost(cost, null_cost_percentile=90):
    """Compute the cost of matching to the null node based on all other costs
    cost: vector of costs (weights on all source->target edges)
    null_cost_percentile: percentile of the cost that should correspond to null node matching
    """
    null_cost = int(np.ceil(np.percentile(cost, null_cost_percentile)))
        
    return null_cost

def extend_graph_null(G, source_idx, null_cost):
    """Extend the graph by adding a null node on target side (linked to source nodes and sink)
    """
    null_capacity = len(source_idx)
    G.add_node('target_null')
    source_null_edges = list(itertools.product(source_idx, ['target_null'], [{'capacity': 1, 'weight':null_cost}]))
    G.add_edges_from(source_null_edges)
    null_sink_edges = list(itertools.product(['target_null'], ['sink'], [{'capacity': null_capacity, 'weight':0}]))
    G.add_edges_from(null_sink_edges)
    
    return G

def get_target_capacities(source_idx, target_idx, method='uniform', seed=456):
    """Compute a vector of capacities from target nodes to sink
    """
    if(method=='inf'):
        capacity = np.inf
        capacities = [capacity]*len(target_idx)
    elif(method=='top'):
        capacity = len(source_idx) + 1000
        capacities = [capacity]*len(target_idx)
    elif(method=='uniform'):
        capacity = int(np.floor(len(source_idx)/len(target_idx)))
        capacities = [capacity]*len(target_idx)
        # randomly distribute remaining cells
        n_remaining = len(source_idx) - np.sum(capacities)
        np.random.seed(seed)
        to_add_idx = np.random.choice(range(len(capacities)),n_remaining, replace=False)
        for i in to_add_idx:
            capacities[i] = capacities[i] + 1
    elif(method=='1to1'):
        capacity = 1
        capacities = [capacity]*len(target_idx)
    else:
        raise NotImplementedError
        
    return capacities

def build_graph_base(source_idx, target_idx, capacity_method='uniform', seed=456):
    """Build a graph base
    """
    G = nx.DiGraph()
    # add initial nodes
    G.add_node('root')
    G.add_node('sink')
    G.add_nodes_from(source_idx)
    G.add_nodes_from(target_idx)
    # add edges
    source_root_edges = list(itertools.product(['root'], source_idx, [{'capacity': 1, 'weight':0}]))
    G.add_edges_from(source_root_edges)

    capacities = get_target_capacities(source_idx, target_idx, method=capacity_method, seed=seed)
    target_sink_edges = [(target_idx[i], 'sink', {'capacity':capacities[i], 'weight':0}) for i in range(len(target_idx))]
    G.add_edges_from(target_sink_edges)
    
    return G

def convert_to_int(vec, factor):
    """Convert float values into integers by multiplying by factor and cropping the floating points
    """
    vec = factor*vec
    vec = [int(x) for x in vec]
    
    return vec

def get_knn_union(knn_source_idx, knn_target_idx, knn_dist,
             knn2_source_idx, knn2_target_idx, knn2_dist):
    """Compute a union of knn connections from one-sided searches
    """
    knn_source_idx = np.append(knn_source_idx, knn2_source_idx)
    knn_target_idx = np.append(knn_target_idx, knn2_target_idx)
    knn_dist = np.append(knn_dist, knn2_dist)
    # remove duplicated connections
    knn_df = pd.DataFrame({'source':knn_source_idx, 'target':knn_target_idx, 'dist':knn_dist})
    knn_df = knn_df.drop_duplicates() 
    knn_source_idx = knn_df['source'].to_list()
    knn_target_idx = knn_df['target'].to_list()
    knn_dist = np.array(knn_df['dist'].to_list())
    
    return knn_source_idx, knn_target_idx, knn_dist


def get_mnn(knn_source_idx, knn_target_idx, knn_dist,
            knn2_source_idx, knn2_target_idx, knn2_dist):
    """Compute an intersection of knn connections (MNN) from one-sided searches
    """    
    knn_connections = [x+'__'+y+'__'+str(z) for x,y,z in zip(knn_source_idx, knn_target_idx, knn_dist)]
    knn2_connections = [x+'__'+y+'__'+str(z) for x,y,z in zip(knn2_source_idx, knn2_target_idx, knn2_dist)]
    keep_connections = set(knn_connections).intersection(set(knn2_connections))
    mnn_source_idx = [x.split('__')[0] for x in list(keep_connections)]
    mnn_target_idx = [x.split('__')[1] for x in list(keep_connections)]
    mnn_dist = np.array([float(x.split('__')[-1]) for x in list(keep_connections)]) 
    
    return mnn_source_idx, mnn_target_idx, mnn_dist

def get_knn(source, target, distance_measure, knn_k, knn_n_jobs, knn_method='one-sided'):
    """Get kNN connections between source and target
    distance_measure: euclidean
    knn_k: k neighbors to be found
    knn_n_jobs: how many processors to use for the job
    knn_method: method to get kNN connections {'one-sided': one-sided, tree built using target, queried using source,
                                    'union': perform kNN search from each side and take union of connections,
                                    'mnn': perform kNN search from each side and take intersection of connections}
    """
    knn = cKDTree(target).query(x=source, k=knn_k, n_jobs=knn_n_jobs)
    knn_source_idx = ['source_'+str(x) for x in np.array(range(source.shape[0])).repeat(knn_k)]
    knn_target_idx = ['target_'+str(x) for x in knn[1].reshape((-1,1)).flatten()]
    knn_dist = knn[0].reshape((-1,1)).flatten()
    
    if(knn_method!='one-sided'):
        knn2 = cKDTree(source).query(x=target, k=knn_k, n_jobs=knn_n_jobs)
        knn2_target_idx = ['target_'+str(x) for x in np.array(range(target.shape[0])).repeat(knn_k)]
        knn2_source_idx = ['source_'+str(x) for x in knn2[1].reshape((-1,1)).flatten()]
        knn2_dist = knn2[0].reshape((-1,1)).flatten()
        if(knn_method=='union'):
            knn_source_idx, knn_target_idx, knn_dist = get_knn_union(knn_source_idx, knn_target_idx, knn_dist,
                                                                     knn2_source_idx, knn2_target_idx, knn2_dist)
        if(knn_method=='mnn'):
            knn_source_idx, knn_target_idx, knn_dist = get_mnn(knn_source_idx, knn_target_idx, knn_dist,
                                                               knn2_source_idx, knn2_target_idx, knn2_dist)
    return knn_source_idx, knn_target_idx, knn_dist


def get_cost_knn_graph(source, target, distance_measure, factor, cost_type, knn_k, knn_n_jobs,
                       knn_method='one-sided', capacity_method='uniform', add_null=True,
                       null_cost_percentile=90, seed=456):
    """Build an extended graph based on knn indices
    """
    knn_source_idx, knn_target_idx, knn_dist = get_knn(source, target, distance_measure, knn_k, knn_n_jobs, knn_method)
    
    if(cost_type=='percentile'):
        import time
        t_start = time.process_time()
        len_p = int(len(knn_dist))
        p = np.linspace(min(knn_dist),max(knn_dist),100)
        knn_dist = np.digitize(knn_dist, bins=p)
        t_stop = time.process_time()
        t = (t_stop-t_start)
        print('Percentile computation took [s]:'+str(t))
        
    knn_dist = convert_to_int(knn_dist, factor)
    print('Max dist: ',np.max(knn_dist))
    
    source_idx = ['source_'+str(x) for x in range(source.shape[0])]
    target_idx = ['target_'+str(x) for x in range(target.shape[0])]
    
    G = build_graph(source_idx, target_idx, knn_source_idx, knn_target_idx, knn_dist,
                    capacity_method=capacity_method, add_null=add_null, 
                    null_cost_percentile=null_cost_percentile, seed=seed)

    print('Number of nodes: ', len(G.nodes))
    print('Number of edges: ', len(G.edges))

    return G
        
def extract_matches_flow(flow_dict, keep_only='source'):
    """Extract the matched pairs
    flow_dict: ooutput from the max_flow_min_cost algorithm
    keep_only: discard all matches where keep_only doesn't appear
               (eg if keep_only=='source', then only inter-technology matches are reported)
    """
    matches_source = []
    matches_target = []
    for f in flow_dict.keys():
        if(keep_only is not None):
            if(keep_only not in f):
                continue
        matches_f = [x for x in flow_dict[f].keys() if flow_dict[f][x]>0]
        matches_source.extend([f]*len(matches_f))
        matches_target.extend(matches_f)
    matches = pd.DataFrame({'source': matches_source,
                            'target': matches_target})
    return matches

def scmatch_mcmf(G):
    """ Use the Min-Cost Max-Flow algorithm to find the best matches between cells across technologies
    """
    t_start = time.process_time()
    
    flow_dict = nx.max_flow_min_cost(G,'root', 'sink', capacity='capacity', weight='weight')   
    
    t_stop = time.process_time()
    t = (t_stop-t_start)
    print('MCMF took [s]:'+str(t))
    
    matches = extract_matches_flow(flow_dict, keep_only='source')
    row_ind = [x.split('_')[-1] for x in matches['source']]
    row_ind = [np.nan if x=='null' else int(x) for x in row_ind]
    col_ind = [x.split('_')[-1] for x in matches['target']]
    col_ind = [np.nan if x=='null' else int(x) for x in col_ind]
    
    return row_ind, col_ind
