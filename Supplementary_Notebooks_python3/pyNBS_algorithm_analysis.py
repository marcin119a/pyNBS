# Import pyNBS modules
from pyNBS import data_import_tools as dit
from pyNBS import network_propagation as prop
from pyNBS import pyNBS_core as core
from pyNBS import pyNBS_single
from pyNBS import consensus_clustering as cc
from pyNBS import pyNBS_plotting as plot

# Import other needed packages
import os
import time
import pandas as pd
import numpy as np

# Import packages needed for measuring clustering similarity
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
import scipy.stats as stats


# ─── Data Loading ─────────────────────────────────────────────────────────────

sm_data_filepath = './Supplementary_Notebook_Data/BLCA_sm_data.txt'
sm_mat = dit.load_binary_mutation_data(sm_data_filepath, filetype='list', delimiter='\t')

BLCA_surv_data = './Supplementary_Notebook_Data/BLCA.clin.merged.surv.txt'

# Output directory setup
outdir = './Supplementary_Notebook_Results/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

alpha = 0.7
clusters, niter = 4, 100


# ─── Helper: cluster comparison ───────────────────────────────────────────────

def compare_clusters(base_assign, new_assign, clusters):
    cc_align = pd.concat([base_assign, new_assign], axis=1).dropna()
    cc_align.columns = ['Base', 'New']
    base_clust_assign = cc_align['Base']
    new_clust_assign = cc_align['New']

    adj_rand_index = adjusted_rand_score(base_clust_assign, new_clust_assign)
    adj_mutual_info = adjusted_mutual_info_score(base_clust_assign, new_clust_assign)
    print('Adjusted Rand Index is: ' + str(adj_rand_index))
    print('Adjusted Mutual Info Score is: ' + str(adj_mutual_info))

    intersect_pats = list(cc_align.index)
    NBS_cont_table_array = []
    for i in range(1, clusters + 1):
        base_cluster = set(base_clust_assign.loc[intersect_pats][base_clust_assign.loc[intersect_pats] == i].index)
        base_pyNBS_cluster_intersect = []
        for j in range(1, clusters + 1):
            new_cluster = set(new_clust_assign.loc[intersect_pats][new_clust_assign.loc[intersect_pats] == j].index)
            base_pyNBS_cluster_intersect.append(len(base_cluster.intersection(new_cluster)))
        NBS_cont_table_array.append(base_pyNBS_cluster_intersect)

    cont_table = pd.DataFrame(
        NBS_cont_table_array,
        index=['Original pyNBS Cluster ' + repr(i) for i in range(1, clusters + 1)],
        columns=['New pyNBS Cluster ' + repr(i) for i in range(1, clusters + 1)]
    )
    print(cont_table)

    chi_sq_test = stats.chi2_contingency(NBS_cont_table_array, correction=False)
    print('Chi-Squared Statistic:', chi_sq_test[0])
    print('Chi-Squared P-Value:', chi_sq_test[1])


# ─── Section 1: pyNBS on BLCA with Cancer Subnetwork (base) ───────────────────

print('\n=== pyNBS on BLCA with Cancer Subnetwork (base) ===')

save_args = {'outdir': outdir, 'job_name': 'BLCA_CSN_base'}

CSN = dit.load_network_file('./Supplementary_Notebook_Data/CancerSubnetwork.txt')
CSN_knnGlap = core.network_inf_KNN_glap(CSN)

CSN_network_nodes = CSN.nodes()
CSN_network_I = pd.DataFrame(np.identity(len(CSN_network_nodes)), index=CSN_network_nodes, columns=CSN_network_nodes)
CSN_kernel = prop.network_propagation(CSN, CSN_network_I, alpha=alpha, symmetric_norm=False)

BLCA_CSN_Hlist = []
for i in range(niter):
    BLCA_CSN_Hlist.append(pyNBS_single.NBS_single(sm_mat, CSN_knnGlap, propNet=CSN, propNet_kernel=CSN_kernel, k=clusters))

BLCA_CSN_NBS_cc_table, BLCA_CSN_NBS_cc_linkage, BLCA_CSN_NBS_cluster_assign = cc.consensus_hclust_hard(BLCA_CSN_Hlist, k=clusters, **save_args)

BLCA_CSN_p = plot.cluster_KMplot(BLCA_CSN_NBS_cluster_assign, BLCA_surv_data, delimiter=',', **save_args)


# ─── Section 2: pyNBS on BLCA with shuffled Cancer Subnetwork ─────────────────

print('\n=== pyNBS on BLCA with shuffled Cancer Subnetwork ===')

save_args['job_name'] = 'BLCA_CSNshuff'

CSNshuff = dit.load_network_file('./Supplementary_Notebook_Data/CancerSubnetwork.txt', degree_shuffle=True)
CSNshuff_knnGlap = core.network_inf_KNN_glap(CSNshuff)

CSNshuff_network_nodes = CSNshuff.nodes()
CSNshuff_network_I = pd.DataFrame(np.identity(len(CSNshuff_network_nodes)), index=CSNshuff_network_nodes, columns=CSNshuff_network_nodes)
CSNshuff_kernel = prop.network_propagation(CSNshuff, CSNshuff_network_I, alpha=alpha, symmetric_norm=False)

BLCA_CSNshuff_Hlist = []
for i in range(niter):
    BLCA_CSNshuff_Hlist.append(pyNBS_single.NBS_single(sm_mat, CSNshuff_knnGlap, propNet=CSNshuff, propNet_kernel=CSNshuff_kernel, k=clusters))

BLCA_CSNshuff_NBS_cc_table, BLCA_CSNshuff_NBS_cc_linkage, BLCA_CSNshuff_NBS_cluster_assign = cc.consensus_hclust_hard(BLCA_CSNshuff_Hlist, k=clusters, **save_args)

BLCA_CSNshuff_p = plot.cluster_KMplot(BLCA_CSNshuff_NBS_cluster_assign, BLCA_surv_data, delimiter=',', **save_args)

compare_clusters(BLCA_CSN_NBS_cluster_assign, BLCA_CSNshuff_NBS_cluster_assign, clusters)


# ─── Section 3: pyNBS on BLCA with HM90 ───────────────────────────────────────

print('\n=== pyNBS on BLCA with HM90 ===')

save_args['job_name'] = 'BLCA_HM90'

HM90 = dit.load_network_file('./Supplementary_Notebook_Data/HumanNet90_Symbol.txt')
HM90_knnGlap = core.network_inf_KNN_glap(HM90)

HM90_network_nodes = HM90.nodes()
HM90_network_I = pd.DataFrame(np.identity(len(HM90_network_nodes)), index=HM90_network_nodes, columns=HM90_network_nodes)
HM90_kernel = prop.network_propagation(HM90, HM90_network_I, alpha=alpha, symmetric_norm=False)

BLCA_HM90_Hlist = []
for i in range(niter):
    BLCA_HM90_Hlist.append(pyNBS_single.NBS_single(sm_mat, HM90_knnGlap, propNet=HM90, propNet_kernel=HM90_kernel, k=clusters))

BLCA_HM90_NBS_cc_table, BLCA_HM90_NBS_cc_linkage, BLCA_HM90_NBS_cluster_assign = cc.consensus_hclust_hard(BLCA_HM90_Hlist, k=clusters, **save_args)

BLCA_HM90_p = plot.cluster_KMplot(BLCA_HM90_NBS_cluster_assign, BLCA_surv_data, delimiter=',', **save_args)

compare_clusters(BLCA_CSN_NBS_cluster_assign, BLCA_HM90_NBS_cluster_assign, clusters)


# ─── Section 4: pyNBS on BLCA with no network propagation ─────────────────────

print('\n=== pyNBS on BLCA with no network propagation ===')

save_args['job_name'] = 'BLCA_CSN_noprop'

BLCA_CSN_noprop_Hlist = []
for i in range(niter):
    BLCA_CSN_noprop_Hlist.append(pyNBS_single.NBS_single(sm_mat, CSN_knnGlap, k=clusters))

BLCA_CSN_noprop_NBS_cc_table, BLCA_CSN_noprop_NBS_cc_linkage, BLCA_CSN_noprop_NBS_cluster_assign = cc.consensus_hclust_hard(BLCA_CSN_noprop_Hlist, k=clusters, **save_args)

BLCA_CSN_noprop_p = plot.cluster_KMplot(BLCA_CSN_noprop_NBS_cluster_assign, BLCA_surv_data, delimiter=',', **save_args)

compare_clusters(BLCA_CSN_NBS_cluster_assign, BLCA_CSN_noprop_NBS_cluster_assign, clusters)


# ─── Section 5: pyNBS on BLCA with no network regularization ──────────────────

print('\n=== pyNBS on BLCA with no network regularization ===')

save_args['job_name'] = 'BLCA_CSN_noreg'
NBS_single_params = {'netNMF_lambda': '0'}

BLCA_CSN_noreg_Hlist = []
for i in range(niter):
    BLCA_CSN_noreg_Hlist.append(pyNBS_single.NBS_single(sm_mat, CSN_knnGlap, k=clusters, **NBS_single_params))

BLCA_CSN_noreg_NBS_cc_table, BLCA_CSN_noreg_NBS_cc_linkage, BLCA_CSN_noreg_NBS_cluster_assign = cc.consensus_hclust_hard(BLCA_CSN_noreg_Hlist, k=clusters, **save_args)

BLCA_CSN_noreg_p = plot.cluster_KMplot(BLCA_CSN_noreg_NBS_cluster_assign, BLCA_surv_data, delimiter=',', **save_args)

compare_clusters(BLCA_CSN_NBS_cluster_assign, BLCA_CSN_noreg_NBS_cluster_assign, clusters)


# ─── Section 6: pyNBS on BLCA with no consensus clustering ────────────────────

print('\n=== pyNBS on BLCA with no consensus clustering ===')

save_args['job_name'] = 'BLCA_CSN_nocc'

BLCA_CSN_nocc_H = pyNBS_single.NBS_single(sm_mat, CSN_knnGlap, propNet=CSN, propNet_kernel=CSN_kernel, k=clusters)

BLCA_CSN_nocc_H.columns = range(1, len(BLCA_CSN_nocc_H.columns) + 1)
BLCA_CSN_nocc_cluster_assign_dict = {}
for pat in BLCA_CSN_nocc_H.index:
    BLCA_CSN_nocc_cluster_assign_dict[pat] = np.argmax(BLCA_CSN_nocc_H.loc[pat])
BLCA_CSN_nocc_cluster_assign = pd.Series(BLCA_CSN_nocc_cluster_assign_dict, name='CC Hard, k=' + repr(clusters))

save_clusters_path = save_args['outdir'] + str(save_args['job_name']) + '_cluster_assignments.csv'
BLCA_CSN_nocc_cluster_assign.to_csv(save_clusters_path)

BLCA_CSN_nocc_p = plot.cluster_KMplot(BLCA_CSN_nocc_cluster_assign, BLCA_surv_data, delimiter=',', **save_args)

compare_clusters(BLCA_CSN_NBS_cluster_assign, BLCA_CSN_nocc_cluster_assign, clusters)
