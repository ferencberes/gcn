import pandas as pd
import numpy as np
import pickle as pkl
import itertools, os
from scipy import sparse
from collections import defaultdict

### TOPIC LABELS ###

def get_experiment_dir(param_helper):
    return "k%i_t%i_r%.2f" % tuple(param_helper.get(param) for param in ["top_k","time_frame","cut_ratio"])

def get_label(target_list, keys):
    label = ''
    for key in keys:
        label += '1' if key in target_list else '0'
    return label

def generate_labels(df, top_authorities):
    targets_for_sources = df.groupby(by=["src"])["trg"].unique()
    sources_with_labels_df = pd.DataFrame(targets_for_sources).reset_index()
    sources_with_labels_df["label"] = sources_with_labels_df["trg"].apply(lambda x : get_label(x, keys=top_authorities))
    print("Number of nodes: %i" % len(sources_with_labels_df))
    print("Number of unique labels: %i" % len(sources_with_labels_df["label"].unique()))
    return sources_with_labels_df


### TOPIC NETWORK ###

def add_edges_to_graph(mentions_df,G,trg_id,time_frame):
    filtered_for_trg = mentions_df[mentions_df["trg"] == trg_id]
    filtered_for_trg = filtered_for_trg.reset_index()[["time","src"]]
    min_time = filtered_for_trg["time"].min()
    idx_set = list(filtered_for_trg[filtered_for_trg["time"] < min_time + time_frame].index)
    edge_set = get_node_pairs(idx_set,filtered_for_trg,all_pair=True)
    G.add_edges_from(edge_set, weight=trg_id)
    #print(idx_set)
    #print(edge_set)
    for i in range(len(idx_set),len(filtered_for_trg)):
        current_time = filtered_for_trg.ix[i]["time"]
        low_idx = len(idx_set)
        for j in range(len(idx_set)):
            if filtered_for_trg.ix[idx_set[j]]["time"] > current_time - time_frame:
                low_idx = j
                break
        idx_set = idx_set[low_idx:] + [i] # update active indices
        edge_set = get_node_pairs(idx_set,filtered_for_trg)
        G.add_edges_from(edge_set, weight=trg_id)
        #print(idx_set)
        #print(edge_set)
    print("Edges were added for trg=%i" % trg_id)

def get_node_pairs(idx_list, filtered_for_trg, all_pair=False):
    node_list = list(filtered_for_trg.ix[idx_list]["src"])
    if len(idx_list) > 1:
        if all_pair:
            return list(itertools.combinations(node_list, 2))
        else:
            return list(zip(node_list[:-1],np.ones(len(node_list)-1,dtype="i")*node_list[-1]))
    else:
        return []   


### TRAIN-TEST SPLIT ###

def get_train_test(df, split_type="random", train_ratio=0.6):
    """Split dataframe into train and test sets. The 'split_split' can be 'temporal' or 'random' with predefined 'train_ratio'"""
    if split_type == "temporal":
        min_time, max_time = df["time"].min(), df["time"].max()
        print((max_time-min_time) // 86400)
        cut_time = min_time + (max_time-min_time) * train_ratio
        train = df[df["time"] <= cut_time]
        test = df[df["time"] > cut_time]
        print("Temporal train-test split was executed!")
    elif split_type == "random":
        msk = np.random.rand(len(df)) < train_ratio
        train, test = df[msk], df[~msk]
        print("Random train-test split was executed!")
    else:
        raise RuntimeError("Invalid split_type '%s'" % split_type)
    return train, test


### EXPORT BINARY FILES ###

def export_test_indices(df, preproc_dir_prefix, dataset_id, label_types=["blended","onehot"]):
    for l_type in label_types:
        output_dir = "%s/%s" % (preproc_dir_prefix, l_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt("%s/ind.%s.test.index" % (output_dir,dataset_id),df.index,fmt="%i")
        
def export_edges(df, giant_graph, preproc_dir_prefix, dataset_id, label_types=["blended","onehot"]):
    n2i_map = dict(zip(df["src"],df.index))
    edge_list_map = defaultdict(list)
    for s,t in giant_graph.edges():
        edge_list_map[n2i_map[s]].append(n2i_map[t])
    for l_type in label_types:
        output_dir = "%s/%s" % (preproc_dir_prefix, l_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open("%s/ind.%s.graph" % (output_dir,dataset_id),"wb+") as outfile:
            pkl.dump(edge_list_map, outfile)
            
def str2arr(label):
    return [int(char) for char in str(label)]

def str2onehot(label,unique_labels,index_pos_map):
    out = np.zeros(len(unique_labels),dtype="i")
    out[index_pos_map[label]] = 1
    return out

def export_labels(export_tuples, df, preproc_dir_prefix, dataset_id, label_types=["blended","onehot"]):
    unique_labels = list(df["label"].unique())
    index_pos_map = dict(zip(unique_labels,range(len(unique_labels))))
    for l_type in label_types:
        output_dir = "%s/%s" % (preproc_dir_prefix, l_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for item in export_tuples:
            label_arr = []
            for index, row in item[0].iterrows():
                if l_type == "blended":
                    label_arr.append(str2arr(row["label"]))
                elif l_type == "onehot":
                    label_arr.append(str2onehot(row["label"],unique_labels,index_pos_map))
                else:
                    raise RuntimeError("Invalid label_type '%s'" % l_type)
            with open("%s/ind.%s.%sy" % (output_dir,dataset_id,item[1]),"wb+") as outfile:
                pkl.dump(label_arr, outfile)

def export_features(export_tuples, preproc_dir_prefix, dataset_id, feature_set=["frequency"], label_types=["blended","onehot"]):
    for l_type in label_types:
        output_dir = "%s/%s" % (preproc_dir_prefix, l_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)    
        for item in export_tuples:
            if len(feature_set) == 1:
                num_nodes = len(item[0])
                row = range(num_nodes)
                col = np.zeros(num_nodes)
                data = item[0][feature_set[0]].as_matrix()
                coord_sparse = sparse.csr_matrix( (data,(row,col)), shape=(num_nodes,1))
            elif len(feature_set) > 1:
                coord_sparse = sparse.csr_matrix(item[0][feature_set].as_matrix())
            else:
                break
            with open("%s/ind.%s.%sx" % (output_dir,dataset_id,item[1]),"wb+") as outfile:
                pkl.dump(coord_sparse, outfile)