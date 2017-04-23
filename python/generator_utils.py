import pandas as pd
import numpy as np
import itertools

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