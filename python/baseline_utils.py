import sys
import pickle as pkl
import tensorflow as tf
import numpy as np
from collections import Counter
from gcn.metrics import masked_accuracy

### Random predictor ###

def random_pred(y_arr, unknown_enabled=False):
    """Random predictor. Predict random labels from a finite set with uniform distribution."""
    new_array = [tuple(row) for row in y_arr]
    c = Counter(new_array)
    if not unknown_enabled:
        dim = y_arr.shape[1]
        zero_label = tuple(0.0 for i in range(dim))
        del c[zero_label]
    unique_labels = list(c.keys())
    #print("Number of unique labels: %i" %  len(unique_labels))
    rnd_labels = []
    # TODO: get random vector right away
    for i in range(len(y_arr)):
        label_idx = np.random.randint(0,len(unique_labels))
        rnd_label = list(unique_labels[label_idx])
        rnd_labels.append(rnd_label)
    return np.array(rnd_labels)


### Frequency-weighted random predictor ###

def weighted_random_pred(label_bin_file_path,size):
    """Frequency weighted random predictor. Predict random labels from a finite set based on their frequencies in the provided binary file.."""
    with open(label_bin_file_path, 'rb') as f:
        if sys.version_info > (3, 0):
            labels_arr = pkl.load(f, encoding='latin1')
        else:
            labels_arr = pkl.load(f)
    new_array = [tuple(row) for row in labels_arr]
    ordered_labels, w_ranges = get_weighted_ranges(new_array)
    rnd_vals = [ordered_labels[find_label_idx(w_ranges)] for i in range(size)]
    return np.array(rnd_vals)
        
def find_label_idx(weighted_ranges):
    rnd_num = np.random.random()
    for i in range(len(weighted_ranges)):
        lower = weighted_ranges[i][0]
        upper = weighted_ranges[i][1]
        if rnd_num >= lower and rnd_num < upper:
            break
    return i

def get_weighted_ranges(label_array):
    c = Counter(label_array)
    total = sum(c.values())
    freq_intervals, rel_freq_intervals = [], []
    lower, upper = 0, 0
    ordered_labels = []
    for pair in c.most_common():
        label, value = pair
        lower = upper
        upper = lower + value
        ordered_labels.append(label)
        freq_intervals.append((lower,upper))
        rel_freq_intervals.append((float(lower)/total,float(upper)/total))
    #print(freq_intervals)
    #print(rel_freq_intervals)
    return ordered_labels, rel_freq_intervals


### Baseline interface ###

def baseline_predict(y_train, y_test, y_val, train_mask, test_mask, val_mask, bin_file_path=None):
    """Get baseline random predictions. If 'bin_file_path' is used then the predictor will take into account the frequencies of the labels in the file."""
    sess = tf.Session()
    acc_vector = []
    for t in [(y_train,train_mask),(y_val,val_mask),(y_test,test_mask)]:
        if bin_file_path == None:
            y_rnd = random_pred(t[0])
        else:
            y_rnd = weighted_random_pred(bin_file_path, len(t[0]))
        acc = sess.run(masked_accuracy(y_rnd,t[0],t[1]))
        acc_vector.append(acc)
    return acc_vector
