import time
import tensorflow as tf
import numpy as np
from gcn.utils import construct_feed_dict

def get_binary_labels(arr):
    """Get binary labels from binary vector."""
    one = np.ones(len((arr)))
    other = one-arr
    two_col = np.array([arr,other])
    return two_col.T

def run(features, y_train, y_test, y_val, train_mask, test_mask, val_mask, num_supports, support, model_func, FLAGS, col_idx=None, verbose=True):
    """Run one session of GCN. If a 'col_idx' is given then gcn will be trained for binary labels for a selected topic."""
    if col_idx == None:
        y_train_tmp, y_test_tmp, y_val_tmp = y_train, y_test, y_val
    else:
        y_train_tmp = get_binary_labels(y_train[:,col_idx])
        y_test_tmp = get_binary_labels(y_test[:,col_idx])
        y_val_tmp = get_binary_labels(y_val[:,col_idx])
    
    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train_tmp.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    
    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    cost_val = []
    
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)
    
    # Train GCN model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train_tmp, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val_tmp, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        if epoch % 10 == 0:
            if verbose:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                      "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                      "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            if verbose:
                print("Early stopping...")
            break

    print("Optimization finished for col_idx=%i!" % (col_idx if col_idx != None else -1))

    if verbose:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    train_acc, val_acc = outs[2], acc
    
    # Test GCN model
    test_cost, test_acc, test_duration = evaluate(features, support, y_test_tmp, test_mask, placeholders)
    if verbose:
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    
    prediction = sess.run(model.predict(), feed_dict=feed_dict)
    return [train_acc, test_acc, val_acc], prediction