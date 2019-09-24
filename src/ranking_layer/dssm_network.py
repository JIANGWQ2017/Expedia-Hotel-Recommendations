import tensorflow as tf
from ranking_layer.params import *



def embedding_layer(fea_name, ids, fea_size, embedding_size, comb = 'mean'):
    with tf.variable_scope(fea_name):
        weights = tf.get_variable(fea_name + 'weights', [fea_size, embedding_size],
                                  initializer=tf.glorot_normal_initializer)
    return tf.nn.embedding_lookup_sparse(weights, tf.sparse_reorder(ids), sp_weights=None,
                                         partition_strategy='mod', combiner=comb)


def embedding_out(train_batch, feature_count, embedding_count):
    '''
    :param train_batch:
    :param feature_count: feature维度map
    :param embedding_count: embedding维度map
    :return: embedding layer 输出
    '''
    feature = []
    for fea in feature_count:
        feature.append(embedding_layer(fea, train_batch[fea], feature_count[fea], embedding_count[fea]))

    return tf.concat(feature, 1)



def weights_and_bias(layer1, layer2):
    weights = tf.get_variable('weights', [layer1, layer2], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable('biases', [layer2], initializer=tf.constant_initializer(0.0))
    return weights, biases


def full_connect_layer(input, layer1, layer2):
    weight, bias = weights_and_bias(layer1, layer2)
    return tf.add(tf.matmul(input, weight), bias)


def user_infer(train_batch):
    user_embedding_output = embedding_out(train_batch, user_feature_count, user_embedding_count)
    with tf.variable_scope('user_layer1'):
        user_layer1 = full_connect_layer(user_embedding_output, user_total_embed_count, user_hidden_layer[0])
        user_layer1_out = tf.nn.relu(user_layer1)

    with tf.variable_scope('user_layer2'):
        user_layer2 = full_connect_layer(user_layer1_out, user_hidden_layer[0], user_hidden_layer[1])
        user_layer2_out = tf.nn.relu(user_layer2)

    with tf.variable_scope('user_layer3'):
        user_layer3 = full_connect_layer(user_layer2_out, user_hidden_layer[1], user_hidden_layer[2])
        user_layer3_out = tf.nn.relu(user_layer3)

    return user_layer3_out


def item_infer(train_batch):
    item_embedding_output = embedding_out(train_batch, item_feature_count, item_embedding_count)
    with tf.variable_scope('item_layer1'):
        item_layer1 = full_connect_layer(item_embedding_output, item_total_embed_count, item_hidden_layer[0])
        item_layer1_out = tf.nn.relu(item_layer1)

    with tf.variable_scope('item_layer2'):
        item_layer2 = full_connect_layer(item_layer1_out, item_hidden_layer[0], item_hidden_layer[1])
        item_layer2_out = tf.nn.relu(item_layer2)

    with tf.variable_scope('item_layer3'):
        item_layer3 = full_connect_layer(item_layer2_out, item_hidden_layer[1], item_hidden_layer[2])
        item_layer3_out = tf.nn.relu(item_layer3)

    return item_layer3_out


def infer(train_batch):
    user_out = user_infer(train_batch)
    item_out = item_infer(train_batch)
    prod = tf.reduce_sum(tf.multiply(user_out, item_out), 1, True)
    return prod