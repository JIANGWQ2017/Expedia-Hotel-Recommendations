import numpy as np
import pandas as pd
import tensorflow as tf
from params import *
from dssm_network import *

def enumerate_value(input_file, feature_name):
    '''
    计算需要one-hot特征的所有可能取值
    :param input_file:  样本输入
    :param feature_name:  特证名
    :return:
    '''
    #feature_value_dict = {}
    value_set = set()
    reader = pd.read_csv(input_file, chunksize=CHUNK_SIZE)
    chunck_count = 1
    for chunck in reader:
        chunck = process_input(chunck)
        value_list = chunck[feature_name].drop_duplicates().to_list()
        for v in value_list:
            value_set.add(v)
        print('Processed %d samples' % (chunck_count*CHUNK_SIZE))
        chunck_count += 1
    res = list(value_set)
    res.sort()
    print(res)
    print(min(res), max(res))

def process_input(batch_data):
    #batch_data = pd.read_csv(input_file, nrows=10)

    # drop 掉id类特征
    batch_data.drop(['user_id'], axis=1, inplace=True)
    # 从date time 中提取 月份，日期 和时间
    batch_data['date_month'] = batch_data['date_time'].apply(lambda x: int(x.split(' ')[0].split('-')[1])-1)
    batch_data['date_day'] = batch_data['date_time'].apply(lambda x: int(x.split(' ')[0].split('-')[2])-1)
    batch_data['time_hour'] = batch_data['date_time'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))
    batch_data.drop(['date_time'], axis=1, inplace=True)

    # # 从checkin checkout 日期算出住宿时间
    # batch_data['srch_ci'] = pd.to_datetime(batch_data['srch_ci'])
    # batch_data['srch_co'] = pd.to_datetime(batch_data['srch_co'])
    # batch_data['stay_days'] = (batch_data['srch_co'] - batch_data['srch_ci']).apply(lambda x: x.days)
    # batch_data.drop(['srch_co', 'srch_ci'], axis=1, inplace=True)

    batch_data.drop(['orig_destination_distance'], axis=1, inplace=True)


    #print(batch_data.groupby(['srch_destination_id', 'hotel_cluster'])['is_booking'].value_counts())
    #print(batch_data.iloc[0])

    return batch_data


def transform(batch_data, type, feature_name, depth):
    '''
    把输入的dataframe格式数据 transform成tensor
    :param batch_data:
    :param type: 区分user侧特征 和 item侧特征
    :param feature_name: 特征名
    :param depth:  可能取值的个数
    :return: 处理好的tensor
    '''
    indices = batch_data[feature_name].to_list()
    with tf.variable_scope(type):
            res = tf.constant(indices)
            tensor = tf.one_hot(res, depth, name=feature_name)
            tensor = tf.cast(tensor, tf.int32)
            return tensor


def make_example(user, item, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'user': tf.train.Feature(bytes_list=tf.train.BytesList(value=[user])),
        'item': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))


def write_tfrecord(labels, user, item, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for label, user, item in zip(labels, user, item):
        #label = label.astype(tf.float32)
        ex = make_example(user.tobytes(), item.tobytes(), label.tobytes())
        writer.write(ex.SerializeToString())
    writer.close()



def read_tfrecord(filename, item_shape, user_shape):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'user': tf.FixedLenFeature([], tf.string),
            'item': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    user = tf.decode_raw(features['user'], tf.float32)
    item = tf.decode_raw(features['item'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    user = tf.reshape(user, [user_shape[1]])
    item = tf.reshape(item, [item_shape[1]])
    label = tf.reshape(label, [1])

    user, item, label = tf.train.batch([user, item, label], batch_size=10)
    return user, item, label




