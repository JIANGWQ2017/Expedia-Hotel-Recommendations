import numpy as np
import pandas as pd
import sys,os


def get_batch_data(input_file):
    input = pd.read_csv(input_file, nrows=10000)
    return input


def process_input(input_file):
    batch_data = get_batch_data(input_file)

    # drop id-class feature
    batch_data.drop(['user_id', 'srch_destination_id'], axis=1)
    # tranform date-time feature
    batch_data['date_month'] = batch_data['date_time'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    batch_data['date_day'] = batch_data['date_time'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    batch_data['time_hour'] = batch_data['date_time'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))

    batch_data.drop(['date_time'], axis=1)

    batch_data['srch_ci_month'] = batch_data['srch_ci'].apply(lambda x: int(x.split('-')[1]))
    batch_data['srch_ci_day'] = batch_data['srch_ci'].apply(lambda x: int(x.split('-')[2]))

    batch_data['srch_co_month'] = batch_data['srch_co'].apply(lambda x: int(x.split('-')[1]))
    batch_data['srch_co_day'] = batch_data['srch_co'].apply(lambda x: int(x.split('-')[2]))

    batch_data.drop(['srch_co', 'srch_ci'])

    print(batch_data.iloc[0])
    #print(batch_data.groupby(['srch_destination_id', 'hotel_cluster'])['is_booking'].value_counts())



if __name__ == '__main__':
    input_file = os.path.join(sys.argv[1], sys.argv[2])
    process_input(input_file)
