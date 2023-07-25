import os.path
import pickle
import random

import pandas as pd
import numpy as np
import scipy.io as scio
from loguru import logger

def id_map(df,path_to,name):
    '''
    map user and item ids
    :param df: original datasets
    :param path_to: save path
    :param name:
    :return:
    '''
    df.sort_values(by=name,inplace=True)
    user_ids = []
    for x in df.groupby(by=name):
        user_ids.append(x[0])
    map = {}
    for i in range(len(user_ids)):
        source = user_ids[i]
        map[source] = i
    save_file = open(path_to, 'wb')
    logger.info('start store')
    pickle.dump(map, save_file)
    save_file.close()


def data_preprocess(df,path_user,path_item):
    '''
    :param data:
    :return:
    '''
    map_user = open(path_user,'rb')
    map_user = pickle.load(map_user)
    map_item = open(path_item, 'rb')
    map_item = pickle.load(map_item)
    df.rename(columns={'user_id':'original_user_id','item_id':'original_item_id'},inplace=True)
    df['user_id'] = None
    logger.info('start transform user_id')
    for key,value in map_user.items():
        logger.info(key)
        df.loc[df['original_user_id'].isin([key]), 'user_id'] = int(value)
    # df['time'] = df['timestamp'].apply(lambda x:datetime.datetime.fromtimestamp(x))
    logger.info('start transform item_id')
    for key,value in map_item.items():
        logger.info(key)
        df.loc[df['original_item_id'].isin([key]), 'item_id'] = int(value)
    logger.info('start sort')
    df.sort_values('timestamp', inplace=True)
    # df.reset_index(inplace=True)
    df.drop_duplicates(subset=['user_id', 'item_id'], keep='last', inplace=True)
    logger.info('start store')
    df.to_csv('../datasets/ciao/ciao_with_rating_timestamp/rating_with_timestamp.csv',index=False)
    return df

def process_rating(df,path,path_to):

    rating = {}
    if 'user_purd_items' in path:
        for x in df.groupby(by='user_id'):
            rating[x[0]] = list(x[1]['rating'])
    else:
        for x in df.groupby(by='item_id'):
            rating[x[0]] = list(x[1]['rating'])
    save_file = open(path_to, 'wb')
    logger.info('start store')
    pickle.dump(rating, save_file)
    save_file.close()
def split_datasets(df,train_path,test_path):
    '''
    split train data and test data
    regard each user's 90% data as train data
    regard each user's 10% data as test data
    :param df:
    :return:
    '''
    train = []
    test = []
    for x in df.groupby(by='user_id'):
        each_train = x[1].sample(frac=0.9,replace=False,random_state=1)[['user_id','item_id','rating']]
        train.append(each_train)
        train_items = list(each_train['item_id'])
        items = list(x[1]['item_id'])
        test_items= list(set(items).difference(set(train_items)))
        each_test = x[1][x[1]['item_id'].isin(test_items)][['user_id','item_id','rating']]
        test.append(each_test)

    # train_path = '../datasets/epinions/processed_data/train.csv'
    train_df = pd.concat(train,axis=0,ignore_index=True)
    train_df.to_csv(train_path,index=False)
    logger.info('store train data, number is {}'.format(len(train_df)))

    # test_path = '../datasets/epinions/processed_data/test.csv'
    test_df = pd.concat(test, axis=0, ignore_index=True)
    test_df.to_csv(test_path, index=False)
    logger.info('store test data, number is {}'.format(len(test_df)))
def split_datasets_classification(df):
    '''
    split train data and test data
    regard each user's 80% data as train data
    regard each user's 20% data as test data
    :param df:
    :return:
    '''
    train = []
    test = []
    for x in df.groupby(by='user_id'):
        test_item = random.choice(list(x[1]['item_id']))
        each_test = x[1][x[1]['item_id'].isin([test_item])][['user_id', 'item_id', 'rating']]
        test.append(each_test)
        items = list(x[1]['item_id'])
        train_items= list(set(items).difference(set([test_item])))
        each_train = x[1][x[1]['item_id'].isin(train_items)][['user_id', 'item_id', 'rating']]
        train.append(each_train)

    train_path = '../datasets/epinion/train.csv'
    train_df = pd.concat(train,axis=0,ignore_index=True)
    train_df.to_csv(train_path,index=False)
    logger.info('store train data, number is {}'.format(len(train_df)))

    test_path = '../datasets/epinion/test.csv'
    test_df = pd.concat(test, axis=0, ignore_index=True)
    test_df.to_csv(test_path, index=False)
    logger.info('store test data, number is {}'.format(len(test)))
def data_to_list(path):
    '''

    :param path:
    :return: list(users),list(items).list(ratings)
    '''
    df = pd.read_csv(path)
    data_u = []
    data_v = []
    data_r = []
    for x in df.groupby(by='user_id'):
        users = list(x[1]['user_id'])
        items = list(x[1]['item_id'])
        ratings = list(x[1]['rating'])
        data_u.extend(users)
        data_v.extend(items)
        data_r.extend(ratings)
    return data_u,data_v,data_r
def process_user_social(path_source,path_map,to_path_1,to_path_2):
    '''
    process user's social network data
    :return:
    '''
    data_social_path = path_source
    data = scio.loadmat(data_social_path)
    df = pd.DataFrame(data=data['trust'],
                      columns=['user_id', 'follow_user_id'])
    map = open(path_map, 'rb')
    map = pickle.load(map)
    df.rename(columns={'user_id': 'original_user_id','follow_user_id':'original_follow_user_id'}, inplace=True)
    df['user_id'] = None
    df['follow_user_id'] = None
    logger.info('start transform')
    for key, value in map.items():
        logger.info(key)
        df.loc[df['original_user_id'].isin([key]), 'user_id'] = int(value)
        df.loc[df['original_follow_user_id'].isin([key]), 'follow_user_id'] = int(value)
    df.drop_duplicates(subset=['user_id', 'follow_user_id'], keep='last', inplace=True)
    df.dropna(axis=0,subset=['user_id','follow_user_id'],inplace=True)
    df.to_csv(to_path_1,index=False)
    user_social = {}
    for x in df.groupby(by='user_id'):
        user_social[x[0]] = list(x[1]['follow_user_id'])
    save_file = open(to_path_2, 'wb')
    print('start store')
    pickle.dump(user_social, save_file)
    save_file.close()
    # process_item_user_interactions(df)
def process_user_r(df):
    ui_interaction = {}
    for x in df.groupby(by='user_id'):
        if len(list(x[1][x[1]['rating']>=3]['item_id'])) == 0:
            continue
        ui_interaction[x[0]] = list(x[1][x[1]['rating']>=3]['item_id'])
    save_file = open('../datasets/ciao/user_r.pickle', 'wb')
    print('start store')
    pickle.dump(ui_interaction, save_file)
    save_file.close()
def process_item_r(df):
    iu_interaction = {}
    for x in df.groupby(by='item_id'):
        if len(list(x[1][x[1]['rating']>=3]['user_id'])) == 0:
            continue
        iu_interaction[x[0]] = list(x[1][x[1]['rating']>=3]['user_id'])
    save_file = open('../datasets/ciao/item_r.pickle', 'wb')
    print('start store')
    pickle.dump(iu_interaction, save_file)
    save_file.close()


def process_item_user_interactions(df,to_path):
    '''
    user set (in training set) who have interacted with the item
    :return:
    '''
    iu_interaction = {}
    for x in df.groupby(by='item_id'):
        iu_interaction[x[0]] = list(x[1]['user_id'])
    save_file = open(to_path, 'wb')
    print('start store')
    pickle.dump(iu_interaction, save_file)
    save_file.close()
def process_user_item_interactions(df,to_path):
    '''
    user's purchased history
    :param df:
    :return:
    '''
    ui_interaction = {}
    for x in df.groupby(by='user_id'):
        ui_interaction[x[0]] = list(x[1]['item_id'])
    save_file = open(to_path, 'wb')
    print('start store')
    pickle.dump(ui_interaction, save_file)
    save_file.close()

def filter_g_k_one(data,k=10,u_name='user_id',i_name='item_id',y_name='rating'):
    '''
    delete the records that user and item interactions lower than k
    '''
    item_group = data.groupby(i_name).agg({y_name:'count'}) #every item has the number of ratings
    item_g10 = item_group[item_group[y_name]>=k].index
    data_new = data[data[i_name].isin(item_g10)]

    user_group = data_new.groupby(u_name).agg({y_name: 'count'})  # every item has the number of ratings
    user_g10 = user_group[user_group[y_name] >= k].index
    data_new_1 = data_new[data_new[u_name].isin(user_g10)]
    return data_new_1

def map_user_item_id(data,user_path,item_path):
    '''
    map original user id and item id into consequent number
    '''
    data = data[['user_id','item_id','rating','timestamp']]
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_to_id = dict(zip(list(user),list(np.arange(user.shape[0]))))
    save_file = open(user_path, 'wb')
    print('start store')
    pickle.dump(user_to_id, save_file)
    save_file.close()
    item_to_id = dict(zip(list(item), list(np.arange(item.shape[0]))))
    save_file = open(item_path, 'wb')
    print('start store')
    pickle.dump(item_to_id, save_file)
    save_file.close()
    print('user num:',user.shape)
    print('item num',item.shape)
    data.rename(columns={'user_id': 'original_user_id', 'item_id': 'original_item_id'}, inplace=True)
    data['user_id'] = data['original_user_id'].map(user_to_id)
    data['item_id'] = data['original_item_id'].map(item_to_id)
    return data
def get_train_instances():
    data = pd.read_csv('../datasets/epinion/epinion_rating_with_timestamp.csv')
    user_ids = []
    item_ids = []
    labels = []
    num_negatives = 4
    df = pd.read_csv('../datasets/epinion/train.csv')
    df = df[['user_id','item_id','rating']]
    # num_users = df['user_id'].unique()
    num_items = len(data['item_id'].unique())
    logger.info('start create train pos and neg samples')
    for index, row in df.iterrows():
        if index % 1000 ==0:
            logger.info(index)
        uid = row['user_id']
        iid = row['item_id']
        user_ids.append(uid)
        item_ids.append(iid)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while len(data[(data['user_id'].isin([uid]))&(data['item_id'].isin([j]))]) > 0:
                j = np.random.randint(num_items)
            user_ids.append(uid)
            item_ids.append(j)
            labels.append(0)
    return user_ids, item_ids, labels


if __name__ == '__main__':
    processed_data_path = '../datasets/epinions/processed_data/'
    #--------first step------------
    # data_path = '../datasets/epinions/rating_with_timestamp.mat'
    # data = scio.loadmat(data_path)
    # df = pd.DataFrame(data=data['rating_with_timestamp'],
    #                   columns=['user_id', 'item_id', 'category_id', 'rating', 'helpfulness', 'timestamp'])
    # data = filter_g_k_one(df,k=5)
    # if not os.path.exists(processed_data_path):
    #     os.makedirs(processed_data_path)
    # data = map_user_item_id(data,user_path=processed_data_path+'user_id_map.pickle',item_path=processed_data_path+'item_id_map.pickle')
    # data.to_csv(processed_data_path+'epinions_rating_with_timestamp.csv', index=False)
    #------------second step: to save time, directly read file-----------
    df = pd.read_csv(processed_data_path+'epinions_rating_with_timestamp.csv')
    process_user_item_interactions(df,to_path=processed_data_path+'user_purd_items.pickle')
    process_item_user_interactions(df,to_path=processed_data_path+'item_purd_users.pickle')
    process_user_social(path_source='../datasets/epinions/trust.mat',path_map=processed_data_path+'user_id_map.pickle',to_path_1=processed_data_path+'user_trust.csv',to_path_2=processed_data_path+'user_social.pickle')
    process_rating(df,path=processed_data_path+'user_purd_items.pickle',path_to=processed_data_path+'user_item_rating.pickle')
    process_rating(df,path=processed_data_path+'item_purd_users.pickle',path_to = processed_data_path+'item_user_rating.pickle')
    split_datasets(df,train_path=processed_data_path+'train.csv',test_path=processed_data_path+'test.csv')




