import numpy as np
import scipy.io as scio
import pandas as pd
import datetime
import os
import pickle
from loguru import logger
from math import sqrt


def calculate_sim(source_path, to_path):
    '''
    calculate item similarity
    :param source_path: item interaction record path
    :param to_path: similarity save path
    :return:
    '''
    data = open(source_path,'rb')
    data = pickle.load(data)
    print(len(data))
    sim = {}
    keys = list(data.keys())
    with open(to_path, 'ab') as f_write:
        for i in range(len(keys)-1):
            key_a,list_a = keys[i],data[keys[i]]
            if i%10000 == 0:
                logger.info('hhh:{}'.format(i))
            for k in range(i+1,len(keys)):
                key_b,list_b = keys[k],data[keys[k]]
                inter = [j for j in list_a if j in list_b]
                # union = list(set(list_a).union(set(list_b)))
                sim_ab = len(inter)/sqrt(len(list_a)*len(list_b))
                if sim_ab < 0.5:
                    continue
                if key_a not in sim.keys():
                    sim[key_a]={}
                sim[key_a][key_b]=sim_ab
            if len(sim) == 0:
                continue
            pickle.dump(sim,f_write)
            sim = {}
            logger.info(i)
if __name__ == '__main__':
    calculate_sim(source_path='../datasets/epinions/processed_data/item_purd_users.pickle',to_path='../datasets/epinions/processed_data/item_social_sim.pickle')








