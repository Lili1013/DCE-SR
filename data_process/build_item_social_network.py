import pickle
import pandas as pd
from loguru import logger

def social_sim_to_csv(data,path_to):
    logger.info('start read')
    i = 1
    with open(path_to,'a') as f:
        while True:
            logger.info(i)
            try:
                each_record = pickle.load(data)
                item_id = list(each_record.keys())[0]
                follow_item_ids = []
                for key,value in list(each_record.values())[0].items():
                    if value >= 0.7:
                        follow_item_ids.append(key)
                # follow_item_ids = list(list(each_record.values())[0].keys())
                if i == 1:
                    f.write('item_id,follow_item_id\n')
                if len(follow_item_ids) > 0:
                    for each in follow_item_ids:
                        f.write(str(item_id)+','+str(each)+'\n')
                        f.write(str(each)+','+str(item_id)+'\n')
            except:
                break
            i += 1

def build_social_network(source_path,to_path):
    '''
    build item relation network according to item's similarity
    :param source_path:
    :param to_path:
    :return:
    '''
    df = pd.read_csv(source_path, delimiter=',')
    df.columns = ['item_id', 'follow_item_id']
    social_dict = {}
    logger.info('start group')
    for x in df.groupby(by='item_id'):
        logger.info(x[0])
        social_dict[x[0]-1] = [i-1 for i in list(x[1]['follow_item_id'])]#reduce 1 is to map item_id->item_id:item_id-1
    save_file = open(to_path, 'wb')
    print('start store')
    pickle.dump(social_dict, save_file)
    save_file.close()

if __name__ == '__main__':
    data_path = '../datasets/epinions/processed_data/item_social_sim.pickle'
    data = open(data_path, 'rb')
    social_sim_to_csv(data, path_to='../datasets/epinions/processed_data/item_trust.txt')#filter sim < 0.7
    build_social_network(source_path='../datasets/epinions/processed_data/item_trust.txt',to_path='../datasets/epinions/processed_data/item_social.pickle')

