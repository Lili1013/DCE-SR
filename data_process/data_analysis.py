import pickle
import pandas as pd

def user_item_interaction_density():

    dir_data = '../datasets/epinions/epinion_with_rating_timestamp/'

    user_pur_path = open(dir_data + "user_pur_items.pickle", 'rb')
    history_u_lists = pickle.load(user_pur_path)
    item_purd_path = open(dir_data + "item_purd_users.pickle", 'rb')
    history_v_lists = pickle.load(item_purd_path)

    user_x = []
    user_y = []
    for key,value in history_u_lists.items():
        if len(value) < 5:
            user_x.append(key)
            user_y.append(len(value))
    print(len(user_x)/len(history_u_lists))
    item_x = []
    item_y = []
    for key,value in history_v_lists.items():
        if len(value) < 5:
            item_x.append(key)
            item_y.append(len(value))
    print(len(item_x)/len(history_v_lists))

if __name__ == '__main__':
    path = '../datasets/epinions/epinion_with_rating_timestamp/test.csv'
    df = pd.read_csv(path)
    item_id_inter={}
    item_id = []
    inter_len = []
    for x in df.groupby(by='item_id'):
        item_id_inter[x[0]] = list(x[1]['user_id'])
        item_id.append(x[0])
        inter_len.append(len(x[1]))
    df_tmp = pd.DataFrame({
        'item_id':item_id,
        'interaction_number':inter_len
    })
    inter_same_group = {}
    for x in df_tmp.groupby(by='interaction_number'):
        inter_same_group[x[0]] = list(x[1]['item_id'])

    print('gg')