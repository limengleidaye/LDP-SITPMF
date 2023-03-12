import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer


def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_explicit_dataset(train_set, user_profile, latent_dim=4, test_size=0.3):
    data_df = pd.read_csv(train_set, sep=',', engine='python')
    user_df = pd.read_csv(user_profile, sep='|', names=['UserId', 'Age', 'Gender', 'Occupation', 'ZipCode'],
                          engine='python')
    user_df = user_df.drop(['ZipCode'], axis=1)
    user_num, item_num = user_df['UserId'].max(), data_df['MovieId'].max()
    feature_coloums = [sparseFeature('user_id', user_num, latent_dim), sparseFeature('item_id', item_num, latent_dim)]

    # 用户、物品id从0开始，将数据集中的用户id、物品id转换
    user2user_encoded = {x: i for i, x in enumerate(range(1, user_num + 1))}
    item2item_encoded = {x: i for i, x in enumerate(range(1, item_num + 1))}
    data_df['UserId'] = data_df['UserId'].map(user2user_encoded)
    data_df['MovieId'] = data_df['MovieId'].map(item2item_encoded)
    user_df['UserId'] = user_df['UserId'].map(user2user_encoded)
    # 用户profile标签信息进行encoding
    est = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform')
    user_df['Age'] = est.fit_transform(user_df['Age'].values.reshape(-1, 1)).astype('int32')
    le = LabelEncoder()
    user_df['Gender'] = le.fit_transform(user_df['Gender'])
    user_df['Occupation'] = le.fit_transform(user_df['Occupation'])

    data_df['random'] = np.random.random(len(data_df))
    test_df = data_df[data_df['random'] < test_size]
    test_df = test_df.reset_index()
    train_df = data_df.sample(frac=1.).drop(labels=test_df['index'])
    train_df = train_df.reset_index(drop=True).drop(columns=['random'], axis=1)
    test_df = test_df.sample(frac=1.).drop(['index', 'random'], axis=1).reset_index(drop=True)
    return feature_coloums, train_df.values.astype('int32'), test_df.values.astype(
        'int32'), user_df.values.tolist()


def create_transition_pattern_dataset(data, item_num):
    tp_matrix = np.zeros((item_num, item_num))
    data_df = pd.DataFrame(data, columns=['UserId', 'MovieId', 'Rating', 'TimeStamp'])
    for _, group in data_df.sort_values(['TimeStamp']).groupby(by=['UserId']):
        pre_item = group.iloc[0]['MovieId']
        index = 0
        for _, row in group.iterrows():
            if index == 0:
                index += 1
                continue
            current_item = row['MovieId']
            tp_matrix[pre_item][current_item] += 1
            pre_item = current_item
            index += 1
    tp_matrix = 1 / (1 + np.exp(-tp_matrix)) + 1
    train_x = []
    train_y = []
    for index, value in np.ndenumerate(tp_matrix > 1.5):
        if value == True:
            x, y = index[0], index[1]
            train_x.append([x, y])
            train_y.append(tp_matrix[x, y])
    return [train_x, train_y]
