from tqdm import tqdm
import random


def get_recommend(recommend_matrix, userId, list, rev):
    ratings = recommend_matrix[userId][list]
    # temp_list = zip(list,ratings)
    return [item[0] for item in sorted(zip(list, ratings), key=lambda x: x[1], reverse=rev)]


def rec(feature_columns, train, test, N, pred_rating_matrix):
    item_num = feature_columns[1]['feat_num']
    all_items = set(range(1, item_num))
    hits = {}
    for n in N:
        hits.setdefault(n, 0)
    test_count = 0

    # 统计结果
    # user_num = dataset.get_feature()[0]['feat_num']
    for user_id in tqdm(test):
        if user_id not in train.keys():
            continue
        train_items = train[user_id]
        test_items = test[user_id]
        other_items = all_items - train_items.union(test_items)
        for idx in test_items:
            random_items = random.sample(other_items, 200)
            random_items.append(idx)
            # =================================获取排序后二级索引中的电影号=========================================
            sort_values = get_recommend(pred_rating_matrix, user_id, random_items, rev=True)
            for n in N:
                hits[n] += int(idx in sort_values[:n])
        test_count += len(test_items)
    for n in N:
        recall = hits[n] / (1.0 * test_count)
        print('N:%d\trecall=%.6f\t' % (n, recall))


def hr(feature_columns, train, test, N, pred_rating_matrix):
    item_num = feature_columns[1]['feat_num']
    all_items = set(range(1, item_num))
    hits = {}
    for n in N:
        hits.setdefault(n, 0)
    test_count = 0

    # 统计结果
    # user_num = dataset.get_feature()[0]['feat_num']
    for user_id in tqdm(test):
        if user_id not in train.keys():
            continue
        train_items = train[user_id]
        test_items = test[user_id]
        other_items = all_items - train_items.union(test_items)
        random_items = random.sample(other_items, 200)
        random_items += list(test_items)
        # =================================获取排序后二级索引中的电影号=========================================
        sort_values = get_recommend(pred_rating_matrix, user_id, random_items, rev=True)
        for n in N:
            hits[n] += 1 if len([item for item in test_items if item in sort_values[:n]]) else 0
        test_count += len(test_items)
    for n in N:
        recall = hits[n] / (1.0 * len(train.keys()))
        print('N:%d\thr=%.6f\t' % (n, recall))
