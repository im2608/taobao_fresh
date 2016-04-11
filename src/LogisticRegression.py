from common import *
import numpy as np


# 得到用户某天对item 的所有操作
def get_behavior_by_date(user_records, checking_date,user_item_pair):
    user_id = user_item_pair[0]
    item_id = user_item_pair[1]
    if (user_id not in user_records or 
        item_id not in user_records[user_id]):
        return None

    behaviors = []
    for records in user_records[user_id][item_id]:
        for each_record in records:
            for behavior_consecutive in each_record:
                    if (behavior_consecutive[1] == checking_date):
                        behaviors.append(behavior_consecutive)
    return behaviors

#根据购物记录，检查user 是否在 checking_date 这一天对 item 有过 behavior type
def get_feature_buy_at_date(user_buy_records, users_behavior_patterns, behavior_type, checking_date, user_item_pairs):
    does_operated = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in user_buy_records or 
            item_id not in user_buy_records[user_id]):
            continue

        behaviors = get_behavior_by_date(user_buy_records, checking_date, user_item_pairs[index])
        for each_beahvior in behaviors:
            if (each_beahvior[1] == behavior_type):
                does_operated[index][0] = 1
                break

        if (does_operated[index][0] == 1):
            continue

        behaviors = get_behavior_by_date(users_behavior_patterns, checking_date, user_item_pairs[index])
        for each_beahvior in behaviors:
            if (each_beahvior[1] == behavior_type):
                does_operated[index][0] = 1
                break

    return does_operated

# 得到用户的 购买过商品数量/浏览过的数量
def get_feature_buy_view_ratio(user_buy_records, user_behavior_patterns, user_item_pairs):
    buy_view_ratio = dict()
    buy_view_ratio_list = []

    buy_view_ratio_list = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in user_buy_records or 
            item_id not in user_buy_records[user_id]):
            continue

        if (user_id in buy_view_ratio):
            buy_view_ratio_list[index][1] = buy_view_ratio[user_id]
            continue

        buy_count = 0
        for item_id, item_buy_records in user_buy_records[user_id].items():
            buy_count += len(item_buy_records)

        # 没有pattern， 所有的view 都转化成了buy
        if (user_id not in user_behavior_patterns):
            buy_view_ratio_list[index][1] = 1

        view_count = 0
        for item_id, item_patterns in user_behavior_patterns[user_id].items():
            view_count += len(item_patterns)

        buy_view_ratio[user_id] = buy_count / (buy_count + view_count)
        buy_view_ratio_list[index][1] = buy_view_ratio[user_id]

    return does_operated

# 商品热度 购买该商品的用户/总用户数
def get_feature_item_popularity(user_buy_records, user_behavior_patterns, user_item_pairs):
    item_popularity_dict = dict()
    item_popularity_list =  np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        if (item_id in item_popularity_dict):
            item_popularity_list[index][0] = item_popularity_dict[item_id]

        buy_item_user_cnt = 0
        for user_id, buy_records in user_buy_records.items():
            if (item_id in buy_records):
                buy_item_user_cnt += 1

        item_popularity_dict[item_id] = buy_item_user_cnt / 20000
        item_popularity_list[index] = item_popularity_dict[item_id]

    return item_popularity_list

# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
def taking_samples(user_buy_records, user_behavior_patterns, nag_per_pos):
    samples = []
    return samples

def logisticRegression(user_buy_records, user_behavior_patterns):
    samples = taking_samples(user_buy_records, user_behavior_patterns, 5)
    feature_cnt = 5
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))
    return 0