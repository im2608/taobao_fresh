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
                buy_date_time = datetime.datetime.strptime(behavior_consecutive[1], "%Y-%m-%d %H").date()
                    if (buy_date_time == checking_date):
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

def get_feature_buy_view_ratio(user_buy_records, user_item_pairs):
    does_operated = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in user_buy_records or 
            item_id not in user_buy_records[user_id]):
            continue    
    return does_operated

def logisticRegression(user_buy_records):
    return 0