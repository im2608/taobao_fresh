from common import *
import numpy as np
from LR_common import *


# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
def feature_item_popularity(behavior_type, item_popularity_dict, user_item_pairs):
    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (item_id in item_popularity_dict and 
            behavior_type in item_popularity_dict[item_id]):
            item_popularity_list[index][0] = item_popularity_dict[item_id][behavior_type]
        else:
            item_popularity_list[index][0] = 0

    return item_popularity_list

# 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数
# 返回 4 个特征
def feature_beahvior_cnt_on_item(begin_date, checking_date, user_item_pairs):
    logging.info("feature_beahvior_cnt_on_item(%s, %s)" % (begin_date, checking_date))
    item_behavior_cnt_dict = dict()
    item_behavior_cnt_list = np.zeros((len(user_item_pairs), 4))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in item_behavior_cnt_dict):
            item_behavior_cnt_list[index] = item_behavior_cnt_dict[item_id]
            continue

        behavior_cnt = [0,0,0,0]
        for user_id, item_buy_records in g_user_buy_transection.items():
            if (item_id not in item_buy_records):
                continue

            for each_record in item_buy_records[item_id]:
                for each_behavior in each_record:
                    if (begin_date <= each_behavior[1].date() and each_behavior[1].date() < checking_date):
                        behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        for user_id, item_opt_records in g_user_behavior_patten.items():
            if (item_id not in item_opt_records):
                continue

            for each_record in item_opt_records[item_id]:
                for each_behavior in each_record:
                    if (begin_date <= each_behavior[1].date() and each_behavior[1].date() < checking_date):
                        behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        item_behavior_cnt_list[index] = behavior_cnt
        item_behavior_cnt_dict[item_id] = behavior_cnt

        logging.info("item %s, behavior %s from %s to %s" % (item_id, behavior_cnt, begin_date, checking_date))

    logging.info("leaving feature_beahvior_cnt_on_item")
    return item_behavior_cnt_list

# item 第一次behavior 距离checking date 的天数, 返回 4 个特征
def feature_days_from_1st_behavior(checking_date, user_item_pairs):
    days_from_1st_dict = dict()
    days_from_1st_list = np.zeros((len(user_item_pairs), 4))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_1st_dict):
            days_from_1st_list[index] = days_from_1st_dict[item_id]
            continue

        days_from_1st_behavior = [checking_date for x in range(4)]

        for user_id, item_buy_records in g_user_buy_transection.items():
            if (item_id not in item_buy_records):
                continue

            for each_record in item_buy_records[item_id]:
                for each_behavior in each_record:
                    if (each_behavior[1].date() < days_from_1st_behavior[each_behavior[0] - 1]):
                        days_from_1st_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        for user_id, item_opt_records in g_user_behavior_patten.items():
            if (item_id not in item_opt_records):
                continue

            for each_record in item_opt_records[item_id]:
                for each_behavior in each_record:
                    if (each_behavior[1].date() < days_from_1st_behavior[each_behavior[0] - 1]):
                        days_from_1st_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        days_from_1st_behavior = list(map(lambda x: (checking_date - x).days, days_from_1st_behavior))
        days_from_1st_list[index] = days_from_1st_behavior
        days_from_1st_dict[item_id] = days_from_1st_behavior
        logging.info("item %s days from 1st behavior to %s: %s " % (item_id, checking_date, days_from_1st_behavior))

    return days_from_1st_list


# item 最后一次behavior 距离checking date 的天数, 返回 4 个特征
def feature_days_from_last_behavior(checking_date, user_item_pairs):
    days_from_last_dict = dict()
    days_from_last_list = np.zeros((len(user_item_pairs), 4))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_last_dict):
            days_from_last_list[index] = days_from_last_dict[item_id]
            continue

        days_from_last_behavior = [None for x in range(4)]

        for user_id, item_buy_records in g_user_buy_transection.items():
            if (item_id not in item_buy_records):
                continue

            for each_record in item_buy_records[item_id]:
                for each_behavior in each_record:
                    if (days_from_last_behavior[each_behavior[0] - 1] == None):
                        days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()
                    else:
                        if (each_behavior[1].date() > days_from_last_behavior[each_behavior[0] - 1] and \
                            each_behavior[1].date() < checking_date):
                            days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        for user_id, item_opt_records in g_user_behavior_patten.items():
            if (item_id not in item_opt_records):
                continue

            for each_record in item_opt_records[item_id]:
                for each_behavior in each_record:
                    if (days_from_last_behavior[each_behavior[0] - 1] == None):
                        days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()
                    else:
                        if (each_behavior[1].date() > days_from_last_behavior[each_behavior[0] - 1] and \
                            each_behavior[1].date() < checking_date):
                            days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        # days_from_last_behavior = list(map(lambda x: (checking_date - x).days, days_from_last_behavior))
        for index in range(len(days_from_last_behavior)):
            if (days_from_last_behavior[index] != None):
                days_from_last_behavior[index] = (checking_date - days_from_last_behavior[index]).days
            else:
                days_from_last_behavior[index] = 0

        days_from_last_list[index] = days_from_last_behavior
        days_from_last_dict[item_id] = days_from_last_behavior
        logging.info("item %s days from last behavior to %s: %s " % (item_id, checking_date, days_from_last_behavior))

    return days_from_last_list
