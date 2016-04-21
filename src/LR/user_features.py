from common import *
import numpy as np
from LR_common import *


####################################################################################################
################################  数据集中，用户的特征  ##############################################
####################################################################################################

#用户一共购买过多少商品
def feature_how_many_buy(user_item_pairs):
    logging.info("entered feature_how_many_buy")
    how_many_buy_list = np.zeros((len(user_item_pairs), 1))
    how_many_buy_dict = dict()

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in how_many_buy_dict):
            how_many_buy_list[index] = how_many_buy_dict[user_id]
            continue
        buy_cnt = 0

        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            buy_cnt += len(item_buy_records)

        how_many_buy_list[index] = buy_cnt
        how_many_buy_dict[user_id] = buy_cnt
        logging.info("user %s bought %d items" % (user_id, buy_cnt))

    logging.info("leaving feature_how_many_buy")

    return how_many_buy_list


#在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车
def feature_how_many_behavior(begin_date, end_date, need_ratio, user_item_pairs):
    logging.info("entered feature_how_many_behavior(%s, %s, %d)" % (begin_date, end_date, need_ratio))
    features = 4
    if (need_ratio):
        features = 7

    how_many_behavior_list = np.zeros((len(user_item_pairs), features))
    how_many_behavior_dict = dict()

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in how_many_behavior_dict):
            how_many_behavior_list[index] = how_many_behavior_dict[user_id]

        #if feature == 7, 前4 个为浏览，收藏，购物车, 购买的数量， 后3个为比例
        behavior_cnt = [0 for x in range(features)]

        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                for each_behavior in each_record:
                    if (each_behavior[1].date() >= begin_date and \
                        each_behavior[1].date() < end_date):
                        behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        for item_id, item_opt_records in g_user_behavior_patten[user_id].items():
            for each_record in item_opt_records:
                for each_behavior in each_record:
                    if (each_behavior[1].date() >= begin_date and \
                        each_behavior[1].date() < end_date):
                        behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        if (need_ratio):
            for behavior_index in range(3):
                if (behavior_cnt[behavior_index] != 0):
                    behavior_cnt[behavior_index + 4] = round(behavior_cnt[3] / behavior_cnt[behavior_index], 4)

        how_many_behavior_list[index] = behavior_cnt
        how_many_behavior_dict[user_id] = behavior_cnt
        logging.info("behavior count %s %s (%s -- %s)" % (user_id, behavior_cnt, begin_date, end_date))

    logging.info("leaving feature_how_many_behavior")
    return how_many_behavior_list

# 用户在 checking date（不包括） 之前每次购买日期距 checking date 的天数的平均值和方差
def feature_mean_days_between_buy_user(checking_date, user_item_pairs):
    logging.info("entered feature_mean_days_between_buy_user(%s)" % checking_date)
    mean_days_between_buy_dict = dict()
    mean_days_between_list = np.zeros((len(user_item_pairs), 2))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in mean_days_between_buy_dict):
            mean_days_between_list[index] = mean_days_between_buy_dict[user_id]
            continue

        days_to_checking_date = []
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                if (each_record[-1][1].date() < checking_date):
                    days_to_checking_date.append((checking_date - each_record[-1][1].date()).days)

        mean_vairance = [0, 0]
        if (len(days_to_checking_date) > 0):            
            mean_vairance[0] = np.round(np.mean(days_to_checking_date), 4)
            mean_vairance[1] = np.round(np.var(days_to_checking_date), 4)

        mean_days_between_buy_dict[user_id] = mean_vairance
        mean_days_between_list[index] = mean_days_between_buy_dict[user_id]

        logging.info("%s %s, %d" % (user_id, mean_days_between_list[index], len(days_to_checking_date)))

    logging.info("leaving feature_mean_days_between_buy_user")

    return mean_days_between_list

# 用户最后一次购买至 checking date（不包括）的天数
def feature_last_buy_user(checking_date, user_item_pairs):
    logging.info("entered feature_last_buy_user(%s)" % checking_date)

    last_buy_dict = dict()
    last_buy_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in last_buy_dict):
            last_buy_list[index] = last_buy_dict[user_id]
            continue
        
        last_buy_date = checking_date
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                if (last_buy_date > each_record[-1][1].date()):
                    last_buy_date = each_record[-1][1].date()

        last_buy_dict[user_id] = (checking_date - last_buy_date).days
        last_buy_list[index] = last_buy_dict[user_id]
        logging.info("%s last buy %s / %d days" % (user_id, last_buy_date, last_buy_list[index]))

    logging.info("leaving feature_last_buy_user")

    return last_buy_list
