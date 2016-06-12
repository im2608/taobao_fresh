from common import *
import numpy as np
#import time


################################################################################################
################################################################################################
################################################################################################
# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
def feature_item_popularity(behavior_type, item_popularity_dict, user_item_pairs, during_training, cur_total_feature_cnt):
    feature_name = "feature_item_popularity"
    if (not during_training and feature_name not in g_useful_feature_info):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id][behavior_type]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")
    
    if (during_training):
        g_feature_info[cur_total_feature_cnt] = feature_name

    return item_popularity_list, 1
################################################################################################
################################################################################################
################################################################################################

def feature_item_popularity2(item_popularity_dict, user_item_pairs, during_training, cur_total_feature_cnt):
    feature_name = "feature_item_popularity"
    if (not during_training and feature_name not in g_useful_feature_info):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    if (during_training):
        g_feature_info[cur_total_feature_cnt] = feature_name

    return item_popularity_list, 1

################################################################################################
################################################################################################
################################################################################################

# 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 平均每天的点击数以及方差
# 返回 12 个特征

def get_everyday_behavior_cnt_of_item(window_start_date, window_end_date, item_records, item_id):
    slide_window_days = (window_end_date - window_start_date).days

    behavior_cnt_every_day = np.zeros((4, slide_window_days))
    for user_id, item_opt_records in item_records[item_id].items():
        for each_record in item_opt_records:
            for each_behavior in each_record:
                beahvior_type = each_behavior[0]
                behavior_date = each_behavior[1].date()                        
                beahvior_cnt  = each_behavior[2]
                if (behavior_date >= window_start_date and behavior_date < window_end_date):
                    day_idx = (behavior_date - window_start_date).days
                    behavior_cnt_every_day[beahvior_type-1, day_idx] += beahvior_cnt

    return behavior_cnt_every_day

def feature_beahvior_cnt_on_item(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    logging.info("feature_beahvior_cnt_on_item(%s, %s)" % (window_start_date, window_end_date))
    features_names = ["feature_beahvior_cnt_on_item_view", 
                      "feature_beahvior_cnt_on_item_fav",
                      "feature_beahvior_cnt_on_item_cart",
                      "feature_beahvior_cnt_on_item_buy",
                      "feature_beahvior_cnt_on_item_view_mean", 
                      "feature_beahvior_cnt_on_item_fav_mean",
                      "feature_beahvior_cnt_on_item_cart_mean",
                      "feature_beahvior_cnt_on_item_buy_mean",
                      "feature_beahvior_cnt_on_item_view_var", 
                      "feature_beahvior_cnt_on_item_fav_var",
                      "feature_beahvior_cnt_on_item_cart_var",
                      "feature_beahvior_cnt_on_item_buy_var"]

    useful_features = None
    if (not during_training):
        useful_features = featuresForForecasting(features_names)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_beahvior_cnt_on_item] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_beahvior_cnt_on_item] has %d useful features" % len(useful_features))

    item_behavior_cnt_dict = dict()
    item_behavior_cnt_list = np.zeros((len(user_item_pairs), 12))
    slide_window_days = (window_end_date - window_start_date).days

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in item_behavior_cnt_dict):
            item_behavior_cnt_list[index] = item_behavior_cnt_dict[item_id]
            continue

        behavior_cnt_every_day = np.zeros((4, slide_window_days))
        if (item_id in g_user_buy_transection_item):
            behavior_cnt_every_day = get_everyday_behavior_cnt_of_item(window_start_date, window_end_date, 
                                                                       g_user_buy_transection_item, item_id)
        if (item_id in g_user_behavior_patten_item):
            behavior_cnt_every_day += get_everyday_behavior_cnt_of_item(window_start_date, window_end_date, 
                                                                        g_user_behavior_patten_item, item_id)
        #0 -- 3 为行为总数， 4--7 为行为每天的平均数， 8--11 为行为的方差
        behavior_cnt_mean_var = [0 for x in range(len(features_names))]
        for i in range(4):
            behavior_cnt_mean_var[i] = np.sum(behavior_cnt_every_day[i])
            behavior_cnt_mean_var[i + 4] = round(np.mean(behavior_cnt_every_day[i]), 2)
            behavior_cnt_mean_var[i + 8] = round(np.var(behavior_cnt_every_day[i]), 2)

        item_behavior_cnt_dict[item_id] = behavior_cnt_mean_var
        item_behavior_cnt_list[index] = behavior_cnt_mean_var

        # logging.info("item %s, behavior cnt, mean, var %s" % (item_id, behavior_cnt_mean_var))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    logging.info("leaving feature_beahvior_cnt_on_item")
    return getUsefulFeatures(during_training, cur_total_feature_cnt, item_behavior_cnt_list, features_names, useful_features)


################################################################################################
################################################################################################
################################################################################################

# item 第一次behavior 距离checking date 的天数, 返回 4 个特征
def feature_days_from_1st_behavior(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    features_names = ["feature_days_from_1st_behavior_view", 
                      "feature_days_from_1st_behavior_fav", 
                      "feature_days_from_1st_behavior_cart", 
                      "feature_days_from_1st_behavior_buy"]
    useful_features = None
    if (not during_training):
        useful_features = featuresForForecasting(features_names)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_days_from_1st_behavior] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_days_from_1st_behavior] has %d useful features" % len(useful_features))


    total_start = time.clock()
    days_from_1st_dict = dict()
    days_from_1st_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_1st_dict):
            days_from_1st_list[index] = days_from_1st_dict[item_id]
            continue

        days_from_1st_behavior = [window_end_date for x in range(4)]

        if (item_id in g_user_buy_transection_item):
            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        behavior_date = each_behavior[1].date()
                        if (behavior_date < days_from_1st_behavior[each_behavior[0] - 1] and 
                            behavior_date >= window_start_date):
                            days_from_1st_behavior[each_behavior[0] - 1] = behavior_date

        if (item_id in g_user_behavior_patten_item):
            for user_id, item_opt_records in g_user_behavior_patten_item[item_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        behavior_date = each_behavior[1].date()
                        if (behavior_date < days_from_1st_behavior[each_behavior[0] - 1] and 
                            behavior_date >= window_start_date):
                            days_from_1st_behavior[each_behavior[0] - 1] = behavior_date

        #days_from_1st_behavior = list(map(lambda x: (checking_date - x).days, days_from_1st_behavior))
        for index in range(len(days_from_1st_behavior)):
            days_from_1st_behavior[index] = (window_end_date - days_from_1st_behavior[index]).days

        days_from_1st_list[index] = days_from_1st_behavior
        days_from_1st_dict[item_id] = days_from_1st_behavior

        # logging.info("item %s days from 1st behavior to %s: %s " % (item_id, window_end_date, days_from_1st_behavior))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return getUsefulFeatures(during_training, cur_total_feature_cnt, days_from_1st_list, features_names, useful_features)

################################################################################################
################################################################################################
################################################################################################


# item 最后一次behavior 距离 window_end_date 的天数, 返回 4 个特征
def feature_days_from_last_behavior(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    features_names = ["feature_days_from_last_behavior_view",
                      "feature_days_from_last_behavior_fav",
                      "feature_days_from_last_behavior_cart",
                      "feature_days_from_last_behavior_buy"]
    useful_features = None
    if (not during_training):
        useful_features = featuresForForecasting(features_names)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_days_from_last_behavior] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_days_from_last_behavior] has %d useful features" % len(useful_features))

    days_from_last_dict = dict()
    days_from_last_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_last_dict):
            days_from_last_list[index] = days_from_last_dict[item_id]
            continue

        days_from_last_behavior = [None for x in range(4)]

        if (item_id in g_user_buy_transection_item):
            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        if (days_from_last_behavior[each_behavior[0] - 1] == None):
                            if (each_behavior[1].date() >= window_start_date and 
                                each_behavior[1].date() < window_end_date):
                                days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()
                        elif (each_behavior[1].date() > days_from_last_behavior[each_behavior[0] - 1] and \
                              each_behavior[1].date() < window_end_date):
                            days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        if (item_id in g_user_behavior_patten):
            for user_id, item_opt_records in g_user_behavior_patten[item_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        if (days_from_last_behavior[each_behavior[0] - 1] == None):
                            if (each_behavior[1].date() >= window_start_date and 
                                each_behavior[1].date() < window_end_date):
                                days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()
                        elif (each_behavior[1].date() > days_from_last_behavior[each_behavior[0] - 1] and \
                              each_behavior[1].date() < window_end_date):
                              days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        # days_from_last_behavior = list(map(lambda x: (checking_date - x).days, days_from_last_behavior))
        for index in range(len(days_from_last_behavior)):
            if (days_from_last_behavior[index] != None):
                days_from_last_behavior[index] = (window_end_date - days_from_last_behavior[index]).days
            else:
                days_from_last_behavior[index] = 0

        days_from_last_list[index] = days_from_last_behavior
        days_from_last_dict[item_id] = days_from_last_behavior
        # logging.info("item %s days from last behavior to %s: %s " % (item_id, window_end_date, days_from_last_behavior))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return getUsefulFeatures(during_training, cur_total_feature_cnt, days_from_last_list, features_names, useful_features)
################################################################################################
################################################################################################
################################################################################################

# [begin date, end date) 期间，总共有多少用户购买了该 item
def feature_how_many_users_bought(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    logging.info("feature_how_many_users_bought (%s, %s)" % (window_start_date, window_end_date))

    feature_name = "feature_how_many_users_bought"
    if (not during_training and feature_name not in g_useful_feature_info):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    how_many_users_bought_dict = dict()
    how_many_users_bought_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        if (item_id in how_many_users_bought_dict):
            how_many_users_bought_list[index] = how_many_users_bought_dict[item_id]
            continue

        users_bought_item = set()
        if (item_id not in g_user_buy_transection_item):
            continue

        for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
            if (user_id in users_bought_item):
                continue

            for each_record in item_buy_records:
                if (each_record[-1][1].date() >= window_start_date and 
                    each_record[-1][1].date() < window_end_date):
                    users_bought_item.add(user_id)
                    break

        user_cnt = len(users_bought_item)
        how_many_users_bought_dict[item_id] = user_cnt
        how_many_users_bought_list[index] = user_cnt

        # logging.info("%d users bought item %s" % (user_cnt, item_id))

    if (during_training):
        g_feature_info[cur_total_feature_cnt] = feature_name

    logging.info("leaving feature_how_many_users_bought")
    return how_many_users_bought_list, 1

################################################################################################
################################################################################################
################################################################################################