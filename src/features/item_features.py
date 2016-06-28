from common import *
import numpy as np
from feature_selection import *


################################################################################################
################################################################################################
################################################################################################
# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
def feature_item_popularity(behavior_type, item_popularity_dict, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_item_popularity"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id][behavior_type]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")
    
    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    return item_popularity_list, 1
################################################################################################
################################################################################################
################################################################################################

def feature_item_popularity2(item_popularity_dict, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_item_popularity"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    return item_popularity_list, 1

################################################################################################
################################################################################################
################################################################################################

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

# 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 平均每天的点击数,方差以及用户在item上behavior的次数占总次数的比例
# 返回 16 个特征
def feature_beahvior_cnt_on_item(pre_days, window_end_date, user_behavior_cnt_on_item, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    logging.info("feature_beahvior_cnt_on_item(%d, %s)" % (pre_days, window_end_date))
    features_names = ["feature_beahvior_cnt_on_item_%d_view" % pre_days, 
                      "feature_beahvior_cnt_on_item_%d_fav" % pre_days,
                      "feature_beahvior_cnt_on_item_%d_cart" % pre_days,
                      "feature_beahvior_cnt_on_item_%d_buy" % pre_days,

                      "feature_beahvior_cnt_on_item_%d_view_mean" % pre_days, 
                      "feature_beahvior_cnt_on_item_%d_fav_mean" % pre_days,
                      "feature_beahvior_cnt_on_item_%d_cart_mean" % pre_days,
                      "feature_beahvior_cnt_on_item_%d_buy_mean" % pre_days,

                      "feature_beahvior_cnt_on_item_%d_view_var" % pre_days, 
                      "feature_beahvior_cnt_on_item_%d_fav_var" % pre_days,
                      "feature_beahvior_cnt_on_item_%d_cart_var" % pre_days,
                      "feature_beahvior_cnt_on_item_%d_buy_var" % pre_days, 

                      "feature_user_beahvior_cnt_on_item_ratio_%d_view" % pre_days,
                      "feature_user_beahvior_cnt_on_item_ratio_%d_fav" % pre_days,
                      "feature_user_beahvior_cnt_on_item_ratio_%d_cart" % pre_days,
                      "feature_user_beahvior_cnt_on_item_ratio_%d_buy" % pre_days,
                      ]

    useful_features = None
    if (not cal_feature_importance):
        useful_features = featuresForForecasting(features_names, final_feature_importance)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_beahvior_cnt_on_item] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_beahvior_cnt_on_item] has %d useful features" % len(useful_features))

    window_start_date = window_end_date - datetime.timedelta(pre_days)

    item_behavior_cnt_dict = dict()
    item_behavior_cnt_list = np.zeros((len(user_item_pairs), len(features_names)))
    slide_window_days = (window_end_date - window_start_date).days

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
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
        #0 -- 3 为行为总数， 4--7 为行为每天的平均数， 8--11 为行为的方差, 12--15 为用户在item上的行为数占item总行为数的比例
        behavior_cnt_mean_var_ratio = [0 for x in range(len(features_names))]
        for i in range(4):
            behavior_cnt_mean_var_ratio[i] = np.sum(behavior_cnt_every_day[i])
            behavior_cnt_mean_var_ratio[i + 4] = round(np.mean(behavior_cnt_every_day[i]), 2)
            behavior_cnt_mean_var_ratio[i + 8] = round(np.var(behavior_cnt_every_day[i]), 2)
            if (behavior_cnt_mean_var_ratio[i] > 0):
                behavior_cnt_mean_var_ratio[i + 12] = round(user_behavior_cnt_on_item[index, i] / behavior_cnt_mean_var_ratio[i], 4)

        item_behavior_cnt_dict[item_id] = behavior_cnt_mean_var_ratio
        item_behavior_cnt_list[index] = behavior_cnt_mean_var_ratio

        # logging.info("item %s, user %s, %s, behavior cnt, mean, var %s" % (item_id, user_id, user_behavior_cnt_on_item[index], behavior_cnt_mean_var_ratio))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    logging.info("leaving feature_beahvior_cnt_on_item")
    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, item_behavior_cnt_list, features_names, useful_features)


################################################################################################
################################################################################################
################################################################################################


def get_item_1st_last_behavior_date(window_start_date, window_end_date, item_id, item_records):
    days_from_1st_behavior = [window_end_date for x in range(4)]
    days_from_last_behavior = [None for x in range(4)]

    for user_id, item_opt_records in item_records[item_id].items():
        for each_record in item_opt_records:
            for each_behavior in each_record:
                behavior_date = each_behavior[1].date()
                beahvior_idx = each_behavior[0] - 1
                if (behavior_date < days_from_1st_behavior[beahvior_idx] and 
                    behavior_date >= window_start_date):
                    days_from_1st_behavior[beahvior_idx] = behavior_date

                if (days_from_last_behavior[beahvior_idx] == None):
                    if (behavior_date >= window_start_date and 
                        behavior_date < window_end_date):
                        days_from_last_behavior[beahvior_idx] = behavior_date
                elif (behavior_date > days_from_last_behavior[beahvior_idx] and \
                      behavior_date < window_end_date):
                    days_from_last_behavior[beahvior_idx] = behavior_date

    for index in range(len(days_from_1st_behavior)):
        days_from_1st_behavior[index] = (window_end_date - days_from_1st_behavior[index]).days

    for index in range(len(days_from_last_behavior)):
        if  (days_from_last_behavior[index] != None):
            days_from_last_behavior[index] = (window_end_date - days_from_last_behavior[index]).days
        else:
            days_from_last_behavior[index] = 0

    return days_from_1st_behavior + days_from_last_behavior


# item 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次 behavior 之间的天数， 返回 12 个特征
def feature_days_from_1st_last_behavior_item(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    logging.info("feature_days_from_1st_last_behavior_item (%s, %s)" % (window_start_date, window_end_date))

    features_names = ["feature_days_from_1st_behavior_item_view", 
                      "feature_days_from_1st_behavior_item_fav", 
                      "feature_days_from_1st_behavior_item_cart", 
                      "feature_days_from_1st_behavior_item_buy",
                      "feature_days_from_last_behavior_item_view", 
                      "feature_days_from_last_behavior_item_fav", 
                      "feature_days_from_last_behavior_item_cart", 
                      "feature_days_from_last_behavior_item_buy",
                      "feature_days_between_1st_last_behavior_item_view", 
                      "feature_days_between_1st_last_behavior_item_fav", 
                      "feature_days_between_1st_last_behavior_item_cart", 
                      "feature_days_between_1st_last_behavior_item_buy"]

    useful_features = None
    if (not cal_feature_importance):
        useful_features = featuresForForecasting(features_names, final_feature_importance)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_days_from_1st_last_behavior_item] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_days_from_1st_last_behavior_item] has %d useful features" % len(useful_features))

    days_from_1st_last_dict = dict()
    days_from_1st_last_list = np.zeros((len(user_item_pairs), len(features_names)))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_1st_last_dict):
            days_from_1st_last_list[index] = days_from_1st_last_dict[item_id]
            continue

        # 共 8 项， 前4 项为第一次行为至end date的天数，后4项为最后一次行为至 end date的天数
        days_from_1st_last_1 = [0 for x in range(8)]
        days_from_1st_last_2 = [0 for x in range(8)]
        if (item_id in g_user_buy_transection_item):
            days_from_1st_last_1 = get_item_1st_last_behavior_date(window_start_date, window_end_date, item_id, g_user_buy_transection_item)

        if (item_id in g_user_behavior_patten_item):
            days_from_1st_last_2 = get_item_1st_last_behavior_date(window_start_date, window_end_date, item_id, g_user_behavior_patten_item)

        for i in range(4):
            if (days_from_1st_last_1[i] > 0):
                if (days_from_1st_last_2[i] > 0):
                    days_from_1st_last_1[i] = max(days_from_1st_last_1[i], days_from_1st_last_2[i])
            else:
                days_from_1st_last_1[i] = days_from_1st_last_2[i]

            if (days_from_1st_last_1[i+4] > 0):
                if (days_from_1st_last_2[i+4] > 0):
                    days_from_1st_last_1[i+4] = min(days_from_1st_last_1[i+4], days_from_1st_last_2[i+4])
            else:
                days_from_1st_last_1[i+4] = days_from_1st_last_2[i+4]

        days_between_1st_last = [0, 0, 0, 0]
        for i in range(4):
            days_between_1st_last[i] = days_from_1st_last_1[i] - days_from_1st_last_1[i+4]

        days_from_1st_last_1.extend(days_between_1st_last)

        days_from_1st_last_list[index] = days_from_1st_last_1
        days_from_1st_last_dict[item_id] = days_from_1st_last_1

        # logging.info("%s item, days from 1st, last behavior %s" % (item_id, days_from_1st_last_1))

        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, days_from_1st_last_list, features_names, useful_features)

################################################################################################
################################################################################################
################################################################################################

# [begin date, end date) 期间，总共有多少用户购买了该 item
def feature_how_many_users_bought_item(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    logging.info("feature_how_many_users_bought (%s, %s)" % (window_start_date, window_end_date))

    feature_name = "feature_how_many_users_bought"

    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
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

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    logging.info("leaving feature_how_many_users_bought")
    return how_many_users_bought_list, 1

################################################################################################
################################################################################################
################################################################################################
# [begin date, end date) 期间， item 的销量
def feature_item_sals_volume(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    logging.info("feature_item_sals_volume (%s, %s)" % (window_start_date, window_end_date))

    feature_name = "feature_item_sals_volume"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    sals_volume_dict = dict()
    sals_volume_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in sals_volume_dict):
            sals_volume_list[index] = sals_volume_dict[item_id]
            continue

        if (item_id not in g_user_buy_transection_item):
            continue

        sales_vol = 0
        for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
            for each_record in item_buy_records:
                if (each_record[-1][1].date() > window_start_date and each_record[-1][1].date() < window_end_date):
                    sales_vol += 1

        sals_volume_dict[item_id] = sales_vol
        sals_volume_list[index] = sales_vol
        # logging.info("%s item sales volume %d " % (item_id, sals_volume_list[index]))

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    logging.info("leaving feature_item_sals_volume")

    return sals_volume_list, 1