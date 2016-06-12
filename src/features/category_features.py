from common import *
import numpy as np
import time


################################################################################################
################################################################################################
################################################################################################

# 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 平均每天的点击数以及方差
# 返回 12 个特征
def get_everyday_behavior_cnt_of_category(window_start_date, window_end_date, item_records, item_category):
    slide_window_days = (window_end_date - window_start_date).days

    behavior_cnt_every_day = np.zeros((4, slide_window_days))

    items_of_category = global_train_category_item[item_category]

    for item_id in items_of_category:  
        if (item_id not in item_records):
            continue

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

# 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 平均每天的点击数以及方差
# 返回 12 个特征
def feature_beahvior_cnt_on_category(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    logging.info("feature_beahvior_cnt_on_category(%s, %s)" % (window_start_date, window_end_date))
    features_names = ["feature_beahvior_cnt_on_category_view", 
                      "feature_beahvior_cnt_on_category_fav",
                      "feature_beahvior_cnt_on_category_cart",
                      "feature_beahvior_cnt_on_category_buy",
                      "feature_beahvior_cnt_on_category_view_mean", 
                      "feature_beahvior_cnt_on_category_fav_mean",
                      "feature_beahvior_cnt_on_category_cart_mean",
                      "feature_beahvior_cnt_on_category_buy_mean",
                      "feature_beahvior_cnt_on_category_view_var", 
                      "feature_beahvior_cnt_on_category_fav_var",
                      "feature_beahvior_cnt_on_category_cart_var",
                      "feature_beahvior_cnt_on_category_buy_var"]

    start_time = time.clock()
    useful_features = None
    if (not during_training):
        useful_features = featuresForForecasting(features_names)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_beahvior_cnt_on_category] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_beahvior_cnt_on_category] has %d useful features" % len(useful_features))

    end_time = time.clock()

    time_for_every_day1 = 0
    time_for_every_day2 = 0
    time_for_calculating = 0

    category_behavior_cnt_dict = dict()
    category_behavior_cnt_list = np.zeros((len(user_item_pairs), len(features_names)))
    slide_window_days = (window_end_date - window_start_date).days

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        item_category = global_train_item_category[item_id]
        if (item_category in category_behavior_cnt_dict):
            category_behavior_cnt_list[index] = category_behavior_cnt_dict[item_category]
            continue

        start_time = time.clock()
        behavior_cnt_every_day = np.zeros((4, slide_window_days))
        if (item_id in g_user_buy_transection_item):
            behavior_cnt_every_day = get_everyday_behavior_cnt_of_category(window_start_date, window_end_date, 
                                                                           g_user_buy_transection_item, item_category)
        end_time = time.clock()
        time_for_every_day1 += end_time - start_time

        start_time = time.clock()
        if (item_id in g_user_behavior_patten_item):
            behavior_cnt_every_day += get_everyday_behavior_cnt_of_category(window_start_date, window_end_date, 
                                                                            g_user_behavior_patten_item, item_category)
        end_time = time.clock()
        time_for_every_day2 += end_time - start_time

        #0 -- 3 为行为总数， 4--7 为行为每天的平均数， 8--11 为行为的方差
        start_time = time.clock()
        behavior_cnt_mean_var = [0 for x in range(len(features_names))]
        for i in range(4):
            behavior_cnt_mean_var[i] = np.sum(behavior_cnt_every_day[i])
            behavior_cnt_mean_var[i + 4] = round(np.mean(behavior_cnt_every_day[i]), 2)
            behavior_cnt_mean_var[i + 8] = round(np.var(behavior_cnt_every_day[i]), 2)

        category_behavior_cnt_dict[item_category] = behavior_cnt_mean_var
        category_behavior_cnt_list[index] = behavior_cnt_mean_var

        end_time = time.clock()
        time_for_calculating += end_time - start_time

        logging.info("%s category, behavior cnt, mean, var %s" % (item_category, behavior_cnt_mean_var))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    # print("every day1 %d, every day2 %d, calculated %d" % (time_for_every_day1, time_for_every_day2, time_for_calculating))

    logging.info("leaving feature_beahvior_cnt_on_category")
    return getUsefulFeatures(during_training, cur_total_feature_cnt, category_behavior_cnt_list, features_names, useful_features)

################################################################################################
################################################################################################
################################################################################################

def get_category_1st_last_behavior_date(window_start_date, window_end_date, item_category, item_records):
    days_from_1st_behavior = [window_end_date for x in range(4)]
    days_from_last_behavior = [None for x in range(4)]

    items_of_category = global_train_category_item[item_category]
   
    for item_id in items_of_category:
        if (item_id not in item_records):
            continue

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


# category 第一次, 最后一次 behavior 距离checking date 的天数, 返回 8 个特征
def feature_days_from_1st_last_behavior_category(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    logging.info("feature_days_from_1st_last_behavior_category (%s, %s)" % (window_start_date, window_end_date))

    features_names = ["feature_days_from_1st_behavior_categroy_view", 
                      "feature_days_from_1st_behavior_categroy_fav", 
                      "feature_days_from_1st_behavior_categroy_cart", 
                      "feature_days_from_1st_behavior_categroy_buy",
                      "feature_days_from_last_behavior_categroy_view", 
                      "feature_days_from_last_behavior_categroy_fav", 
                      "feature_days_from_last_behavior_categroy_cart", 
                      "feature_days_from_last_behavior_categroy_buy"]

    useful_features = None
    if (not during_training):
        useful_features = featuresForForecasting(features_names)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_days_from_1st_behavior] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_days_from_1st_behavior] has %d useful features" % len(useful_features))


    days_from_1st_last_dict = dict()
    days_from_1st_last_list = np.zeros((len(user_item_pairs), len(features_names)))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        item_category = global_train_item_category[item_id]

        if (item_category in days_from_1st_last_dict):
            days_from_1st_last_list[index] = days_from_1st_last_dict[item_category]
            continue

        # 共 8 项， 前4 项为第一次行为至end date的天数，后4项为最后一次行为至 end date的天数
        days_from_1st_last_1 = get_category_1st_last_behavior_date(window_start_date, window_end_date, item_category, g_user_buy_transection_item)
        days_from_1st_last_2 = get_category_1st_last_behavior_date(window_start_date, window_end_date, item_category, g_user_behavior_patten_item)

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

        days_from_1st_last_list[index] = days_from_1st_last_1
        days_from_1st_last_dict[item_category] = days_from_1st_last_1

        logging.info("%s category, days from 1st, last behavior %s" % (item_category, days_from_1st_last_1))

        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return getUsefulFeatures(during_training, cur_total_feature_cnt, days_from_1st_last_list, features_names, useful_features)

################################################################################################
################################################################################################
################################################################################################

# [begin date, end date) 期间，总共有多少用户购买了该 category
def feature_how_many_users_bought_category(window_start_date, window_end_date, user_item_pairs, during_training, cur_total_feature_cnt):
    logging.info("feature_how_many_users_bought_category (%s, %s)" % (window_start_date, window_end_date))

    feature_name = "feature_how_many_users_bought_category"
    if (not during_training and feature_name not in g_useful_feature_info):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    how_many_users_bought_dict = dict()
    how_many_users_bought_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        item_category = global_train_item_category[item_id]

        if (item_category in how_many_users_bought_dict):
            how_many_users_bought_list[index] = how_many_users_bought_dict[item_category]
            continue

        items_of_category = global_train_category_item[item_category]
        logging.info("category %s has %d items" % (item_category, len(items_of_category)))

        users_bought_item = set()
        for item_id in items_of_category:
            if (item_id not in g_user_buy_transection_item):
                continue

            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    if (each_record[-1][1].date() >= window_start_date and 
                        each_record[-1][1].date() < window_end_date):
                        users_bought_item.add(user_id)
                        break

        user_cnt = len(users_bought_item)
        how_many_users_bought_dict[item_id] = user_cnt
        how_many_users_bought_list[index] = user_cnt

        logging.info("%d users bought category %s" % (user_cnt, item_category))

    if (during_training):
        g_feature_info[cur_total_feature_cnt] = feature_name

    logging.info("leaving feature_how_many_users_bought_category")
    return how_many_users_bought_list, 1

################################################################################################
################################################################################################
################################################################################################
