from common import *
import numpy as np
import time
from feature_selection import *
from sklearn.preprocessing import OneHotEncoder

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

# 前 pre_days 天， category 上 各个behavior 的总次数, 平均每天的点击数以及方差

# 返回 17 个特征
def feature_beahvior_cnt_on_category(pre_days, window_end_date, user_item_pairs):
    logging.info("feature_beahvior_cnt_on_category(%d, %s)" % (pre_days, window_end_date))

    features_cnt = 17
  
    window_start_date = window_end_date - datetime.timedelta(pre_days)

    time_for_every_day1 = 0
    time_for_every_day2 = 0
    time_for_calculating = 0

    category_behavior_cnt_dict = dict()
    category_behavior_cnt_list = np.zeros((len(user_item_pairs), features_cnt))
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
        behavior_cnt_mean_var = [0 for x in range(features_cnt)]
        for i in range(4):
            behavior_cnt_mean_var[i] = np.sum(behavior_cnt_every_day[i])
            behavior_cnt_mean_var[i + 4] = round(np.mean(behavior_cnt_every_day[i]), 2)
            behavior_cnt_mean_var[i + 8] = round(np.var(behavior_cnt_every_day[i]), 2)

        # 购物车/浏览
        if (behavior_cnt_mean_var[0] > 0):
            behavior_cnt_mean_var[12] = behavior_cnt_mean_var[2] / behavior_cnt_mean_var[0]
        # 购物车/收藏
        if (behavior_cnt_mean_var[1] > 0):
            behavior_cnt_mean_var[13] = behavior_cnt_mean_var[2] / behavior_cnt_mean_var[1]
        # 购买/浏览
        if (behavior_cnt_mean_var[0] > 0):
            behavior_cnt_mean_var[14] = behavior_cnt_mean_var[3] / behavior_cnt_mean_var[0]
        # 购买/收藏
        if (behavior_cnt_mean_var[1] > 0):
            behavior_cnt_mean_var[15] = behavior_cnt_mean_var[3] / behavior_cnt_mean_var[1]
        # 购买/收藏
        if (behavior_cnt_mean_var[2] > 0):
            behavior_cnt_mean_var[16] = behavior_cnt_mean_var[3] / behavior_cnt_mean_var[2]

        category_behavior_cnt_dict[item_category] = behavior_cnt_mean_var
        category_behavior_cnt_list[index] = behavior_cnt_mean_var

        end_time = time.clock()
        time_for_calculating += end_time - start_time

        # logging.info("%s category, behavior cnt, mean, var %s" % (item_category, behavior_cnt_mean_var))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    # print("every day1 %d, every day2 %d, calculated %d" % (time_for_every_day1, time_for_every_day2, time_for_calculating))

    category_behavior_cnt_list = preprocessing.scale(category_behavior_cnt_list)

    logging.info("leaving feature_beahvior_cnt_on_category")
    return category_behavior_cnt_list

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


# category 第一次, 最后一次 behavior 距离checking date 的天数, 以及 behavior 第一次, 最后一次之间的天数， 返回 12 个特征
def feature_days_from_1st_last_behavior_category(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_days_from_1st_last_behavior_category (%s, %s)" % (window_start_date, window_end_date))

    features_cnt = 12

    days_from_1st_last_dict = dict()
    days_from_1st_last_list = np.zeros((len(user_item_pairs), features_cnt))

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

        days_between_1st_last = [0, 0, 0, 0]
        for i in range(4):
            days_between_1st_last[i] = days_from_1st_last_1[i] - days_from_1st_last_1[i+4]

        days_from_1st_last_1.extend(days_between_1st_last)

        days_from_1st_last_list[index] = days_from_1st_last_1
        days_from_1st_last_dict[item_category] = days_from_1st_last_1

        # logging.info("%s category, days from 1st, last behavior %s" % (item_category, days_from_1st_last_1))

        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    # 因为之后要做交叉特征， category 的第一次, 最后一次 behavior 距离checking date 的天数 先不做归一化，做完交叉特征之后再做
    days_from_1st_last_list[:, 8:12] = preprocessing.scale(days_from_1st_last_list[:, 8:12])

    return days_from_1st_last_list

################################################################################################
################################################################################################
################################################################################################

# [begin date, end date) 期间，总共有多少用户购买了该 category, 以及购买category的用户数在所有cagetory中的排序
def feature_how_many_users_bought_category(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_how_many_users_bought_category (%s, %s)" % (window_start_date, window_end_date))

    how_many_users_bought_dict = dict()

    # 第一列为用户数，第二列为排序
    how_many_users_bought_list = np.zeros((len(user_item_pairs), 2))

    users_cnt_bought_category = dict()

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        item_category = global_train_item_category[item_id]

        if (item_category in how_many_users_bought_dict):
            how_many_users_bought_list[index, 0] = how_many_users_bought_dict[item_category]
            continue

        items_of_category = global_train_category_item[item_category]
        # logging.info("category %s has %d items" % (item_category, len(items_of_category)))

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
        users_cnt_bought_category[item_category] = user_cnt        
        how_many_users_bought_list[index, 0] = user_cnt

    sorted_users_bought_category = sorted(users_cnt_bought_category.items(), key=lambda item:item[1], reverse=True)

    # 购买category的用户数相同，排序也相同
    rank = 1
    cur_user_cnt = sorted_users_bought_category[0][1]
    for category_user_cnt in sorted_users_bought_category:
        category = category_user_cnt[0]
        user_cnt = category_user_cnt[1]
        if (user_cnt < cur_user_cnt):
            cur_user_cnt = user_cnt
            rank += 1
        users_cnt_bought_category[category] = rank

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]
        how_many_users_bought_list[index, 1] = users_cnt_bought_category[item_category]

        # logging.info("%d users bought category %s" % (user_cnt, item_category))

    # 将排序做onehot编码
    # rank_onehot = oneHotEncodeRank(how_many_users_bought_list[:, 1])
    # feature_mat = np.column_stack((how_many_users_bought_list[:, 0], rank_onehot))

    logging.info("leaving feature_how_many_users_bought_category, features_cnt %d" % how_many_users_bought_list.shape[1])
    return how_many_users_bought_list

################################################################################################
################################################################################################
################################################################################################
# [begin date, end date) 期间， category 的销量, 以及销量的排序
def feature_category_sals_volume(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_category_sals_volume (%s, %s)" % (window_start_date, window_end_date))

    pairs_count = len(user_item_pairs)

    sals_volume_dict = dict()
    # 第一列为销量， 第二列为排序
    sals_volume_list = np.zeros((pairs_count, 2))

    # category的index，供记录排序使用
    category_index_dict = dict()

    for index in range(pairs_count):
        item_id = user_item_pairs[index][1]

        item_category = global_train_item_category[item_id]
        if (item_category not in category_index_dict):
            category_index_dict[item_category] = []

        category_index_dict[item_category].append(index)

        if (item_category in sals_volume_dict):
            sals_volume_list[index] = sals_volume_dict[item_category]
            continue

        items_of_category = global_train_category_item[item_category]

        sales_vol = 0
        for item_id in items_of_category:  
            if (item_id not in g_user_buy_transection_item):
                continue
            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    if (each_record[-1][1].date() >= window_start_date and each_record[-1][1].date() < window_end_date):
                        sales_vol += 1

        sals_volume_dict[item_category] = sales_vol
        sals_volume_list[index] = sales_vol
        # logging.info("%s category sales volume %d " % (item_id, sals_volume_list[index]))

    # category 销量的排序
    sorted_sale_vol = sorted(sals_volume_dict.items(), key=lambda item:item[1], reverse=True)    
    sorted_rank = getRankFromSortedTuple(sorted_sale_vol)
    for i, rank in enumerate(sorted_rank):
        item_category = sorted_sale_vol[i][0]
        for category_idx in category_index_dict[item_category]:
            sals_volume_list[category_idx, 1] = rank

    sals_volume_list[:, 0] = preprocessing.scale(sals_volume_list[:, 0])

    logging.info("oneHotEncodeRank feature_category_sals_volume sals_volume_list[:, 1] %d" % np.max(sals_volume_list[:, 1]))

    # rank_onehot = oneHotEncodeRank(sals_volume_list[:, 1])
    # feature_mat = np.column_stack((sals_volume_list[:, 0], rank_onehot))

    logging.info("leaving feature_category_sals_volume, features_cnt %d" % sals_volume_list.shape[1])

    return sals_volume_list