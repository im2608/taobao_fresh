from common import *
import numpy as np
from feature_selection import *


################################################################################################
################################################################################################
################################################################################################
# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
def feature_item_popularity(behavior_type, item_popularity_dict, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_item_popularity"

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

def feature_item_popularity2(item_popularity_dict, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_item_popularity"

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
def feature_beahvior_cnt_on_item(pre_days, window_end_date, user_behavior_cnt_on_item, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
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
    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, item_behavior_cnt_list, features_names)


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
def feature_days_from_1st_last_behavior_item(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
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

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, days_from_1st_last_list, features_names)

################################################################################################
################################################################################################
################################################################################################

def get_user_cnt_behavior_on_item(window_start_date, window_end_date, item_records, users_behavior_on_item, behavior_weight_on_item):
    for item_id, user_buy_record in item_records.items():
        item_category = global_train_item_category[item_id]

        if (item_id not in users_behavior_on_item):
            users_behavior_on_item[item_id] = dict()
            users_behavior_on_item[item_id][BEHAVIOR_TYPE_VIEW] = set()
            users_behavior_on_item[item_id][BEHAVIOR_TYPE_FAV] = set()
            users_behavior_on_item[item_id][BEHAVIOR_TYPE_CART] = set()
            users_behavior_on_item[item_id][BEHAVIOR_TYPE_BUY] = set()

        if (item_category not in behavior_weight_on_item):
            behavior_weight_on_item[item_category] = dict()

        if (item_id not in behavior_weight_on_item[item_category]):
            behavior_weight_on_item[item_category][item_id] = 0

        for user_id, item_sale_records in user_buy_record.items():
            for each_record in item_sale_records:
                for behavior in each_record:
                    if (behavior[1].date() >= window_start_date and behavior[1].date() < window_end_date):
                        behavior_type = behavior[0]
                        users_behavior_on_item[item_id][behavior_type].add(user_id)
                        behavior_weight_on_item[item_category][item_id] += g_behavior_weight[behavior_type] * behavior[2]
    return


# [begin date, end date) 期间，总共有多少用户在该 item 上进行了各种操作，按照操作数量进行加权，得到 item 上的加权在 category 中的排序
def feature_how_many_users_behavior_item(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    logging.info("feature_how_many_users_bought (%s, %s)" % (window_start_date, window_end_date))

    features_names = ["feature_how_many_users_view",
                      "feature_how_many_users_fav",
                      "feature_how_many_users_cart",
                      "feature_how_many_users_bought",
                      "feature_item_behavior_weight"]

    users_behavior_on_item = dict()
    behavior_weight_on_item = dict()

    get_user_cnt_behavior_on_item(window_start_date, window_end_date, g_user_buy_transection_item, users_behavior_on_item, behavior_weight_on_item)
    get_user_cnt_behavior_on_item(window_start_date, window_end_date, g_user_behavior_patten_item, users_behavior_on_item, behavior_weight_on_item)
    
    for item_category in behavior_weight_on_item:
        sorted_behavior_weight = sorted(behavior_weight_on_item[item_category].items(), key=lambda item:item[1], reverse=True)

        # logging.info("sorted_behavior_weight is %s" % sorted_behavior_weight)
        for i, behavior_weight in enumerate(sorted_behavior_weight):
            behavior_weight_on_item[item_category][behavior_weight[0]] = i + 1

    beahvior_types = [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART, BEHAVIOR_TYPE_BUY]

    how_many_users_behavior_item = np.zeros((len(user_item_pairs), len(features_names)))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        for i, behavior in enumerate(beahvior_types):
            how_many_users_behavior_item[index, i] = len(users_behavior_on_item[item_id][behavior])

        how_many_users_behavior_item[index, 4] = behavior_weight_on_item[item_category][item_id]

        # logging.info("users count behavior on item %s %s" % (item_id, how_many_users_behavior_item[index]))

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, how_many_users_behavior_item, features_names)

################################################################################################
################################################################################################
################################################################################################

def get_item_sale_vol_in_category(window_start_date, window_end_date):
    items_sales_vol_in_category = dict()
    for item_id, user_buy_records in g_user_buy_transection_item.items():
        item_category = global_train_item_category[item_id]
        if (item_category not in items_sales_vol_in_category):
            items_sales_vol_in_category[item_category] = dict()

        sales_vol = 0
        for user_id, item_sale_records in user_buy_records.items():
            for each_record in item_sale_records:
                if (each_record[-1][1].date() >= window_start_date and each_record[-1][1].date() < window_end_date):
                    sales_vol += 1

        items_sales_vol_in_category[item_category][item_id] = sales_vol
        # logging.info("get_item_sale_vol_in_category(%s, %s), category %s, item %s, sale vol %d" %
        #              (window_start_date, window_end_date, item_category, item_id, sales_vol))

    for item_category in items_sales_vol_in_category:
        # 在 category 内部按照 item 的销量排序
        sorted_item_sal_vol = sorted(items_sales_vol_in_category[item_category].items(), key=lambda item:item[1], reverse=True)

        # item_sal_vol = （item id, item sal vol)
        for i, item_sal_vol in enumerate(sorted_item_sal_vol):
            items_sales_vol_in_category[item_category][item_sal_vol[0]] = (item_sal_vol[1], i + 1)

    return items_sales_vol_in_category

# [begin date, end date) 期间 item 的销量, 以及 item 的销量在 category 中其他 item 销量的排序
def feature_item_sals_volume(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    logging.info("feature_item_sals_volume (%s, %s)" % (window_start_date, window_end_date))

    features_names = ["feature_item_sals_volume", "feature_item_sals_volume_rank"]

    # category 中各个 item 的销量，用于排序
    items_sales_vol_in_category = get_item_sale_vol_in_category(window_start_date, window_end_date)

    sals_volume_list = np.zeros((len(user_item_pairs), len(features_names)))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        if (item_category in items_sales_vol_in_category and 
            item_id in items_sales_vol_in_category[item_category]):
            sals_volume_list[index, 0] = items_sales_vol_in_category[item_category][item_id][0]
            sals_volume_list[index, 1] = items_sales_vol_in_category[item_category][item_id][1]

        # logging.info("%s item sales volume %s " % (item_id, sals_volume_list[index]))

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, sals_volume_list, features_names)
