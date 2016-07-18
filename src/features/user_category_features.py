from common import *
import numpy as np
from feature_selection import *



######################################################################################################
######################################################################################################
######################################################################################################


#在 [windw_start_date, window_end_dat) 范围内， user 对 category 购买间隔的平均天数以及方差
def feature_mean_days_between_buy_user_category(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    features_names = ["feature_mean_days_between_buy_user_category", "feature_var_days_between_buy_user_category"]

    user_category_mean_buy_dict = dict()

    buy_mean_days_list = np.zeros((len(user_item_pairs), len(features_names)))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        item_category = global_train_item_category[item_id]

        if ((user_id, item_category) in user_category_mean_buy_dict):
            buy_mean_days_list[index] = user_category_mean_buy_dict[(user_id, item_category)]

        if (user_id not in g_user_buy_transection):
            continue

        buy_date = set()
        for item_id_can, item_buy_records in g_user_buy_transection[user_id].items():
            if (item_category != global_train_item_category[item_id_can]):
                continue
            for each_record in item_buy_records:
                if (each_record[-1][1].date() >= window_start_date and
                    each_record[-1][1].date() < window_end_date):
                    buy_date.add(each_record[-1][1].date())

        if (len(buy_date) == 0):
            continue

        buy_date = list(buy_date)
        buy_date.sort()

        # 若只购买过一次，则将购买日期至checking date 作为平均天数
        if (len(buy_date) == 1):
            buy_date.append(window_end_date)

        days_between_buy = []
        for date_index in range(1, len(buy_date)):
            days_between_buy.append((buy_date[date_index] - buy_date[date_index-1]).days)

        mean_vairance = [0, 0]
        mean_vairance[0] = np.round(np.mean(days_between_buy))
        mean_vairance[1] = np.round(np.var(days_between_buy))

        buy_mean_days_list[index] = mean_vairance
        user_category_mean_buy_dict[(user_id, item_category)] = mean_vairance

        # logging.info("as of %s, user %s, category %s, mean vairance days %s" %\
        #              (window_end_date, user_id, item_category, buy_mean_days_list[index]))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, buy_mean_days_list, features_names)

######################################################################################################
######################################################################################################
######################################################################################################

#[window_start_date, window_end_date) 时间内， 用户一共购买过多少同类型的商品
def feature_how_many_buy_category(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_how_many_buy_category"

    how_many_buy = np.zeros((len(user_item_pairs), 1))

    how_many_buy_dict = {}

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        if ((user_id, item_category) in how_many_buy_dict):
            how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]
            continue

        buy_count = 0
        if (user_id not in g_user_buy_transection):
            continue

        for item_id_can, buy_records in g_user_buy_transection[user_id].items():
            # 不属于同一个 category， skip
            if (global_train_item_category[item_id_can] != item_category):
                continue

            for each_record in buy_records:
                if (each_record[-1][1].date() >= window_start_date and 
                    each_record[-1][1].date() < window_end_date):
                    buy_count += 1

        how_many_buy_dict[(user_id, item_category)] = buy_count
        how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]
        # logging.info("%s to %s, %s bought category %s %d" % (window_start_date, window_end_date, user_id, item_category, buy_count)) 
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    return how_many_buy, 1

######################################################################################################
######################################################################################################
######################################################################################################

# 用户最后一次操作同类型的商品至 checking_date（不包括） 的天数，
# todo： 此处有个问题： 如果用户连续购买了同一个item，则此处会有多条相同的购物记录，但是函数中没有处理这种情况
def get_last_opt_category_date(user_records, window_start_date, window_end_date, user_id, item_category):
    days = [0, 0, 0, 0]

    if (user_id not in user_records):
        return days

    last_opt_date = [window_start_date for x in range(4)]

    for item_id_can, item_opt_records in user_records[user_id].items():
        # 不属于同一个 category， skip
        if (global_train_item_category[item_id_can] != item_category):
            continue

        for each_record in item_opt_records:
            for index in range(len(each_record)-1, -1, -1):
                #each_record 已经按照时间排好序
                behavior_type = each_record[index][0]
                behavior_date = each_record[index][1].date()
                if (behavior_date > last_opt_date[behavior_type - 1] and \
                    behavior_date < window_end_date):
                    last_opt_date[behavior_type - 1] = behavior_date
                    days[behavior_type - 1] = (window_end_date - behavior_date).days
    return days

# 用户最后一次操作同类型的商品至 window_end_date （不包括） 的天数，返回4个特征
# todo： 此处有个问题： 如果用户连续购买了同一个item，则此处会有多条相同的购物记录，但是函数中没有处理这种情况
def feature_last_opt_category(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    features_names = ["feature_last_opt_category_view", 
                      "feature_last_opt_category_fav", 
                      "feature_last_opt_category_cart", 
                      "feature_last_opt_category_buy"]

    days_from_last_opt_cat_dict = dict()
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), len(features_names)))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        if ((user_id, item_category) in days_from_last_opt_cat_dict):
            days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
            continue

        days = get_last_opt_category_date(g_user_buy_transection, window_start_date, window_end_date, user_id, item_category)
        if (user_id in g_user_behavior_patten):
            days2 = get_last_opt_category_date(g_user_behavior_patten, window_start_date, window_end_date, user_id, item_category)
            for index in range(len(days)):
                if (days[index] == 0):
                    if (days2 != 0):
                        days[index] = days2[index]
                elif (days2[index] != 0):
                    days[index] = min(days[index], days2[index])

        days_from_last_opt_cat_dict[(user_id, item_category)] = days

        days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
        # logging.info("%s last opted category %s, days %s to %s" % \
        #              (user_id, item_category, days_from_last_opt_cat_dict[(user_id, item_category)], window_end_date))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, days_from_last_opt_cat_list, features_names)


######################################################################################################
######################################################################################################
######################################################################################################


#截止到checking date(不包括)， 用户在category 上的购买浏览转化率 在category上购买过的数量/浏览过的category数量
def feature_buy_view_ratio(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_buy_view_ratio"

    buy_view_ratio_dict = dict()

    buy_view_ratio_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in g_user_buy_transection):
            continue

        item_category = global_train_item_category[item_id]
        if ((user_id, item_category) in buy_view_ratio_dict):
            buy_view_ratio_list[index, 0] = buy_view_ratio_dict[(user_id, item_category)]
            continue

        buy_count = 0
        for item_id_can, item_buy_records in g_user_buy_transection[user_id].items():
            if (global_train_item_category[item_id_can] != item_category):
                continue
            for each_record in item_buy_records:
                if (each_record[-1][1].date() < window_end_date and 
                    each_record[-1][1].date() >= window_start_date):
                    buy_count += 1

        # 没有pattern， 所有的view 都转化成了buy
        if (user_id not in g_user_behavior_patten):
            buy_view_ratio_list[index][0] = 1
            buy_view_ratio_dict[(user_id, item_category)] = 1
            continue

        viewed_categories = set()
        for item_id in g_user_behavior_patten[user_id]:
            viewed_categories.add(global_train_item_category[item_id])

        buy_view_ratio_dict[(user_id, item_category)] = round(buy_count / (buy_count + len(viewed_categories)), 4)
        buy_view_ratio_list[index, 0] = buy_view_ratio_dict[(user_id, item_category)]
        # logging.info("as of %s, %s bought category %s %d, viewed %d, ratio %.4f" % \
        #              (window_end_date, user_id, item_category, buy_count, len(viewed_categories), buy_view_ratio_list[index, 0]))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")
    
    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    return buy_view_ratio_list, 1


######################################################################################################
######################################################################################################
######################################################################################################

def get_behavior_cnt_of_days_caregory(user_records, pre_days, end_date, behavior_cnt, category, user_id):
    
    begin_date = end_date - datetime.timedelta(pre_days)

    if (user_id not in user_records):
        return behavior_cnt

    for item_id, item_opt_records in user_records[user_id].items():
        category_can = global_train_item_category[item_id]
        if (category != category_can):
            continue

        for each_record in item_opt_records:
            for behavior_consecutive in each_record:
                if (behavior_consecutive[1].date() >= begin_date and 
                    behavior_consecutive[1].date() < end_date):
                    behavior_cnt[behavior_consecutive[0] - 1] += behavior_consecutive[2]

    return behavior_cnt
# user 在 category 上各个行为的次数以及在item上各个行为的次数占category上次数的比例
# 返回 8 个特征
def feature_user_behavior_cnt_on_category(pre_days, window_end_date, user_item_pairs, beahvior_cnt_on_item, cal_feature_importance, cur_total_feature_cnt):
    logging.info("feature_user_behavior_cnt_on_category (%d, %s)" % (pre_days, window_end_date))

    features_names = ["feature_user_behavior_cnt_on_category_pre_days_%d_view" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_fav" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_cart" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_buy" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_item_ratio_view" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_item_ratio_fav" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_item_ratio_cart" % pre_days,
                      "feature_user_behavior_cnt_on_category_pre_days_%d_item_ratio_buy" % pre_days,
                      ]

    behavior_cnt_list = np.zeros((len(user_item_pairs), len(features_names)))
    behavior_cnt_dict = dict()

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        behavior_cnt = [0, 0, 0, 0]
        if ((user_id, item_category) in behavior_cnt_dict):
            behavior_cnt = behavior_cnt_dict[(user_id, item_category)].copy()
        else:
            get_behavior_cnt_of_days_caregory(g_user_buy_transection, pre_days, window_end_date, behavior_cnt, item_category, user_id)
            get_behavior_cnt_of_days_caregory(g_user_behavior_patten, pre_days, window_end_date, behavior_cnt, item_category, user_id)
            behavior_cnt_dict[(user_id, item_category)] = behavior_cnt
            behavior_cnt = behavior_cnt_dict[(user_id, item_category)].copy()

        if (beahvior_cnt_on_item is not None):
            behavior_cnt_ratio = [0, 0, 0, 0]
            for i in range(4):
                if (behavior_cnt[i] > 0):
                    behavior_cnt_ratio[i] = beahvior_cnt_on_item[index, i] / behavior_cnt[i]
            behavior_cnt.extend(behavior_cnt_ratio)

        behavior_cnt_list[index] = behavior_cnt

        # logging.info("user %s, %s %s, behavior on category %s, %s" % (user_id, item_id, beahvior_cnt_on_item[index], item_category, behavior_cnt_list[index]))

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, behavior_cnt_list, features_names)

######################################################################################################
######################################################################################################
######################################################################################################
