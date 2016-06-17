from common import *
import numpy as np
from feature_selection import *



######################################################################################################
######################################################################################################
######################################################################################################


#在 [windw_start_date, window_end_dat) 范围内， user 对 category 购买间隔的平均天数以及方差
def feature_mean_days_between_buy_user_category(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_mean_days_between_buy_user_category"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    user_category_mean_buy_dict = dict()

    buy_mean_days_list = np.zeros((len(user_item_pairs), 2))

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

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    return buy_mean_days_list, 2

######################################################################################################
######################################################################################################
######################################################################################################

#[window_start_date, window_end_date) 时间内， 用户一共购买过多少同类型的商品
def feature_how_many_buy_category(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    feature_name = "feature_how_many_buy_category"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

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
def feature_last_opt_category(window_start_date, window_end_date, user_item_pairs, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    features_names = ["feature_last_opt_category_view", 
                      "feature_last_opt_category_fav", 
                      "feature_last_opt_category_cart", 
                      "feature_last_opt_category_buy"]
    useful_features = None
    if (not cal_feature_importance):
        useful_features = featuresForForecasting(features_names, final_feature_importance)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_last_opt_category] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_last_opt_category] has %d useful features" % len(useful_features))

    days_from_last_opt_cat_dict = dict()
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 4))

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

    return getUsefulFeatures(cal_feature_importance, cur_total_feature_cnt, days_from_last_opt_cat_list, features_names, useful_features)


######################################################################################################
######################################################################################################
######################################################################################################
