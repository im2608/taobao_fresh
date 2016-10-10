from common import *
import numpy as np
from feature_selection import *


################################################################################################
################################################################################################
################################################################################################

# 得到在item上有过各个行为的用户列表
def get_user_opted_item_category(window_start_date, window_end_date, item_records, user_opted_item_category_dict):
    for item_id, item_opt_records in item_records.items():
        if (item_id not in user_opted_item_category_dict):
            user_opted_item_category_dict[item_id] = [set() for x in range(4)]

        item_category = global_train_item_category[item_id]
        if (item_category not in user_opted_item_category_dict):
            user_opted_item_category_dict[item_category] = [set() for x in range(4)]

        for user_id, user_opt_records in item_opt_records.items():
            for each_record in user_opt_records:
                for each_behavior in each_record:
                    if (each_behavior[1].date() >= window_start_date and each_behavior[1].date() < window_end_date):
                        beahvior_type = each_behavior[0]
                        user_opted_item_category_dict[item_id][beahvior_type - 1].add(user_id)
                        user_opted_item_category_dict[item_category][beahvior_type - 1].add(user_id)

# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
# 返回 4 个特征
def feature_item_popularity(window_start_date, window_end_date, user_item_pairs):

    item_popularity_dict = dict()
    item_popularity_list = np.zeros((len(user_item_pairs), 4))

    user_opted_item_category_dict = dict()
    get_user_opted_item_category(window_start_date, window_end_date, g_user_buy_transection_item, user_opted_item_category_dict)
    get_user_opted_item_category(window_start_date, window_end_date, g_user_behavior_patten_item, user_opted_item_category_dict)

    for item_id_category, user_list in user_opted_item_category_dict.items():
        user_cnt = [len(user_list[i]) for i in range(len(user_list))]
        user_opted_item_category_dict[item_id_category] = user_cnt

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        user_cnt_on_item = user_opted_item_category_dict[item_id]
        user_cnt_on_category = user_opted_item_category_dict[item_category]

        for i in range(len(user_cnt_on_item)):
            if (user_cnt_on_category[i] > 0):
                item_popularity_list[index, i] = user_cnt_on_item[i] / user_cnt_on_category[i]

    return item_popularity_list
################################################################################################
################################################################################################
################################################################################################

def feature_item_popularity2(item_popularity_dict, user_item_pairs):
    feature_name = "feature_item_popularity"

    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    item_popularity_list = preprocessing.scale(item_popularity_list)

    return item_popularity_list

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

def get_item_buy_ratio(window_start_date, window_end_date, item_id):
    if (item_id not in g_user_buy_transection_item):
        return 0

    users_buy_item = set()

    if (item_id in g_user_buy_transection_item):
        for user_id, item_opt_records in g_user_buy_transection_item[item_id].items():
            for each_record in item_opt_records:
                if (each_record[-1][1].date() >= window_start_date and each_record[-1][1].date() < window_end_date):
                    users_buy_item.add(user_id)
                    break

    users_opted_item = set()
    if (item_id in g_user_behavior_patten_item):
        for user_id, item_opt_records in g_user_behavior_patten_item[item_id].items():
            for each_record in item_opt_records:
                if (each_record[0][1].date() >= window_start_date and each_record[-1][1].date() < window_end_date):
                    users_opted_item.add(user_id)                
                    break

    users_opted_item = users_opted_item.union(users_buy_item)

    if (len(users_opted_item) == 0):
        return 0

    return  round(len(users_buy_item) / len(users_opted_item), 4)

# 在 [begin_date, checking_date) 期间， item 上各个 behavior 的总次数, 平均每天的点击数,方差, 
# item的 购买数/浏览，收藏，购物车 的比率, item 的购物车/浏览，收藏的比率
# 浏览/总数， 收藏/总数， 购物车/总数， 购买/总数
# 以及用户在item上behavior的次数占总次数的比例
# 转化率： 购买item的用户数/访问过item的用户数, 以及item的转化率在category中的排序
# 返回 27 个特征
def feature_beahvior_cnt_on_item(pre_days, window_end_date, user_behavior_cnt_on_item, user_item_pairs):
    logging.info("feature_beahvior_cnt_on_item(%d, %s)" % (pre_days, window_end_date))

    feature_count = 26

    window_start_date = window_end_date - datetime.timedelta(pre_days)

    item_behavior_cnt_dict = dict()
    item_behavior_cnt_list = np.zeros((len(user_item_pairs), feature_count))
    slide_window_days = (window_end_date - window_start_date).days

    convert_ratio_dict = dict()

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        #0 -- 3 为行为总数， 4--7 为行为每天的平均数， 8--11 为行为的方差, 12 -- 14 item的购买数/点击，收藏，购物车 的比率
        # 15-16 item 的购物车/浏览，收藏的比率
        behavior_cnt_mean_var = [0 for x in range(22)]

        if (item_id in item_behavior_cnt_dict):
            behavior_cnt_mean_var = item_behavior_cnt_dict[item_id].copy()
        else:
            behavior_cnt_every_day = np.zeros((4, slide_window_days))
            if (item_id in g_user_buy_transection_item):
                behavior_cnt_every_day = get_everyday_behavior_cnt_of_item(window_start_date, window_end_date, 
                                                                           g_user_buy_transection_item, item_id)
            if (item_id in g_user_behavior_patten_item):
                behavior_cnt_every_day += get_everyday_behavior_cnt_of_item(window_start_date, window_end_date, 
                                                                            g_user_behavior_patten_item, item_id)
            for i in range(4):
                behavior_cnt_mean_var[i] = np.sum(behavior_cnt_every_day[i])
                behavior_cnt_mean_var[i + 4] = round(np.mean(behavior_cnt_every_day[i]), 2)
                behavior_cnt_mean_var[i + 8] = round(np.var(behavior_cnt_every_day[i]), 2)

            # 购买数/浏览, 购物车/浏览
            if (behavior_cnt_mean_var[0] > 0):
                behavior_cnt_mean_var[12] = behavior_cnt_mean_var[3] / behavior_cnt_mean_var[0]
                behavior_cnt_mean_var[15] = behavior_cnt_mean_var[2] / behavior_cnt_mean_var[0]

            # 购买数/收藏， 购物车/收藏
            if (behavior_cnt_mean_var[1] > 0):
                behavior_cnt_mean_var[13] = behavior_cnt_mean_var[3] / behavior_cnt_mean_var[1]
                behavior_cnt_mean_var[16] = behavior_cnt_mean_var[2] / behavior_cnt_mean_var[1]

            # 购买数/购物车
            if (behavior_cnt_mean_var[2] > 0):
                behavior_cnt_mean_var[14] = behavior_cnt_mean_var[3] / behavior_cnt_mean_var[2]

            total_behavior_cnt = sum(behavior_cnt_mean_var[0:4])
            if (total_behavior_cnt > 0):
                for i in range(4):
                    behavior_cnt_mean_var[i + 17] = behavior_cnt_mean_var[i] / total_behavior_cnt

            # 转化率： 购买item的用户数/访问过item的用户数
            behavior_cnt_mean_var[21] = get_item_buy_ratio(window_start_date, window_end_date, item_id)
            item_category = global_train_item_category[item_id]

            # 同类型商品的转化率
            if (item_category not in convert_ratio_dict):
                convert_ratio_dict[item_category] = set()
            convert_ratio_dict[item_category].add((item_id, behavior_cnt_mean_var[21]))

            item_behavior_cnt_dict[item_id] = behavior_cnt_mean_var.copy()

        # 用户在item上的行为数占item总行为数的比例
        user_behavior_cnt_ratio = [0, 0, 0, 0]
        for i in range(4):
            if (behavior_cnt_mean_var[i] > 0):
                user_behavior_cnt_ratio[i] = round(user_behavior_cnt_on_item[index, i] / behavior_cnt_mean_var[i], 4)
        behavior_cnt_mean_var.extend(user_behavior_cnt_ratio)

        item_behavior_cnt_list[index] = behavior_cnt_mean_var

        # logging.info("item %s, user %s, %s, behavior cnt, mean, var %s" % (item_id, user_id, user_behavior_cnt_on_item[index], behavior_cnt_mean_var))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    item_behavior_cnt_list = preprocessing.scale(item_behavior_cnt_list)

    # 对category中的item的转化率进行排序
    for category, convert_ratio_in_caterory in convert_ratio_dict.items():
        sorted_ratio = sorted(convert_ratio_in_caterory, key=lambda item:item[1], reverse=True)
        convert_ratio_dict[category] = dict()

        rank = 1
        cur_ratio = sorted_ratio[0][1]
        for i, item_ratio in enumerate(sorted_ratio):
            item_id = item_ratio[0]
            convert_ratio = item_ratio[1]

            if (convert_ratio < cur_ratio):
                cur_ratio = convert_ratio
                rank += 1

            convert_ratio_dict[category][item_id] = rank

    convert_ratio_rank = []
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        category = global_train_item_category[item_id]
        convert_ratio_rank.append(convert_ratio_dict[category][item_id])

    item_behavior_cnt_list = np.column_stack((item_behavior_cnt_list, convert_ratio_rank))

    logging.info("leaving feature_beahvior_cnt_on_item")

    return item_behavior_cnt_list

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
def feature_days_from_1st_last_behavior_item(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_days_from_1st_last_behavior_item (%s, %s)" % (window_start_date, window_end_date))

    days_from_1st_last_dict = dict()
    days_from_1st_last_list = np.zeros((len(user_item_pairs), 12))

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

    # 因为之后要做交叉特征， item 第一次, 最后一次 behavior 距离checking date 的天数 先不做归一化，做完交叉特征之后再做
    days_from_1st_last_list[:, 8:12] = preprocessing.scale(days_from_1st_last_list[:, 8:12])

    return days_from_1st_last_list

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
def feature_how_many_users_behavior_item(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_how_many_users_bought (%s, %s)" % (window_start_date, window_end_date))

    users_behavior_on_item = dict()
    behavior_weight_on_item = dict()

    get_user_cnt_behavior_on_item(window_start_date, window_end_date, g_user_buy_transection_item, users_behavior_on_item, behavior_weight_on_item)
    get_user_cnt_behavior_on_item(window_start_date, window_end_date, g_user_behavior_patten_item, users_behavior_on_item, behavior_weight_on_item)

    for item_category in behavior_weight_on_item:
        sorted_behavior_weight = sorted(behavior_weight_on_item[item_category].items(), key=lambda item:item[1], reverse=True)

        # logging.info("sorted_behavior_weight is %s" % sorted_behavior_weight)
        rank = 1
        cur_weight = sorted_behavior_weight[0][1]
        for behavior_weight in sorted_behavior_weight:
            if (cur_weight > behavior_weight[1]):
                cur_weight = behavior_weight[1]
                rank += 1
            behavior_weight_on_item[item_category][behavior_weight[0]] = rank

    beahvior_types = [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART, BEHAVIOR_TYPE_BUY]

    how_many_users_behavior_item = np.zeros((len(user_item_pairs), 5))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        for i, behavior in enumerate(beahvior_types):
            how_many_users_behavior_item[index, i] = len(users_behavior_on_item[item_id][behavior])

        how_many_users_behavior_item[index, 4] = behavior_weight_on_item[item_category][item_id]

        # logging.info("users count behavior on item %s %s" % (item_id, how_many_users_behavior_item[index]))

    # rank_onehot = oneHotEncodeRank(how_many_users_behavior_item[:, 4])

    # 购买用户数在后边还要用到交叉特征， 这里先不对购买用户数做归一化，在做完交叉特征之后再做
    how_many_users_behavior_item[:, 0] = preprocessing.scale(how_many_users_behavior_item[:, 0])
    how_many_users_behavior_item[:, 1] = preprocessing.scale(how_many_users_behavior_item[:, 1])
    how_many_users_behavior_item[:, 2] = preprocessing.scale(how_many_users_behavior_item[:, 2])

    # feature_mat = np.column_stack((how_many_users_behavior_item[:, 0:4], rank_onehot))

    logging.info("feature_how_many_users_behavior_item returns features count %d" % how_many_users_behavior_item.shape[1])

    return how_many_users_behavior_item

################################################################################################
################################################################################################
################################################################################################

def get_item_sale_vol(window_start_date, window_end_date, item_id):
    if (item_id not in g_user_buy_transection_item):
        return 0

    sales_vol = 0
    for user_id, user_buy_records in g_user_buy_transection_item[item_id].items():
        for each_record in user_buy_records:
            if (each_record[-1][1].date() >= window_start_date and 
                each_record[-1][1].date() < window_end_date):
                sales_vol += 1

    return sales_vol

# [begin date, end date) 期间 item 的销量
def feature_item_sals_volume(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_item_sals_volume (%s, %s)" % (window_start_date, window_end_date))

    items_sales_vol_dict = dict()

    sals_volume_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in items_sales_vol_dict):
            sals_volume_list[index] = items_sales_vol_dict[item_id]
            continue

        sales_vol = get_item_sale_vol(window_start_date, window_end_date, item_id)
        sals_volume_list[index] = sales_vol
        items_sales_vol_dict[item_id] = sales_vol

    logging.info("leaving feature_item_sals_volume")

    return sals_volume_list

################################################################################################
################################################################################################
################################################################################################
def get_multiple_buy_ratio(window_start_date, window_end_date, item_id):    
    if (item_id not in g_user_buy_transection_item):
        return 0

    multiply_buy_user = dict()
    for user_id, item_opt_records in g_user_buy_transection_item[item_id].items():
        for each_record in item_opt_records:
            if (each_record[-1][1].date() >= window_start_date and each_record[-1][1].date() < window_end_date):
                if (user_id not in multiply_buy_user):
                    multiply_buy_user[user_id] = 1
                else:
                    multiply_buy_user[user_id] += 1

    multiply_buy_user_cnt = 0
    for user_id, buy_cnt in multiply_buy_user.items():
        if (buy_cnt > 1):
            multiply_buy_user_cnt += 1
    
    if (len(multiply_buy_user) > 0):
        return multiply_buy_user_cnt / len(multiply_buy_user)
    else:
        return 0

# [begin date, end date) 期间, 多次购买该item的用户的比例, 老客户率
def feature_multiple_buy_ratio(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_multiple_buy_ratio (%s, %s)" % (window_start_date, window_end_date))

    multiply_buy_dict = dict()

    multiply_buy_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        if (item_id in multiply_buy_dict):
            multiply_buy_list[index] = multiply_buy_dict[item_id]

        ratio = get_multiple_buy_ratio(window_start_date, window_end_date, item_id)

        multiply_buy_list[index] = ratio
        multiply_buy_dict[item_id] = ratio

        if (ratio > 0.0):
            logging.info("multiply buy ratio %.4f" % ratio)

    logging.info("leaving feature_multiple_buy_ratio")

    return multiply_buy_list


################################################################################################
################################################################################################
################################################################################################

def get_item_fav_cart_cnt(item_fav_cart_dict, featrue_cnt, window_start_date, window_end_date, item_records):
    for item_id, user_opt_records in item_records.items():
        if (item_id not in item_fav_cart_dict):
            # 0- 23 为item在各个小时上 fav 的数量， 24-27 为 cart 的数量
            item_fav_cart_dict[item_id] = [0 for x in range(featrue_cnt)]

        for user_id, user_opt_item_records in user_opt_records.items():
            for each_record in user_opt_item_records:
                for each_behavior in each_record:
                    if (each_behavior[1].date() >= window_start_date and 
                        each_behavior[1].date() < window_end_date):
                        if (each_behavior[0] == BEHAVIOR_TYPE_FAV):
                            item_fav_cart_dict[item_id][each_behavior[1].hour] += 1
                        elif (each_behavior[0] == BEHAVIOR_TYPE_CART):
                            item_fav_cart_dict[item_id][24 + each_behavior[1].hour] += 1

# [window_start_date, window_end_date) 范围内，item 在24 个小时上的收藏和加购物车数
# 返回48 个特征
def feature_item_fav_cart_in_24H(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_item_fav_cart_in_24H(%s, %s)" % (window_start_date, window_end_date))
    featrue_cnt = 48
    item_fav_cart_dict = dict()
    item_fav_cart_list = np.zeros((len(user_item_pairs), featrue_cnt))

    get_item_fav_cart_cnt(item_fav_cart_dict, featrue_cnt, window_start_date, window_end_date, g_user_buy_transection_item)
    get_item_fav_cart_cnt(item_fav_cart_dict, featrue_cnt, window_start_date, window_end_date, g_user_behavior_patten_item)
    
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        item_fav_cart_list[index] = item_fav_cart_dict[item_id]

        # logging.info("feature_item_fav_cart_in_24H item %s: %s" % (item_id, item_fav_cart_list[index]))

    return item_fav_cart_list
######################################################################################################
######################################################################################################
######################################################################################################
