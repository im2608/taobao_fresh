from common import *
import numpy as np
from feature_selection import *


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# 某个用户某天对item 是否有 behavior_type 操作
def get_behavior_by_date(user_records, behavior_type, checking_date, user_item_pair):
    user_id = user_item_pair[0]
    item_id = user_item_pair[1]
    if (user_id not in user_records or 
        item_id not in user_records[user_id]):
        return 0

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
                if (behavior_consecutive[1].date() == checking_date and \
                    behavior_consecutive[0] == behavior_type):
                    return 1
    return 0


#检查user 是否在 checking_date 这一天对 item 有过 behavior type
def feature_behavior_on_date(behavior_type, checking_date, user_item_pairs):    
    feature_name = "feature_behavior_on_date_%d" % behavior_type

    returned_feature_cnt = 1

    does_operated = np.zeros((len(user_item_pairs), 1))
    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        #检查在购物记录中, 在 checking date 是否有过 behavior type 操作
        does_operated[index][0] = get_behavior_by_date(g_user_buy_transection, behavior_type, checking_date, user_item_pairs[index])
        if (does_operated[index][0] == 1):
            continue

        #检查在 patterns 中, 在 checking date 是否有过 behavior type 操作
        does_operated[index][0] = get_behavior_by_date(g_user_behavior_patten, behavior_type, checking_date, user_item_pairs[index])
        if (index % 1000 == 0):            
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return does_operated

######################################################################################################
######################################################################################################
######################################################################################################

# 用户行为统计
# 用户在checking_date之前 [begin_date, end_date)(不包括end_date) 天购买（浏览， 收藏， 购物车）所有商品的总次数, 以及用户在category
# 上的各个行为的总次数
def get_behavior_cnt_of_days(user_records, begin_date, end_date, total_behavior_cnt, user_id):
    if (user_id not in user_records):
        return

    for item_id, item_opt_records in user_records[user_id].items():
        for each_record in item_opt_records:
            for behavior_consecutive in each_record:
                if (behavior_consecutive[1].date() >= begin_date and 
                    behavior_consecutive[1].date() < end_date):
                    total_behavior_cnt[behavior_consecutive[0] - 1] += behavior_consecutive[2]
    return

# user id 在 checking date(不包括) 之前 pre_days 天对 item id 进行的behavior type 操作的次数
def userBehaviorCntOnItemBeforeCheckingDate(user_records, user_id, item_id, checking_date, pre_days, behavior_cnt):
    if (user_id not in user_records or item_id not in user_records[user_id]):
        return

    begin_date = checking_date - datetime.timedelta(pre_days)

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() >= begin_date and
                behavior_consecutive[1].date() < checking_date):
                behavior_cnt[behavior_consecutive[0] - 1] += behavior_consecutive[2]
    return

# 用户checking_date（不包括）之前 pre_days 天浏览， 收藏， 购物车， 购买该商品的次数, 这些次数占该用户购买（浏览， 收藏， 购物车）所有商品的总次数的比例,
# 用户在item上pre_days 天购买/浏览，  
# 用户在item上pre_days 天购买/购物车， 
# 用户在item上 pre_days 天购买/收藏， 
# 用户在item上 pre_days 天购物车/收藏， 
# 用户在item上pre_days 天购物车/浏览
# 用户在item上的(浏览， 收藏， 购物车， 购买）的次数 * 用户的购买率
# 返回 17 个特征
def feature_user_item_behavior_ratio(checking_date, pre_days, user_buy_ratio, user_item_pairs):
    logging.info("feature_user_item_behavior_ratio %d, %s" % (pre_days, checking_date))

    user_item_pop_list = np.zeros((len(user_item_pairs), 17))
    user_behavior_cnt_item_dict = dict()

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        begin_date = checking_date - datetime.timedelta(pre_days)

        user_total_behavior_cnt = [0, 0, 0, 0]
        if ((user_id, item_id) in user_behavior_cnt_item_dict):
            user_total_behavior_cnt = user_behavior_cnt_item_dict[(user_id, item_id)]
        else:
            get_behavior_cnt_of_days(g_user_buy_transection, begin_date, checking_date, user_total_behavior_cnt, user_id)
            get_behavior_cnt_of_days(g_user_behavior_patten, begin_date, checking_date, user_total_behavior_cnt, user_id)
            user_behavior_cnt_item_dict[(user_id, item_id)] = user_total_behavior_cnt

        user_behavior_cnt_on_item = [0, 0, 0, 0]

        #user 在 item 上的点击数， 购物记录中也包含了其他类型的操作, 所以要先过一遍 buy records
        userBehaviorCntOnItemBeforeCheckingDate(g_user_buy_transection, user_id, item_id, checking_date, pre_days, user_behavior_cnt_on_item)
        userBehaviorCntOnItemBeforeCheckingDate(g_user_behavior_patten, user_id, item_id, checking_date, pre_days, user_behavior_cnt_on_item)

        behavior_cnt_ratio = [0 for x in range(13)]
        for behavior in range(len(user_behavior_cnt_on_item)):
            if (user_total_behavior_cnt[behavior] != 0):
                behavior_cnt_ratio[behavior] = round(user_behavior_cnt_on_item[behavior] / user_total_behavior_cnt[behavior], 4)

        # 用户在item上 pre_days 天购买/浏览， ，   
        if ( user_behavior_cnt_on_item[0] > 0):
            behavior_cnt_ratio[4] = user_behavior_cnt_on_item[3] / user_behavior_cnt_on_item[0]

        # 用户在item上pre_days 天购买/购物车
        if ( user_behavior_cnt_on_item[1] > 0):
            behavior_cnt_ratio[5] = user_behavior_cnt_on_item[3] / user_behavior_cnt_on_item[1]

        # 用户在item上 pre_days 天购买/收藏，
        if ( user_behavior_cnt_on_item[2] > 0):
            behavior_cnt_ratio[6] = user_behavior_cnt_on_item[3] / user_behavior_cnt_on_item[2]

        # 用户在item上 pre_days 天购物车/收藏，
        if (user_behavior_cnt_on_item[1] > 0):
            behavior_cnt_ratio[7] = user_behavior_cnt_on_item[2] / user_behavior_cnt_on_item[1]

        # 用户在item上 pre_days 天购物车/浏览
        if (user_behavior_cnt_on_item[0] > 0):    
            behavior_cnt_ratio[8] = user_behavior_cnt_on_item[2] / user_behavior_cnt_on_item[0]

        # 用户在item上的(浏览， 收藏， 购物车， 购买）的次数 * 用户的购买率
        for i in range(4):
            behavior_cnt_ratio[i + 9] = user_behavior_cnt_on_item[i] * user_buy_ratio[index]

        user_behavior_cnt_on_item.extend(behavior_cnt_ratio)
 
        user_item_pop_list[index] = user_behavior_cnt_on_item

        # logging.info("(%s, %s), %s - %s , behavior cnt %s / %s" %
        #              (user_id, item_id, begin_date, checking_date, user_behavior_cnt_on_item, user_total_behavior_cnt))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    user_item_pop_list = preprocessing.scale(user_item_pop_list)

    return user_item_pop_list


######################################################################################################
######################################################################################################
######################################################################################################

# 用户第一次，最后一次操作 item 至 checking_date(不包括）) 的天数，若没有同类型的操作则返回 0
def get_user_item_1st_last_behavior_date(window_start_date, window_end_date, user_id, item_id, user_records):
    if (user_id not in user_records or \
        item_id not in user_records[user_id]):
        return [0 for x in range(8)]

    days_from_1st_behavior = [window_end_date for x in range(4)]
    days_from_last_behavior = [None for x in range(4)]

    for each_record in user_records[user_id][item_id]:
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



def get_last_opt_item_date(user_records, window_start_date, window_end_date, user_id, item_id):
    days = [0, 0, 0, 0]

    if (user_id not in user_records or \
        item_id not in user_records[user_id]):
        return days

    last_opt_date = [window_start_date for x in range(4)]

    for each_record in user_records[user_id][item_id]:
        for index in range(len(each_record)-1, -1, -1):
            #each_record 已经按照时间排好序
            behavior_type = each_record[index][0]
            behavior_date = each_record[index][1].date()
            if (behavior_date > last_opt_date[behavior_type - 1] and \
                behavior_date < window_end_date):            
                last_opt_date[behavior_type - 1] = behavior_date
                days[behavior_type - 1] = (window_end_date - behavior_date).days
                break 
    return days


# 用户第一次，最后一次操作 item 至 checking_date(不包括) 的天数，以及在item上最后一次cart 至最后一次buy之间的天数, 返回13个特征
def feature_user_item_1stlast_opt(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_last_opt_item (%s, %s)" % (window_start_date, window_end_date))

    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 13))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        # 得到用户第一次，最后一次操作 item 至 checking_date(不包括) 的天数，
        # 共 8 项， 前4 项为第一次行为至end date的天数，后4项为最后一次行为至 end date的天数
        days_from_1st_last_1 = get_user_item_1st_last_behavior_date(window_start_date, window_end_date, user_id, item_id, g_user_buy_transection)
        days_from_1st_last_2 = get_user_item_1st_last_behavior_date(window_start_date, window_end_date, user_id, item_id, g_user_behavior_patten)

        # days = get_last_opt_item_date(g_user_buy_transection, window_start_date, window_end_date, user_id, item_id)
        # days2 = get_last_opt_item_date(g_user_behavior_patten, window_start_date, window_end_date, user_id, item_id)
        # for index in range(4):
        #     if (days[index] == 0):
        #         if (days2 != 0):
        #             days[index] = days2[index]
        #     elif (days2[index] != 0):
        #         days[index] = min(days[index], days2[index])

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

        # 用户在 item上最后一次cart 至最后一次buy之间的天数
        days_between_last_cart_buy = 0
        if (days_from_1st_last_1[7] > 0):
            days_between_last_cart_buy = days_from_1st_last_1[7] - days_from_1st_last_1[6]

        # 用户第一次，最后一次操作 item 之间的天数
        days_between_1st_last = [0, 0, 0, 0]
        for i in range(4):
            days_between_1st_last[i] = days_from_1st_last_1[i] - days_from_1st_last_1[i+4]

        days_from_1st_last_1.extend(days_between_1st_last)
        days_from_1st_last_1.append(days_between_last_cart_buy)

        days_from_last_opt_cat_list[index] =  days_from_1st_last_1

        # logging.info("user (%s, %s) itme %s" % (user_id, item_id, days_from_1st_last_1))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    days_from_last_opt_cat_list = preprocessing.scale(days_from_last_opt_cat_list)

    return days_from_last_opt_cat_list

######################################################################################################
######################################################################################################
######################################################################################################

#用户第一次购买 item 前的各个 behavior 数
def get_behavior_cnt_before_date(user_records, before_date, user_id, item_id, behavior_cnt):
    if (user_id not in user_records or 
        item_id not in user_records[user_id]):
        return

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() <= before_date and 
                behavior_consecutive[0] != BEHAVIOR_TYPE_BUY):
                behavior_cnt[behavior_consecutive[0] - 1] += behavior_consecutive[2]
    return

#用户第一次购买 item 前， 在 item 上的的各个 behavior 的数量, 3个特征
def feature_behavior_cnt_before_1st_buy(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_behavior_cnt_before_1st_buy(%s, %s)" % (window_start_date, window_end_date))

    features_names = ["feature_behavior_cnt_before_1st_buy_view", 
                      "feature_behavior_cnt_before_1st_buy_fav", 
                      "feature_behavior_cnt_before_1st_buy_cart"]

    behavior_cnt_before_1st_buy_list = np.zeros((len(user_item_pairs), len(features_names)))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in g_user_buy_transection or
            item_id not in g_user_buy_transection[user_id]):
            continue

        first_buy_date = None
        for each_record in g_user_buy_transection[user_id][item_id]:
            if ((first_buy_date is None or first_buy_date < each_record[-1][1].date()) and 
                each_record[-1][1].date() < window_end_date):
                first_buy_date = each_record[-1][1].date()

        if (first_buy_date is None):
            # logging.info("%s has not bought %s in %s to %s" % (user_id, item_id, window_start_date, window_end_date))
            continue

        behavior_cnt = [0, 0, 0]

        get_behavior_cnt_before_date(g_user_buy_transection, first_buy_date, user_id, item_id, behavior_cnt)
        get_behavior_cnt_before_date(g_user_behavior_patten, first_buy_date, user_id, item_id, behavior_cnt)

        behavior_cnt_before_1st_buy_list[index] = behavior_cnt
        #logging.info("%s (%s, %s) 1st buy %s, behavior cnt %s" % (getCurrentTime(), user_id, item_id, first_buy_date, behavior_cnt))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    behavior_cnt_before_1st_buy_list = preprocessing.scale(behavior_cnt_before_1st_buy_list)

    return behavior_cnt_before_1st_buy_list

######################################################################################################
######################################################################################################
######################################################################################################

#在 [windw_start_date, window_end_dat) 范围内， user 对 item 购买间隔的平均天数
def feature_mean_days_between_buy_user_item(window_start_date, window_end_dat, user_item_pairs):
    feature_name = "feature_mean_days_between_buy_user_item"

    samle_cnt = len(user_item_pairs)
    buy_mean_days_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        if (user_id not in g_user_buy_transection or \
            item_id not in g_user_buy_transection[user_id]):
            continue

        buy_date = []
        for each_record in g_user_buy_transection[user_id][item_id]:
            if (each_record[-1][1].date() >= window_start_date and 
                each_record[-1][1].date() < window_end_dat):
                buy_date.append(each_record[-1][1].date())

        buy_date.sort()

        # 若只购买过一次，则将购买日期至checking date 作为平均天数
        if (len(buy_date) == 1):
            buy_date.append(window_end_dat)

        days = 0
        for date_index in range(1, len(buy_date)):
            days += (buy_date[date_index] - buy_date[date_index-1]).days

        buy_mean_days_list[index] = round(days / (len(buy_date) - 1))
        # logging.info("user %s, item %s, mean buy days %d" % (user_id, item_id, buy_mean_days_list[index]))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    buy_mean_days_list = preprocessing.scale(buy_mean_days_list)

    return buy_mean_days_list

######################################################################################################
######################################################################################################
######################################################################################################


