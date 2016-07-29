from common import *
import numpy as np
from feature_selection import *



####################################################################################################
################################  数据集中，用户的特征  ##############################################
####################################################################################################

# 距离 end_date pre_days 天内， 用户总共有过多少次浏览，收藏，购物车，购买的行为, 购买/浏览， 购买/收藏， 购买/购物车, 购物车/收藏， 购物车/浏览的比率,
# 返回 9 个特征
def feature_how_many_behavior_user(pre_days, end_date, user_item_pairs):
    begin_date = end_date - datetime.timedelta(pre_days)
    logging.info("entered feature_how_many_behavior_user(%s, %s)" % (begin_date, end_date))

    feature_cnt = 9

    how_many_behavior_list = np.zeros((len(user_item_pairs), feature_cnt))
    how_many_behavior_dict = dict()

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in how_many_behavior_dict):
            how_many_behavior_list[index] = how_many_behavior_dict[user_id]
            continue

        #前4 个为浏览，收藏，购物车, 购买的数量， 后5个为比例
        behavior_cnt = [0 for x in range(feature_cnt)]

        if (user_id in g_user_buy_transection):
            for item_id, item_buy_records in g_user_buy_transection[user_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        if (each_behavior[1].date() >= begin_date and \
                            each_behavior[1].date() < end_date):
                            behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        if (user_id in g_user_behavior_patten):
            for item_id, item_opt_records in g_user_behavior_patten[user_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        if (each_behavior[1].date() >= begin_date and \
                            each_behavior[1].date() < end_date):
                            behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        # 购买/浏览， 购买/收藏， 购买/购物车
        for behavior_index in range(3):
            if (behavior_cnt[behavior_index] != 0):
                behavior_cnt[behavior_index + 4] = round(behavior_cnt[3] / behavior_cnt[behavior_index], 4)

        # 购物车/浏览
        if (behavior_cnt[0] > 0):
            behavior_cnt[7] = behavior_cnt[2] / behavior_cnt[0]

        # 购物车/收藏
        if (behavior_cnt[1] > 0):
            behavior_cnt[7] = behavior_cnt[2] / behavior_cnt[1]

        how_many_behavior_list[index] = behavior_cnt
        how_many_behavior_dict[user_id] = behavior_cnt
        # logging.info("behavior count %s %s (%s -- %s)" % (user_id, behavior_cnt, begin_date, end_date))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    how_many_behavior_list[:, 0:4] = preprocessing.scale(how_many_behavior_list[:, 0:4])

    logging.info("leaving feature_how_many_behavior_user")
    return how_many_behavior_list


######################################################################################################
######################################################################################################
######################################################################################################

# 用户在 checking date（不包括） 之前每次购买间隔的天数的平均值和方差, 返回两个特征
def feature_mean_days_between_buy_user(window_start_date, window_end_date, user_item_pairs):
    logging.info("entered feature_mean_days_between_buy_user(%s, %s)" % (window_start_date, window_end_date))

    features_names = ["feature_mean_days_between_buy_user_mean", "feature_mean_days_between_buy_user_variance"]

    mean_days_between_buy_dict = dict()
    mean_days_between_list = np.zeros((len(user_item_pairs), 2))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in mean_days_between_buy_dict):
            mean_days_between_list[index] = mean_days_between_buy_dict[user_id]
            continue

        if (user_id not in g_user_buy_transection):
            continue

        buy_date = set()            
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                if (each_record[-1][1].date() >= window_start_date and 
                    each_record[-1][1].date() < window_end_date):
                    buy_date.add(each_record[-1][1].date())

        if (len(buy_date) == 0):
            continue

        buy_date = list(buy_date)
        buy_date.sort()

        # 若只购买过一次，则将购买日期至 end date 作为平均天数
        if (len(buy_date) == 1):
            buy_date.append(window_end_date)

        days_between_buy = []
        for date_index in range(1, len(buy_date)):
            days_between_buy.append((buy_date[date_index] - buy_date[date_index-1]).days)

        mean_vairance = [0, 0]
        mean_vairance[0] = np.round(np.mean(days_between_buy), 2)
        mean_vairance[1] = np.round(np.var(days_between_buy), 2)
        # logging.info("user mean days to buy: %s %s" % (user_id, mean_vairance))

        # 实际训练中去掉小数点
        mean_vairance[0] = np.round(np.mean(days_between_buy))
        mean_vairance[1] = np.round(np.var(days_between_buy))
        mean_days_between_buy_dict[user_id] = mean_vairance
        mean_days_between_list[index] = mean_days_between_buy_dict[user_id]

        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    mean_days_between_list = preprocessing.scale(mean_days_between_list)

    logging.info("leaving feature_mean_days_between_buy_user")
    return mean_days_between_list


######################################################################################################
######################################################################################################
######################################################################################################


# 用户最后一次行为至 window_end_date （不包括）的天数, 没有该行为则为 0, 返回4个特征
def get_last_opt_item_date(user_records, window_start_date, window_end_date, user_id):
    days = [0, 0, 0, 0]

    if (user_id not in user_records):
        return days

    last_opt_date = [window_start_date for x in range(4)]

    for item_kd, behavior_record in user_records[user_id].items():
        for each_record in behavior_record:
            for index in range(len(each_record)-1, -1, -1):
                #each_record 已经按照时间排好序
                behavior_type = each_record[index][0]
                behavior_date = each_record[index][1].date()
                if (behavior_date > last_opt_date[behavior_type - 1] and \
                    behavior_date < window_end_date):            
                    last_opt_date[behavior_type - 1] = behavior_date
                    days[behavior_type - 1] = (window_end_date - behavior_date).days
    return days


# [window_start_date, window_end_date) 期间， 用户最后一次行为至 window_end_date （不包括）的天数, 没有该行为则为 0, 返回4个特征
def feature_last_behavior_user(window_start_date, window_end_date, user_item_pairs):
    logging.info("entered feature_last_behavior_user(%s, %s)" % (window_start_date, window_end_date))

    last_behavior_user_dict = dict()
    last_behavior_user_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in last_behavior_user_dict):
            last_behavior_user_list[index] = last_behavior_user_dict[user_id]
            continue

        days = get_last_opt_item_date(g_user_buy_transection, window_start_date, window_end_date, user_id)
        days2 = get_last_opt_item_date(g_user_behavior_patten, window_start_date, window_end_date, user_id)
        for index in range(len(days)):
            if (days[index] == 0):
                if (days2 != 0):
                    days[index] = days2[index]
            elif (days2[index] != 0):
                days[index] = min(days[index], days2[index])

        last_behavior_user_dict[user_id] = days
        last_behavior_user_list[index] = days
        # logging.info("user %s, last behavior %s" % (user_id, days))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    logging.info("leaving feature_last_buy_user")

    last_behavior_user_list = preprocessing.scale(last_behavior_user_list)

    return last_behavior_user_list


######################################################################################################
######################################################################################################
######################################################################################################

#截止到checking_date（不包括）， 用户有多少天进行了各种类型的操作
# 返回 4 个特征
def feature_how_many_days_for_behavior(window_start_date, window_end_date, user_item_pairs):    
    logging.info("feature_how_many_days_for_behavior %s -- %s" % (window_start_date, window_end_date))

    features_names = ["feature_how_many_days_for_behavior_view", 
                      "feature_how_many_days_for_behavior_fav", 
                      "feature_how_many_days_for_behavior_cart", 
                      "feature_how_many_days_for_behavior_buy"]

    hwo_many_days_for_behavior_dict = dict()
    hwo_many_days_for_behavior_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in hwo_many_days_for_behavior_dict):
            hwo_many_days_for_behavior_list[index] = hwo_many_days_for_behavior_dict[user_id]
            continue

        days_for_behavior_dict = dict()
        days_for_behavior_dict[BEHAVIOR_TYPE_VIEW] = set()
        days_for_behavior_dict[BEHAVIOR_TYPE_FAV] = set()
        days_for_behavior_dict[BEHAVIOR_TYPE_CART] = set()
        days_for_behavior_dict[BEHAVIOR_TYPE_BUY] = set()

        if (user_id in g_user_buy_transection):
            for item_id, item_buy_records in g_user_buy_transection[user_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        if (each_behavior[1].date() >= window_start_date and
                            each_behavior[1].date() < window_end_date):
                            days_for_behavior_dict[each_behavior[0]].add(each_behavior[1].date())

        if (user_id in g_user_behavior_patten):
            for item_id, item_opt_records in g_user_behavior_patten[user_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        if (each_behavior[1].date() >= window_start_date and
                            each_behavior[1].date() < window_end_date):
                            days_for_behavior_dict[each_behavior[0]].add(each_behavior[1].date())

        days_for_behavior = [len(days_for_behavior_dict[x]) for x in range(1, 5)]

        hwo_many_days_for_behavior_list[index] = days_for_behavior
        hwo_many_days_for_behavior_dict[user_id] = days_for_behavior
        # logging.info("how many dasy for behavior %s %s" % (user_id, days_for_behavior))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    logging.info("leaving feature_how_many_days_for_behavior")

    return hwo_many_days_for_behavior_list


######################################################################################################
######################################################################################################
######################################################################################################

# 返回 user id 在 [week begin, week end) 内的购物列表
def buy_list_in_week(user_id, each_day, window_end_date):
    buy_set = set()
    for item_id, item_buy_records in g_user_buy_transection[user_id].items():
        for each_record in item_buy_records:
            if (window_start_date <= each_record[-1][1].date()  and each_record[-1][1].date() < window_end_date):
                buy_set.add(item_id)

    logging.info("%s bought %s from %s to %s" % (user_id, buy_set, window_start_date, window_end_date))
    return buy_set

# [start date, end date) 范围内，用户购买过 1/2/3/4 ... ... /slide window days 次的item有多少， 返回 slide window days 个特征
# 用户在同一天内多次购买同一个item算一次
# 例如 用户在 第1天购买了item1，item2， item3， 然后在第5天又购买了该item1, 第6 天购买了 item2， 第7 天购买了item3，第 8 天有购买了item3
# 用户购买过item1， item2两次，购买过item3 三次，则buy_in_days_list[2] = 2， buy_in_days_list[3] = 1
def feature_how_many_buy_in_days(window_start_date, window_end_date, user_item_pairs):    
    logging.info("feature_how_many_buy_in_days (%s, %s)" % (window_start_date, window_end_date))
    slide_window_days = (window_end_date - window_start_date).days
    features_names = []
    for day in range(1, slide_window_days + 1):
        features_names.append("feature_how_many_buy_in_days_%d" % day)    

    buy_in_days_list = np.zeros((len(user_item_pairs), slide_window_days))
    buy_in_days_dict = dict()

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]

        if (user_id in buy_in_days_dict):
            buy_in_days_list[index] = buy_in_days_dict[user_id]
            continue

        if (user_id not in g_user_buy_transection):
            continue

        #统计每个 item 出现的次数， 0 -- slide window day 次
        buy_in_days = [0 for x in range(slide_window_days)]

        #item 在哪天被user购买
        date_user_buy_items = dict()
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                buy_date = each_record[-1][1].date()
                if (buy_date >= window_start_date and buy_date < window_end_date):
                    if (item_id not in date_user_buy_items):
                        date_user_buy_items[item_id] = set()
                    date_user_buy_items[item_id].add(buy_date)

        for item_id, days_buy_item in date_user_buy_items.items():
            buy_in_days[len(days_buy_item) - 1] += 1

        buy_in_days_list[index] = buy_in_days
        buy_in_days_dict[user_id] = buy_in_days

        # logging.info("how many buy in days user %s %s" % (user_id, buy_in_days))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    logging.info("leaving feature_how_many_buy_in_weeks")

    return buy_in_days_list

######################################################################################################
######################################################################################################
######################################################################################################
