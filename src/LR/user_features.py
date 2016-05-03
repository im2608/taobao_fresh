from common import *
import numpy as np
from LR_common import *


####################################################################################################
################################  数据集中，用户的特征  ##############################################
####################################################################################################


#截止到 checking_date（不包括）， 用户一共购买过多少同类型的商品
def feature_how_many_buy(checking_date, user_item_pairs):
    how_many_buy = np.zeros((len(user_item_pairs), 1))

    how_many_buy_dict = {}

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
                if (each_record[-1][1].date() < checking_date):
                    buy_count += 1

        how_many_buy_dict[(user_id, item_category)] = buy_count
        how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]

    return how_many_buy

#在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车
def feature_how_many_behavior(begin_date, end_date, need_ratio, user_item_pairs):
    logging.info("entered feature_how_many_behavior(%s, %s, %d)" % (begin_date, end_date, need_ratio))
    features = 4
    if (need_ratio):
        features = 7

    how_many_behavior_list = np.zeros((len(user_item_pairs), features))
    how_many_behavior_dict = dict()

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in how_many_behavior_dict):
            how_many_behavior_list[index] = how_many_behavior_dict[user_id]

        #if feature == 7, 前4 个为浏览，收藏，购物车, 购买的数量， 后3个为比例
        behavior_cnt = [0 for x in range(features)]

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

        if (need_ratio):
            for behavior_index in range(3):
                if (behavior_cnt[behavior_index] != 0):
                    behavior_cnt[behavior_index + 4] = round(behavior_cnt[3] / (behavior_cnt[3] + behavior_cnt[behavior_index]), 4)

        how_many_behavior_list[index] = behavior_cnt
        how_many_behavior_dict[user_id] = behavior_cnt
        logging.info("behavior count %s %s (%s -- %s)" % (user_id, behavior_cnt, begin_date, end_date))

    logging.info("leaving feature_how_many_behavior")
    return how_many_behavior_list

# 用户在 checking date（不包括） 之前每次购买日期距 checking date 的天数的平均值和方差
def feature_mean_days_between_buy_user(checking_date, user_item_pairs):
    logging.info("entered feature_mean_days_between_buy_user(%s)" % checking_date)
    mean_days_between_buy_dict = dict()
    mean_days_between_list = np.zeros((len(user_item_pairs), 2))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in mean_days_between_buy_dict):
            mean_days_between_list[index] = mean_days_between_buy_dict[user_id]
            continue

        days_to_checking_date = []
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                if (each_record[-1][1].date() < checking_date):
                    days_to_checking_date.append((checking_date - each_record[-1][1].date()).days)

        mean_vairance = [0, 0]
        if (len(days_to_checking_date) > 0):            
            mean_vairance[0] = np.round(np.mean(days_to_checking_date), 4)
            mean_vairance[1] = np.round(np.var(days_to_checking_date), 4)

        mean_days_between_buy_dict[user_id] = mean_vairance
        mean_days_between_list[index] = mean_days_between_buy_dict[user_id]

        logging.info("%s %s, %d" % (user_id, mean_days_between_list[index], len(days_to_checking_date)))

    logging.info("leaving feature_mean_days_between_buy_user")

    return mean_days_between_list

# 用户最后一次购买至 checking date（不包括）的天数
def feature_last_buy_user(checking_date, user_item_pairs):
    logging.info("entered feature_last_buy_user(%s)" % checking_date)

    last_buy_dict = dict()
    last_buy_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        if (user_id in last_buy_dict):
            last_buy_list[index] = last_buy_dict[user_id]
            continue
        
        last_buy_date = checking_date
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                if (last_buy_date > each_record[-1][1].date()):
                    last_buy_date = each_record[-1][1].date()

        last_buy_dict[user_id] = (checking_date - last_buy_date).days
        last_buy_list[index] = last_buy_dict[user_id]
        logging.info("%s last buy %s / %d days" % (user_id, last_buy_date, last_buy_list[index]))

    logging.info("leaving feature_last_buy_user")

    return last_buy_list

#截止到checking_date（不包括）， 用户有多少天进行了各种类型的操作
# 返回 4 个特征
def feature_how_many_days_for_behavior(checking_date, user_item_pairs):    
    logging.info("feature_how_many_days_for_behavior %s" % checking_date)
    hwo_many_days_for_behavior_dict = dict()
    hwo_many_days_for_behavior_list = np.zeros((len(user_item_pairs), 4))

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

        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            for each_record in item_buy_records:
                for each_behavior in each_record:
                    if (each_behavior[1].date() < checking_date):
                        days_for_behavior_dict[each_behavior[0]].add(each_behavior[1].date())

        if (user_id in g_user_behavior_patten):
            for item_id, item_opt_records in g_user_behavior_patten[user_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        if (each_behavior[1].date() < checking_date):
                            days_for_behavior_dict[each_behavior[0]].add(each_behavior[1].date())

        days_for_behavior = [len(days_for_behavior_dict[x]) for x in range(1, 5)]

        hwo_many_days_for_behavior_list[index] = days_for_behavior
        hwo_many_days_for_behavior_dict[user_id] = days_for_behavior
        logging.info("how many dasy for behavior %s %s" % (user_id, days_for_behavior))

    logging.info("leaving feature_how_many_days_for_behavior")

    return hwo_many_days_for_behavior_list

# 返回 user id 在 [week begin, week end) 内的购物列表
def buy_list_in_week(user_id, week_begin, week_end):
    buy_set = set()
    for item_id, item_buy_records in g_user_buy_transection[user_id].items():
        for each_record in item_buy_records:
            if (week_begin <= each_record[-1][1].date()  and each_record[-1][1].date() < week_end):
                buy_set.add(item_id)

    logging.info("%s bought %s from %s to %s" % (user_id, buy_set, week_begin, week_end))
    return buy_set

# 截止到checking_date（不包括），
# 用户A有1周购买的商品有多少种
# 用户A有2周购买的商品有多少种
# 用户A有3周购买的商品有多少种
# 用户A有4周购买的商品有多少种
# 返回 4 个特征
def feature_how_many_buy_in_weeks(checking_date, user_item_pairs):
    logging.info("feature_how_many_buy_in_weeks %s " % checking_date)

    how_many_buy_in_weeks_dict = dict()
    how_many_buy_in_weeks_list = np.zeros((len(user_item_pairs), 4))

    week_end = checking_date
    weeks_range = []
    # 得到 4 周的始末时间
    for i in range(4):
        week_begin = week_end - datetime.timedelta(7)
        weeks_range.append((week_begin, week_end))
        week_end = week_begin

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]

        if (user_id in how_many_buy_in_weeks_dict):
            how_many_buy_in_weeks_list[index] = how_many_buy_in_weeks_dict[user_id]
            continue

        #统计每个 item 出现的次数， 1 -- 4 次
        buy_count_dict = dict()
        for each_week in weeks_range:
            buy_set_for_week =buy_list_in_week(user_id, each_week[0], each_week[1])
            for item_id in buy_set_for_week:
                if (item_id not in buy_count_dict):
                    buy_count_dict[item_id] = 0
                buy_count_dict[item_id] += 1

        buy_count_for_weeks = [0, 0, 0, 0]
        for item_id in buy_count_dict:
            buy_count_for_weeks[buy_count_dict[item_id] - 1] += 1

        how_many_buy_in_weeks_dict[user_id] = buy_count_for_weeks
        how_many_buy_in_weeks_list[index] = buy_count_for_weeks

        logging.info("user %s %s" % (user_id, buy_count_for_weeks))

    logging.info("leaving feature_how_many_buy_in_weeks")

    return how_many_buy_in_weeks_list
