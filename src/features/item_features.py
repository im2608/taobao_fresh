from common import *
import numpy as np
#import time


################################################################################################
################################################################################################
################################################################################################
# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
def feature_item_popularity(behavior_type, item_popularity_dict, user_item_pairs):
    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id][behavior_type]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return item_popularity_list
################################################################################################
################################################################################################
################################################################################################

def feature_item_popularity2(item_popularity_dict, user_item_pairs):
    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]
        item_popularity_list[index] = item_popularity_dict[item_id]
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return item_popularity_list

################################################################################################
################################################################################################
################################################################################################

# 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数
# 返回 4 个特征
def feature_beahvior_cnt_on_item(begin_date, checking_date, user_item_pairs):

    total_start = time.clock()

    logging.info("feature_beahvior_cnt_on_item(%s, %s)" % (begin_date, checking_date))
    item_behavior_cnt_dict = dict()
    item_behavior_cnt_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in item_behavior_cnt_dict):
            item_behavior_cnt_list[index] = item_behavior_cnt_dict[item_id]
            continue

        behavior_cnt = [0,0,0,0]
        if (item_id in g_user_buy_transection_item):
            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        if (begin_date <= each_behavior[1].date() and each_behavior[1].date() < checking_date):
                            behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        if (item_id in g_user_behavior_patten_item):
            for user_id, item_opt_records in g_user_behavior_patten_item[item_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        if (begin_date <= each_behavior[1].date() and each_behavior[1].date() < checking_date):
                            behavior_cnt[each_behavior[0] - 1] += each_behavior[2]

        item_behavior_cnt_list[index] = behavior_cnt
        item_behavior_cnt_dict[item_id] = behavior_cnt

        logging.info("item %s, behavior %s from %s to %s" % (item_id, behavior_cnt, begin_date, checking_date))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    total_end = time.clock()
    total_time = total_end - total_start

    print("+++++++++++++++++ total time %.2f" % total_time)

    logging.info("leaving feature_beahvior_cnt_on_item")
    return item_behavior_cnt_list


################################################################################################
################################################################################################
################################################################################################

# item 第一次behavior 距离checking date 的天数, 返回 4 个特征
def feature_days_from_1st_behavior(checking_date, user_item_pairs):
    total_start = time.clock()
    days_from_1st_dict = dict()
    days_from_1st_list = np.zeros((len(user_item_pairs), 4))

    part1_time = 0.0
    part11_time = 0.0
    part12_time = 0.0

    part2_time = 0.0
    part21_time = 0.0
    part22_time = 0.0

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_1st_dict):
            days_from_1st_list[index] = days_from_1st_dict[item_id]
            continue

        days_from_1st_behavior = [checking_date for x in range(4)]

        part1_start = time.clock()
        part11_start = part1_start
        if (item_id in g_user_buy_transection_item):
            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        behavior_date = each_behavior[1].date()
                        if (behavior_date < days_from_1st_behavior[each_behavior[0] - 1]):
                            days_from_1st_behavior[each_behavior[0] - 1] = behavior_date

        part11_end = time.clock()
        part11_time += part11_end - part11_start

        part12_start = part11_end

        if (item_id in g_user_behavior_patten_item):
            for user_id, item_opt_records in g_user_behavior_patten_item[item_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        behavior_date = each_behavior[1].date()
                        if (behavior_date < days_from_1st_behavior[each_behavior[0] - 1]):
                            days_from_1st_behavior[each_behavior[0] - 1] = behavior_date

        part12_end = time.clock()
        part1_end = part12_end

        part12_time += part12_end - part12_start
        part1_time += part1_end - part1_start

        part2_start = time.clock()
        part21_start = part2_start
        #days_from_1st_behavior = list(map(lambda x: (checking_date - x).days, days_from_1st_behavior))
        for index in range(len(days_from_1st_behavior)):
            days_from_1st_behavior[index] = (checking_date - days_from_1st_behavior[index]).days
        part21_end = time.clock()
        part21_time += part21_end - part21_start

        part22_start = part21_end
        days_from_1st_list[index] = days_from_1st_behavior
        days_from_1st_dict[item_id] = days_from_1st_behavior

        logging.info("item %s days from 1st behavior to %s: %s " % (item_id, checking_date, days_from_1st_behavior))
        part22_end = time.clock()
        part2_end = part22_end

        part22_time += part22_end - part22_start 
        part2_time += part2_end - part2_start

        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    total_end = time.clock()
    total_time = total_end - total_start
    print("=============== part 1-1 %.2f/%.2f, part 1-2 %.2f/%.2f, part 2-1 %.2f/%.2f  total time %.2f" %
          (part11_time, part1_time, part12_time, part1_time, part21_time, part2_time, total_time))

    return days_from_1st_list

################################################################################################
################################################################################################
################################################################################################


# item 最后一次behavior 距离checking date 的天数, 返回 4 个特征
def feature_days_from_last_behavior(checking_date, user_item_pairs):
    days_from_last_dict = dict()
    days_from_last_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        item_id = user_item_pairs[index][1]

        if (item_id in days_from_last_dict):
            days_from_last_list[index] = days_from_last_dict[item_id]
            continue

        days_from_last_behavior = [None for x in range(4)]

        if (item_id in g_user_buy_transection_item):
            for user_id, item_buy_records in g_user_buy_transection_item[item_id].items():
                for each_record in item_buy_records:
                    for each_behavior in each_record:
                        if (days_from_last_behavior[each_behavior[0] - 1] == None):
                            days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()
                        else:
                            if (each_behavior[1].date() > days_from_last_behavior[each_behavior[0] - 1] and \
                                each_behavior[1].date() < checking_date):
                                days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        if (item_id in g_user_behavior_patten):
            for user_id, item_opt_records in g_user_behavior_patten[item_id].items():
                for each_record in item_opt_records:
                    for each_behavior in each_record:
                        if (days_from_last_behavior[each_behavior[0] - 1] == None):
                            days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()
                        else:
                            if (each_behavior[1].date() > days_from_last_behavior[each_behavior[0] - 1] and \
                                each_behavior[1].date() < checking_date):
                                days_from_last_behavior[each_behavior[0] - 1] = each_behavior[1].date()

        # days_from_last_behavior = list(map(lambda x: (checking_date - x).days, days_from_last_behavior))
        for index in range(len(days_from_last_behavior)):
            if (days_from_last_behavior[index] != None):
                days_from_last_behavior[index] = (checking_date - days_from_last_behavior[index]).days
            else:
                days_from_last_behavior[index] = 0

        days_from_last_list[index] = days_from_last_behavior
        days_from_last_dict[item_id] = days_from_last_behavior
        logging.info("item %s days from last behavior to %s: %s " % (item_id, checking_date, days_from_last_behavior))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return days_from_last_list
################################################################################################
################################################################################################
################################################################################################