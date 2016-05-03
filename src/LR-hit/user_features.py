from common import *
import numpy as np
from LR_common import *

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

            buy_count += len(buy_records)

        how_many_buy_dict[(user_id, item_category)] = buy_count
        how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]
        logging.info("as of %s, %s bought category %s %d" % (checking_date, user_id, item_category, buy_count)) 

    return how_many_buy

# 用户最后一次操作同类型的商品至 checking_date（不包括） 的天数，
def get_last_opt_category_date(user_records, checking_date, behavior_type, user_id, item_category):
    days = 0

    if (user_id not in user_records):
        return days

    last_opt_date = datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date()

    for item_id_can, item_opt_records in user_records[user_id].items():
        # 不属于同一个 category， skip
        if (item_id_can not in global_train_item_category or\
            global_train_item_category[item_id_can] != item_category):
            continue

        for each_record in item_opt_records:
            for index in range(len(each_record)-1, -1, -1):
                if (each_record[index][0] != behavior_type):
                    continue

                #each_record 已经按照时间排好序
                if (each_record[index][1].date() >= last_opt_date and \
                    each_record[index][1].date() < checking_date):
                    last_opt_date = each_record[index][1].date()
                    days = (checking_date - last_opt_date).days
                    break

    return days   

# 用户最后一次操作同类型的商品至 checking_date（不包括） 的天数，
def feature_last_opt_category(checking_date, behavior_type, user_item_pairs):
    days_from_last_opt_cat_dict = dict()
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]

        if ((user_id, item_category) in days_from_last_opt_cat_dict):
            days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
            continue

        days_from_last_opt_cat_dict[(user_id, item_category)] = 0

        days = get_last_opt_category_date(g_user_buy_transection, checking_date, behavior_type, user_id, item_category)
        if (behavior_type != BEHAVIOR_TYPE_BUY and user_id in g_user_behavior_patten):
            days2 = get_last_opt_category_date(g_user_behavior_patten, checking_date, behavior_type, user_id, item_category)
            if (days ==0):
                if (days2 != 0):
                    days = days2
            elif (days2 != 0):
                days = min(days, days2)

        days_from_last_opt_cat_dict[(user_id, item_category)] = days

        days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
        logging.info("%s last opted category %s with %d, days %d to %s" % \
                     (user_id, item_category, behavior_type, days_from_last_opt_cat_dict[(user_id, item_category)], checking_date))

    return days_from_last_opt_cat_list

# 用户最后一次操作 item 至 checking_date(不包括）) 的天数，若没有同类型的操作则返回 0
def get_last_opt_item_date(user_records, checking_date, behavior_type, user_id, item_id):
    days = 0
    if (user_id not in user_records or \
        item_id not in user_records[user_id]):
        return days

    last_opt_date = datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date()
    
    for each_record in user_records[user_id][item_id]:
        for index in range(len(each_record)-1, -1, -1):
            if (each_record[index][0] != behavior_type):
                continue

            #each_record 已经按照时间排好序
            if (each_record[index][1].date() >= last_opt_date and \
                each_record[index][1].date() < checking_date):            
                last_opt_date = each_record[index][1].date()
                days = (checking_date - last_opt_date).days
            break 

    return days

# 用户最后一次操作 item 至 checking_date(包括）) 的天数，
def feature_last_opt_item(checking_date, behavior_type, user_item_pairs):
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        days = get_last_opt_item_date(g_user_buy_transection, checking_date, \
                                               behavior_type, user_id, item_id)
        if (behavior_type != BEHAVIOR_TYPE_BUY):
            days2 = get_last_opt_item_date(g_user_behavior_patten, checking_date, \
                                           behavior_type, user_id, item_id)
            if (days ==0):
                if (days2 != 0):
                    days = days2
            elif (days2 != 0):
                days = min(days, days2)

        days_from_last_opt_cat_list[index] =  days

        logging.info("user %s last opted item %s with %d, days %d to %s" % \
            (user_id, item_id, behavior_type, days_from_last_opt_cat_list[index], checking_date))

    return days_from_last_opt_cat_list

#用户第一次购买 item 前的各个 behavior 数
def get_behavior_cnt_before_date(user_records, behavior_type, before_date, user_id, item_id):
    if (user_id not in user_records or 
        item_id not in user_records[user_id]):
        return 0

    behavior_cnt = 0
    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() <= before_date and 
                behavior_consecutive[0] == behavior_type):
                behavior_cnt += 1

    return behavior_cnt

#用户第一次购买 item 前的各个 behavior 数
def feature_behavior_cnt_before_1st_buy(behavior_type, user_item_pairs):
    behavior_cnt_before_1st_buy_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in g_user_buy_transection or 
            item_id not in g_user_buy_transection[user_id]):
            continue

        first_buy_date = datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date()
        for each_record in g_user_buy_transection[user_id][item_id]:
            if (first_buy_date < each_record[-1][1].date()):
                first_buy_date = each_record[-1][1].date()

        behavior_cnt = get_behavior_cnt_before_date(g_user_buy_transection, behavior_type, \
                                                    first_buy_date, user_id, item_id)

        behavior_cnt += get_behavior_cnt_before_date(g_user_behavior_patten, behavior_type, \
                                                     first_buy_date, user_id, item_id)
        behavior_cnt_before_1st_buy_list[index] = behavior_cnt
        logging.info("%s (%s, %s) 1st buy %s, behavior %d cnt  %d" % (getCurrentTime(), user_id, item_id, \
                     first_buy_date, behavior_type, behavior_cnt))

    return behavior_cnt_before_1st_buy_list



# user 对 item 购买间隔的平均天数
def mean_days_between_buy(checking_date, user_item_pairs):
    samle_cnt = len(user_item_pairs)
    for user_item in user_item_pairs:
        user_id = user_item[0]
        item_id = user_item[1]
    return 