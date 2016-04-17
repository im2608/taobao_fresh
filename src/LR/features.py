from common import *
import numpy as np
from LR_common import *


# user id 在 checking date(不包括) 之前 pre_days 天对 item id 进行的behavior type 操作的次数
def userBehaviorCntOnItemBeforeCheckingDate(user_records, user_id, item_id, behavior_type, checking_date, pre_days):
    begin_date = checking_date - datetime.timedelta(pre_days)
    behavior_cnt = 0

    if (user_id not in user_records or item_id not in user_records[user_id]):
        return 0

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_type != behavior_consecutive[0]):
                continue

            if (behavior_consecutive[1].date() >= begin_date and
                behavior_consecutive[1].date() < checking_date):
                behavior_cnt += behavior_consecutive[2]

    logging.info("user %s, item %s, begin date %s, behavior type %d, count %d" % (user_id, item_id, begin_date, behavior_type, behavior_cnt))
    return behavior_cnt
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

#检查user 是否在 checking_date 这一天对 item 有过 behavior type
def feature_behavior_on_date(behavior_type, checking_date, user_item_pairs):
    does_operated = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        #检查在购物记录中, 在 checking date 是否有过 behavior type 操作
        does_operated[index][0] = get_behavior_by_date(g_user_buy_transection, behavior_type, checking_date, user_item_pairs[index])
        if (does_operated[index][0] == 1):
            continue

        #检查在 patterns 中, 在 checking date 是否有过 behavior type 操作
        does_operated[index][0] = get_behavior_by_date(g_user_behavior_patten, behavior_type, checking_date, user_item_pairs[index])

    return does_operated


# 某个用户某天对item 是否有 behavior_type 操作
def get_behavior_by_date(user_records, behavior_type, checking_date,user_item_pair):
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


# 得到用户在某商品上的购买浏览转化率 购买过的某商品数量/浏览过的商品数量
def feature_buy_view_ratio(user_item_pairs):
    buy_view_ratio = dict()

    buy_view_ratio_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in g_user_buy_transection):
            continue

        if (user_id in buy_view_ratio):
            buy_view_ratio_list[index, 0] = buy_view_ratio[user_id]
            continue

        buy_count = 0
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            buy_count += len(item_buy_records)

        # 没有pattern， 所有的view 都转化成了buy
        if (user_id not in g_user_behavior_patten):
            buy_view_ratio_list[index][0] = 1
            continue

        view_count = 0
        for item_id, item_patterns in g_user_behavior_patten[user_id].items():
            view_count += len(item_patterns)

        buy_view_ratio[user_id] = round(buy_count / (buy_count + view_count), 4)
        buy_view_ratio_list[index, 0] = buy_view_ratio[user_id]

    return buy_view_ratio_list

# 商品热度 购买该商品的用户数/购买同类型商品总用户数
def calculate_item_popularity():
    item_popularity_dict = dict()
    item_category_buy_cnt = dict()

    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id in item_buy_records:            
            item_category = getCatalogByItemId(item_id)
            if item_category is None:
                continue

            if (item_id not in item_popularity_dict):
                item_popularity_dict[item_id] = 0
            item_popularity_dict[item_id] += 1

            if (item_category not in item_category_buy_cnt):
                item_category_buy_cnt[item_category] = 0
            item_category_buy_cnt[item_category] += 1

    for item_id in item_popularity_dict:
        item_category = getCatalogByItemId(item_id)
        item_popularity_dict[item_id] = round(item_popularity_dict[item_id]/item_category_buy_cnt[item_category], 4)

    logging.info("item popularity %s" % item_popularity_dict)
    return item_popularity_dict

# 商品热度 购买该商品的用户数/购买同类型商品总用户数
def feature_item_popularity(item_popularity_dict, user_item_pairs):
    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (item_id in item_popularity_dict):
            item_popularity_list[index][0] = item_popularity_dict[item_id]
        else:
            item_popularity_list[index][0] = 0

    return item_popularity_list

# 用户行为统计
# 用户checking_date之前 pre_days 天购买（浏览， 收藏， 购物车）该商品的次数/用户购买（浏览， 收藏， 购物车）的总数量
def feature_user_item_behavior_ratio(checking_date, behavior_type, pre_days, user_item_pairs):
    user_item_pop = dict()
    user_item_pop_list = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        if (g_user_behavior_count[user_id][behavior_type] == 0):
            user_item_pop[index] = 0
            continue

        behavior_cnt = 0
        if (behavior_type == BEHAVIOR_TYPE_BUY):
            behavior_cnt = userBehaviorCntOnItemBeforeCheckingDate(g_user_buy_transection, \
                user_id, item_id, behavior_type, checking_date, pre_days)
        else:
            #购物记录中也包含了其他类型的操作, 所以要先过一遍 buy records
            behavior_cnt = userBehaviorCntOnItemBeforeCheckingDate(g_user_buy_transection, \
                user_id, item_id, behavior_type, checking_date, pre_days)
            behavior_cnt += userBehaviorCntOnItemBeforeCheckingDate(g_user_behavior_patten, \
                user_id, item_id, behavior_type, checking_date, pre_days)

        user_item_pop_list[index] = behavior_cnt / g_user_behavior_count[user_id][behavior_type]

    logging.info("checking date %s, behavior %d, pre days %d" % (checking_date, behavior_type, pre_days))
    logging.info(user_item_pop_list)
    return user_item_pop_list

#用户一共购买过多少同类型的商品 * 最后一天购买至今的天数的倒数
def feature_how_many_buy(last_buy, checking_date, user_item_pairs):
    how_many_buy = np.zeros((len(user_item_pairs), 1))

    how_many_buy_dict = {}

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = getCatalogByItemId(item_id)
        if item_category == None:
            continue

        if ((user_id, item_category) in how_many_buy_dict):
            how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]
            continue

        buy_count = 0
        for item_id_can, buy_records in g_user_buy_transection[user_id].items():
            # 不属于同一个 category， skip
            if (getCatalogByItemId(item_id_can) != item_category):
                continue

            buy_count += len(buy_records)

        how_many_buy_dict[(user_id, item_category)] = buy_count * last_buy[index]
        how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]

    return how_many_buy

# 用户最后一次操作同类型的商品至 checking_date 的天数，
# 天数需要取倒数
def get_last_opt_category_date(user_records, checking_date, behavior_type, user_item_pairs):
    last_opt_date = datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date()
    for item_id_can, item_opt_records in user_records[user_id].items():
        # 不属于同一个 category， skip
        if (getCatalogByItemId(item_id_can) != item_category):
            continue

        for each_record in item_opt_records:
            for index in range(len(each_record)-1, -1, -1):
                if (each_record[index][0] != behavior_type):
                    continue

                #each_record 已经按照时间排好序
                if (each_record[index][1].date() > last_opt_date and \
                    each_record[index][1].date() <= checking_date):
                    last_opt_date = each_record[index][1].date()
                    break 

    return last_opt_date   

def feature_last_opt_category(checking_date, behavior_type, user_item_pairs):
    days_from_last_opt_cat_dict = dict()
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = getCatalogByItemId(item_id)

        if ((user_id, item_category) in days_from_last_opt_cat_dict):
            days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
            continue

        days_from_last_opt_cat_dict[(user_id, item_category)] = 0

        last_opt_date = get_last_opt_category_date(g_user_buy_transection, checking_date, behavior_type, user_item_pairs)
        if (behavior_type != BEHAVIOR_TYPE_BUY):
            last_opt_date2 = get_last_opt_category_date(g_user_behavior_patten, checking_date, behavior_type, user_item_pairs)
            last_opt_date = max(last_opt_date, last_opt_date2)

        days_from_last_opt = checking_date - last_opt_date
        if (days_from_last_opt.days == 0):
            #同一天购买取 1
            days_from_last_opt_cat_dict[(user_id, item_category)] = 1
        else:
            #天数取倒数， 时间越远该数字越小
            days_from_last_opt_cat_dict[(user_id, item_category)] =  1/ days_from_last_opt.days

        days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
        logging.info("last buy (%s, %s) %d" % (user_id, item_category, days_from_last_opt_cat_dict[(user_id, item_category)]))

    return days_from_last_opt_cat_list


# 用户最后一次操作 item 至 checking_date 的天数，
# 天数需要取倒数
def get_last_opt_item_date(user_records, checking_date, behavior_type, user_id, item_id):
    last_opt_date = datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date()
    if (user_id not in user_records or item_id not in user_records[user_id]):
        return last_opt_date

    for item_opt_records in user_records[user_id][item_id]:
        for each_record in item_opt_records:
            for index in range(len(each_record)-1, -1, -1):
                if (each_record[index][0] != behavior_type):
                    continue

                #each_record 已经按照时间排好序
                if (each_record[index][1].date() >= last_opt_date and \
                    each_record[index][1].date() <= checking_date):
                    last_opt_date = each_record[index][1].date()
                    break 

    return last_opt_date

#最后一次操作 item 至 checking_date 的天数的倒数
def feature_last_opt_item(checking_date, behavior_type, user_item_pairs):
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        last_opt_date = get_last_opt_item_date(g_user_buy_transection, checking_date, \
                                               behavior_type, user_id, item_id)
        if (behavior_type != BEHAVIOR_TYPE_BUY):
            last_opt_date2 = get_last_opt_item_date(g_user_behavior_patten, checking_date, \
                                                    behavior_type, user_id, item_id)
            last_opt_date = max(last_opt_date, last_opt_date2)

        days_from_last_opt = checking_date - last_opt_date
        if (days_from_last_opt.days == 0):
            #同一天操作取 1
            days_from_last_opt_cat_dict[(user_id, item_category)] = 1
        else:
            #天数取倒数， 时间越远该数字越小
            days_from_last_opt_cat_dict[(user_id, item_category)] =  1/ days_from_last_opt.days

        days_from_last_opt_cat_list[index] = days_from_last_opt_cat_dict[(user_id, item_category)]
        logging.info("last behavior %d (%s, %s) %d" % (behavior_type, user_id, item_id, \
                     days_from_last_opt_cat_dict[(user_id, item_category)]))

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

def feature_behavior_cnt_before_1st_buy(behavior_type, user_item_pairs):
    behavior_cnt_before_1st_buy_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in g_user_buy_transection or 
            item_id not in g_user_buy_transection[user_id]):
            continue

        first_buy_date = g_user_buy_transection[user_id][item_id][-1][1].date()
        for each_record in g_user_buy_transection[user_id][item_id]:
            if (first_buy_date < each_record[-1][1].date()):
                first_buy_date = each_record[-1][1].date()

        behavior_cnt = get_behavior_cnt_before_date(g_user_buy_transection, behavior_type, \
                                                    first_buy_date, user_id, item_id)

        behavior_cnt += get_behavior_cnt_before_date(g_user_behavior_patten, behavior_type, \
                                                     first_buy_date, user_id, item_id)
        user_item_pairs[index] = behavior_cnt
        logging.info("(%s, %s) 1st buy %s, behavior %d cnt  %d" % (user_id, item_id, \
                     first_buy_date, behavior_type, behavior_cnt)

    return behavior_cnt_before_1st_buy_list

