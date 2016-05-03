from common import *
import numpy as np
from LR_common import *



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


#截止到checking date(不包括)， 用户在category 上的购买浏览转化率 在category上购买过的数量/浏览过的category数量
def feature_buy_view_ratio(checking_date, user_item_pairs):
    buy_view_ratio_dict = dict()

    buy_view_ratio_list = np.zeros((len(user_item_pairs), 1))

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
                if (each_record[-1][1].date() < checking_date):
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
        logging.info("as of %s, %s bought category %s %d, viewed %d, ratio %.4f" % \
                     (checking_date, user_id, item_category, buy_count, len(viewed_categories), buy_view_ratio_list[index, 0]))

    return buy_view_ratio_list



# 用户行为统计
# 用户在checking_date之前 [begin_date, end_date)(不包括end_date) 天购买（浏览， 收藏， 购物车）所有商品的总次数
def get_behavior_cnt_of_days(user_records, begin_date, end_date, behavior_type, user_id):
    if (user_id not in user_records):
        return 0
    behavior_cnt = 0
    for item_id, item_opt_records in user_records[user_id].items():
        for each_record in item_opt_records:
            for behavior_consecutive in each_record:
                if (behavior_consecutive[0] == behavior_type and \
                    behavior_consecutive[1].date() >= begin_date and 
                    behavior_consecutive[1].date() < end_date):
                    behavior_cnt += behavior_consecutive[2]
    return behavior_cnt

# user id 在 checking date(不包括) 之前 pre_days 天对 item id 进行的behavior type 操作的次数
def userBehaviorCntOnItemBeforeCheckingDate(user_records, user_id, item_id, behavior_type, checking_date, pre_days):
    begin_date = checking_date - datetime.timedelta(pre_days)
    behavior_cnt = 0

    if (user_id not in user_records or item_id not in user_records[user_id]):
        return 0

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() >= begin_date and
                behavior_consecutive[1].date() < checking_date and 
                behavior_type == behavior_consecutive[0]):
                behavior_cnt += behavior_consecutive[2]

    return behavior_cnt

# 用户checking_date（不包括）之前 pre_days 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
def feature_user_item_behavior_ratio(checking_date, behavior_type, pre_days, user_item_pairs):
    user_item_pop = dict()
    user_item_pop_list = np.zeros((len(user_item_pairs), 1))
    user_behavior_cnt_dict = dict()
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        total_behavior_cnt = 0
        if (user_id in user_behavior_cnt_dict):
            total_behavior_cnt = user_behavior_cnt_dict[user_id]
        else:
            begin_date = checking_date - datetime.timedelta(pre_days)
            total_behavior_cnt = get_behavior_cnt_of_days(g_user_buy_transection, begin_date, checking_date, behavior_type, user_id)
            total_behavior_cnt += get_behavior_cnt_of_days(g_user_behavior_patten, begin_date, checking_date, behavior_type, user_id)

        user_behavior_cnt_dict[user_id] = total_behavior_cnt
        if (total_behavior_cnt == 0):
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

        user_item_pop_list[index] = behavior_cnt / total_behavior_cnt

        logging.info("(%s, %s), checking date %s, behavior %d, pre days %d, behavior cnt %d / %d" %
                     (user_id, item_id, checking_date, behavior_type, pre_days, behavior_cnt, total_behavior_cnt))

    return user_item_pop_list

# user 对 item 购买间隔的平均天数
def mean_days_between_buy(checking_date, user_item_pairs):
    samle_cnt = len(user_item_pairs)
    for user_item in user_item_pairs:
        user_id = user_item[0]
        item_id = user_item[1]
    return 