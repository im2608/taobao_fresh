from common import *
import numpy as np



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

#截止到checking date(不包括)， 用户在category 上的购买浏览转化率 在category上购买过的数量/浏览过的category数量
def feature_buy_view_ratio(window_start_date, window_end_date, user_item_pairs):
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
                    each_record[-1][1].date >= window_start_date):
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
                     (window_end_date, user_id, item_category, buy_count, len(viewed_categories), buy_view_ratio_list[index, 0]))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")


    return buy_view_ratio_list


######################################################################################################
######################################################################################################
######################################################################################################

# 用户行为统计
# 用户在checking_date之前 [begin_date, end_date)(不包括end_date) 天购买（浏览， 收藏， 购物车）所有商品的总次数
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
def userBehaviorCntOnItemBeforeCheckingDate(user_records, user_id, item_id, checking_date, pre_days, behavior_cnt, behavior_type):
    if (user_id not in user_records or item_id not in user_records[user_id]):
        return

    begin_date = checking_date - datetime.timedelta(pre_days)

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() >= begin_date and
                behavior_consecutive[1].date() < checking_date and
                behavior_consecutive[0] == behavior_type):
                behavior_cnt[behavior_type - 1] += behavior_consecutive[2]
    return

# 用户checking_date（不包括）之前 pre_days 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
def feature_user_item_behavior_ratio(checking_date, pre_days, user_item_pairs):
    user_item_pop_list = np.zeros((len(user_item_pairs), 1))
    user_behavior_cnt_dict = dict()
    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        total_behavior_cnt = [0, 0, 0, 0]
        if (user_id in user_behavior_cnt_dict):
            total_behavior_cnt = user_behavior_cnt_dict[user_id]
        else:
            begin_date = checking_date - datetime.timedelta(pre_days)
            get_behavior_cnt_of_days(g_user_buy_transection, begin_date, checking_date, total_behavior_cnt, user_id)
            get_behavior_cnt_of_days(g_user_behavior_patten, begin_date, checking_date, total_behavior_cnt, user_id)

        user_behavior_cnt_dict[user_id] = total_behavior_cnt
        if (sum(total_behavior_cnt) == 0):
            continue

        behavior_cnt = [0, 0, 0, 0]

        #购物记录中也包含了其他类型的操作, 所以要先过一遍 buy records
        for behavior_type in [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART, BEHAVIOR_TYPE_BUY]:
            userBehaviorCntOnItemBeforeCheckingDate(g_user_buy_transection, \
                user_id, item_id, checking_date, pre_days, behavior_cnt, behavior_type)
            userBehaviorCntOnItemBeforeCheckingDate(g_user_behavior_patten, \
                user_id, item_id, checking_date, pre_days, behavior_cnt, behavior_type)

        for index in range(len(behavior_cnt)):
            if (total_behavior_cnt[index] != 0):
                user_item_pop_list[index] = behavior_cnt[index] / total_behavior_cnt[index]

        logging.info("(%s, %s), %s - %s , behavior cnt %s / %s" %
                     (user_id, item_id, begin_date, checking_date, behavior_cnt, total_behavior_cnt))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return user_item_pop_list

######################################################################################################
######################################################################################################
######################################################################################################


#[window_start_date, window_end_date) 时间内， 用户一共购买过多少同类型的商品

def feature_how_many_buy(window_start_date, window_end_date, user_item_pairs):
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
                    each_record[-1][1].date() < window_end_date)
                    buy_count += 1

        how_many_buy_dict[(user_id, item_category)] = buy_count
        how_many_buy[index] = how_many_buy_dict[(user_id, item_category)]
        logging.info("as of %s, %s bought category %s %d" % (checking_date, user_id, item_category, buy_count)) 
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return how_many_buy

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
def feature_last_opt_category(window_start_date, window_end_date, user_item_pairs):
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
        logging.info("%s last opted category %s, days %s to %s" % \
                     (user_id, item_category, days_from_last_opt_cat_dict[(user_id, item_category)], window_end_date))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return days_from_last_opt_cat_list


######################################################################################################
######################################################################################################
######################################################################################################

# 用户最后一次操作 item 至 checking_date(不包括）) 的天数，若没有同类型的操作则返回 0
def get_last_opt_item_date(user_records, checking_date, user_id, item_id):
    days = [0, 0, 0, 0]

    if (user_id not in user_records or \
        item_id not in user_records[user_id]):
        return days

    last_opt_date = [datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date() for x in range(4)]

    for each_record in user_records[user_id][item_id]:
        for index in range(len(each_record)-1, -1, -1):
            #each_record 已经按照时间排好序
            behavior_type = each_record[index][0]
            behavior_date = each_record[index][1].date()
            if (behavior_date >= last_opt_date[behavior_type - 1] and \
                behavior_date < checking_date):            
                last_opt_date[behavior_type - 1] = behavior_date
                days[behavior_type - 1] = (checking_date - behavior_date).days
                break 
    return days

# 用户最后一次操作 item 至 checking_date(不包括) 的天数，
def feature_last_opt_item(checking_date, user_item_pairs):
    days_from_last_opt_cat_list = np.zeros((len(user_item_pairs), 4))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        days = get_last_opt_item_date(g_user_buy_transection, checking_date, user_id, item_id)
        days2 = get_last_opt_item_date(g_user_behavior_patten, checking_date, user_id, item_id)
        for index in range(len(days)):
            if (days[index] == 0):
                if (days2 != 0):
                    days[index] = days2[index]
            elif (days2[index] != 0):
                days[index] = min(days[index], days2[index])

        days_from_last_opt_cat_list[index] =  days

        logging.info("user %s last opted item %s days %s to %s" % \
            (user_id, item_id, days_from_last_opt_cat_list[index], checking_date))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

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
def feature_behavior_cnt_before_1st_buy(checking_date, user_item_pairs):
    logging.info("feature_behavior_cnt_before_1st_buy(%s)" % checking_date)

    behavior_cnt_before_1st_buy_list = np.zeros((len(user_item_pairs), 3))

    total_cnt = len(user_item_pairs)
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (item_id not in g_user_buy_transection[user_id]):
            continue

        first_buy_date = None
        for each_record in g_user_buy_transection[user_id][item_id]:
            if ((first_buy_date == None or first_buy_date < each_record[-1][1].date()) and
                each_record[-1][1].date() < checking_date):
                first_buy_date = each_record[-1][1].date()

        if (first_buy_date is None):
            logging.info("%s has not bought %s before %s" % (user_id, item_id, checking_date))
            continue

        behavior_cnt = [0, 0, 0]

        get_behavior_cnt_before_date(g_user_buy_transection, first_buy_date, user_id, item_id, behavior_cnt)
        get_behavior_cnt_before_date(g_user_behavior_patten, first_buy_date, user_id, item_id, behavior_cnt)

        behavior_cnt_before_1st_buy_list[index] = behavior_cnt
        logging.info("%s (%s, %s) 1st buy %s, behavior cnt %s" % (getCurrentTime(), user_id, item_id, \
                     first_buy_date, behavior_cnt))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return behavior_cnt_before_1st_buy_list


######################################################################################################
######################################################################################################
######################################################################################################

# user 对 item 购买间隔的平均天数
def mean_days_between_buy_user_item(checking_date, user_item_pairs):
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
            if (each_record[-1][1].date() < checking_date):
                buy_date.append(each_record[-1][1].date())

        buy_date.sort()

        # 若只购买过一次，则将购买日期至checking date 作为平均天数
        if (len(buy_date) == 1):
            buy_date.append(checking_date)

        days = 0
        for date_index in range(1, len(buy_date)):
            days += (buy_date[date_index] - buy_date[date_index-1]).days

        buy_mean_days_list[index] = round(days / (len(buy_date) - 1))
        logging.info("user %s, item %s, mean buy days %d" % (user_id, item_id, buy_mean_days_list[index]))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return buy_mean_days_list

######################################################################################################
######################################################################################################
######################################################################################################


#截止到checking date（不包括）， user 对 category 购买间隔的平均天数以及方差
def feature_mean_days_between_buy_user_category(checking_date, user_item_pairs):
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
                if (each_record[-1][1].date() < checking_date):
                    buy_date.add(each_record[-1][1].date())

        if (len(buy_date) == 0):
            continue

        buy_date = list(buy_date)
        buy_date.sort()

        # 若只购买过一次，则将购买日期至checking date 作为平均天数
        if (len(buy_date) == 1):
            buy_date.append(checking_date)

        days_between_buy = []
        for date_index in range(1, len(buy_date)):
            days_between_buy.append((buy_date[date_index] - buy_date[date_index-1]).days)

        mean_vairance = [0, 0]
        mean_vairance[0] = np.round(np.mean(days_between_buy), 4)
        mean_vairance[1] = np.round(np.var(days_between_buy), 4)

        buy_mean_days_list[index] = mean_vairance
        user_category_mean_buy_dict[(user_id, item_category)] = mean_vairance

        logging.info("as of %s, user %s, category %s, mean vairance days %s" %\
                     (checking_date, user_id, item_category, buy_mean_days_list[index]))
        if (index % 1000 == 0):
            print("        %d / %d calculated\r" % (index, total_cnt), end="")

    return buy_mean_days_list    

######################################################################################################
######################################################################################################
######################################################################################################



