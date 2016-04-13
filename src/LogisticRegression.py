from common import *
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

g_user_buy_transection = dict()

#用于验证的用户购买行为
g_user_buy_transection_verify = dict()


#最终的预测结果
g_final_forecast = dict()
g_buy_record_cnt_forecast = 0.0

#在用户的操作记录中，从最后一条购买记录到train时间结束之间的操作行为记录，
#以它们作为patten
g_user_behavior_patten = dict()

#总共的购买记录数
g_buy_record_cnt = 0
g_buy_record_cnt_verify = 0.0

g_pattern_cnt = 0.0


# 每条购物记录在 redis 中都表现为字符串 
#"[ [(1, 2014-01-01 23, 35), (2, 2014-01-02 22, 1)], [(1, 2014-01-02 23, 35), (2, 2014-01-03 14, 1)] ]"
def loadRecordsFromRedis(need_verify):
    global g_buy_record_cnt
    global g_buy_record_cnt_verify
    global g_pattern_cnt

    # 得到所有的用户
    all_users = redis_cli.get("all_users").decode()
    all_users = all_users.split(",")

    total_user = len(all_users)
    print("%s loadRecordsFromRedis, here total %d users" % (getCurrentTime(), total_user))

    #根据用户得到用户操作过的item id
    user_index = 0
    skiped_user = 0
    for user_id in all_users:
        # if (user_id != '100673077'):
        #     continue

        #读取购物记录
        g_user_buy_transection[user_id] = dict()

        user_whole_info = redis_cli.hgetall(user_id)

        item_id_list = user_whole_info[bytes("item_id".encode())].decode()
        if (len(item_id_list) > 0):
            item_id_list = item_id_list.split(",")
            for item_id in item_id_list:
                item_buy_record = user_whole_info[bytes(item_id.encode())].decode()
                g_user_buy_transection[user_id][item_id] = getRecordsFromRecordString(item_buy_record)
                logging.info("%s %s buy record %s " % (user_id, item_id, g_user_buy_transection[user_id][item_id]))
                g_buy_record_cnt += len(g_user_buy_transection[user_id][item_id])
        else:
            user_index += 1
            skiped_user += 1

        #得到用户的patterns
        tmp = bytes("item_id_pattern".encode())
        if tmp not in user_whole_info:
            continue

        item_pattern_list = user_whole_info[tmp].decode()
        if (len(item_pattern_list) == 0):
            logging.info("user %s has no patterns!")
            continue

        item_pattern_list = item_pattern_list.split(",")
        g_pattern_cnt += len(item_pattern_list)
        if (user_id not in g_user_behavior_patten):
            g_user_behavior_patten[user_id] = dict()

        for item_id in item_pattern_list:
            tmp = item_id + "_pattern"
            item_pattern = user_whole_info[bytes(tmp.encode())].decode()
            g_user_behavior_patten[user_id][item_id] = getRecordsFromRecordString(item_pattern)

            #logging.info("%s %s pattern is %s" % (user_id, item_id, g_user_behavior_patten[user_id][item_id]))

        user_index += 1
        print("%d / %d users read\r" % (user_index, total_user), end="")

    print("%s total buy count %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_pattern_cnt))
    logging.info("%s total buy count %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_pattern_cnt))


# 得到某个用户某天对item 的所有操作
def get_behavior_by_date(user_records, behavior_type, checking_date,user_item_pair):
    user_id = user_item_pair[0]
    item_id = user_item_pair[1]
    if (user_id not in user_records or 
        item_id not in user_records[user_id]):
        return None

    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
                if (behavior_consecutive[1].date() == checking_date and \
                    behavior_consecutive[0] == behavior_type):
                    return 1
    return 0

#根据购物记录，检查user 是否在 checking_date 这一天对 item 有过 behavior type
def get_feature_behavior_on_date(behavior_type, checking_date, user_item_pairs):
    does_operated = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in g_user_buy_transection or 
            item_id not in g_user_buy_transection[user_id]):
            continue

        #检查在购物记录中, 在 checking date 是否有过 behavior type 操作
        does_operated[index][0] = get_behavior_by_date(g_user_buy_transection, behavior_type, checking_date, user_item_pairs[index])
        if (does_operated[index][0] == 1):
            continue

        #检查在 patterns 中, 在 checking date 是否有过 behavior type 操作
        does_operated[index][0] = get_behavior_by_date(g_user_behavior_patten, behavior_type, checking_date, user_item_pairs[index])

    return does_operated

# 得到用户的购买浏览转化率 购买过商品数量/浏览过的数量
def get_feature_buy_view_ratio(user_item_pairs):
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

# 计算所有商品的热度 购买该商品的用户/总用户数
def calculate_item_popularity():
    item_popularity_dict = dict()

    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id in item_buy_records:
            if (item_id not in item_popularity_dict):
                item_popularity_dict[item_id] = 0
            item_popularity_dict[item_id] += 1

    for item_id in item_popularity_dict:
        item_popularity_dict[item_id] = round(item_popularity_dict[item_id]/2000, 4)

    logging.info("item popularity %s" % item_popularity_dict)
    return item_popularity_dict

# 商品热度 购买该商品的用户/总用户数
def get_feature_item_popularity(item_popularity_dict, user_item_pairs):
    item_popularity_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (item_id in item_popularity_dict):
            item_popularity_list[index][0] = item_popularity_dict[item_id]
        else:
            item_popularity_list[index][0] = 0

    return item_popularity_list

#最后一次购买同类型的商品至 checking_date 的天数
def get_feature_last_buy(checking_date, user_item_pairs):
    days_from_last_buy_cat_dict = dict()
    days_from_last_buy_cat_list = np.zeros((len(user_item_pairs), 1))

    for index in range(len(user_item_pairs)):

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        sample_item_category = getCatalogByItemId(item_id)
        if sample_item_category == None:
            continue

        if ((user_id, sample_item_category) in days_from_last_buy_cat_dict):
            days_from_last_buy_cat_list[index] = days_from_last_buy_cat_dict[(user_id, sample_item_category)]
            continue

        days_from_last_buy_cat_dict[(user_id, sample_item_category)] = 0

        for item_id_can, buy_records in g_user_buy_transection[user_id].items():
            # 不属于同一个 category， skip
            if (getCatalogByItemId(item_id_can) != sample_item_category):
                continue

            for each_record in buy_records:
                for index in range(len(each_record) -1, -1, -1):

                    behavior_type = each_record[index][0]
                    behavior_time = each_record[index][1]
                    if (behavior_type != BEHAVIOR_TYPE_BUY or \
                        behavior_time.date() > checking_date):
                        continue

                    days_from_last_buy = checking_date - behavior_time.date()
                    if (days_from_last_buy_cat_dict[(user_id, sample_item_category)] < days_from_last_buy.days):
                        days_from_last_buy_cat_dict[(user_id, sample_item_category)] =  1/ days_from_last_buy.days
                    else:
                        days_from_last_buy_cat_dict[(user_id, sample_item_category)] = 1

        days_from_last_buy_cat_list[index] = days_from_last_buy_cat_dict[(user_id, sample_item_category)]
        logging.info("last buy (%s, %s) %d" % (user_id, sample_item_category, days_from_last_buy_cat_dict[(user_id, sample_item_category)]))

    return days_from_last_buy_cat_list


# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品

def taking_samples(positive_samples_cnt_per_user, checking_date, nag_per_pos, item_popularity):
    samples = []
    Ymat = []
    index = 0

    logging.info("taking samples, positive could be %d, nagetive could be %d" % \
                (positive_samples_cnt_per_user, positive_samples_cnt_per_user * nag_per_pos))

    #在购物记录中根据 checking date 采集正样本
    for user_id in g_user_buy_transection:        
        item_list_user_bought = getPositiveSamplesListByUser(checking_date, user_id, positive_samples_cnt_per_user)

        actual_pos = len(item_list_user_bought)
        for i in range(actual_pos):
            samples.append((user_id, item_list_user_bought[i]))
            Ymat.append(1)


        #负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品
        nagetive_cnt = actual_pos * nag_per_pos
        item_list_user_opted = getNagetiveSampleListByUser(checking_date, user_id, nagetive_cnt, item_list_user_bought)
        actual_nag = len(item_list_user_opted)
        for i in range(actual_nag):
            samples.append((user_id, item_list_user_opted[i]))
            Ymat.append(0)

        logging.info("%s acutal positive %d, acutal nagetive %d" % (user_id, actual_pos, actual_nag))

    return samples, Ymat

# 对用户操作过的item 采样, sample_cnt 采集多少个样本
# TODO: 这里需要增强，按照item的热度来采样
def getPositiveSamplesListByUser(checking_date, user_id, sample_cnt):
    item_list_user_bought = set()
    #得到符合条件的item list
    for item_id, item_buy_records in g_user_buy_transection[user_id].items():
        for each_record in item_buy_records:
            buy_behavior = each_record[-1]
            if (buy_behavior[0] != BEHAVIOR_TYPE_BUY):
                logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                continue

            if (buy_behavior[1].date() != checking_date):
                continue

            item_list_user_bought.add(item_id)

    #只采样出现在 test set 中的 item
    items_in_test_set = filterItemByTestset(item_list_user_bought)
    item_list_user_bought = list(items_in_test_set)

    return item_list_user_bought[0 : sample_cnt]
    
def getNagetiveSampleListByUser(checking_date, user_id, sample_cnt, positive_samples):
    nagetive_sample_candidates = []
    #在购物记录中采样在checking date这一天没有购买的商品
    for item_id, item_buy_records in g_user_buy_transection[user_id].items():
        for each_record in item_buy_records:
            buy_behavior = each_record[-1]
            if (buy_behavior[0] != BEHAVIOR_TYPE_BUY):
                logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                continue

            if (buy_behavior[1].date() == checking_date):
                continue

            nagetive_sample_candidates.append(item_id)

    #在pattern 中采样用户浏览过的item
    nagetive_sample_candidates.extend(list(g_user_behavior_patten[user_id].keys()))

    candidates_in_test_set = set()

    # 如果用户购买了某产品， 然后又浏览的该商品，则该商品会出现在 pattern 中，
    # 为了防止正/负样本同时采样到该商品，这里做过滤
    for item_id in nagetive_sample_candidates:
        if (item_id in positive_samples):
            logging.warn("%s: pos/nag take %d as sample at same time, skip!")
        else:
            candidates_in_test_set.add(item_id)

    nagetive_samples = filterItemByTestset(candidates_in_test_set)

    return list(nagetive_samples)[: sample_cnt]

def filterItemByTestset(item_id_set):
    items_in_test_set = set()
    for item_id in item_id_set:
        item_category = getCatalogByItemId(item_id)
        if (item_category is not None):
            items_in_test_set.add(item_id)

    return items_in_test_set

def logisticRegression():
    positive_samples_cnt_per_user = 2
    nag_per_pos = 5
    checking_date = datetime.datetime.strptime("2014-12-17", "%Y-%m-%d").date()
    print("%s checking date %s", (getCurrentTime(), checking_date))

    #item 的热度
    print("%s calculating popularity..." % getCurrentTime())
    item_popularity_dict = calculate_item_popularity()

    print("%s taking samples..." % getCurrentTime())
    samples, Ymat = taking_samples(positive_samples_cnt_per_user, checking_date, nag_per_pos, item_popularity_dict)
    print("samples count %d, Ymat count %d" % (len(samples), len(Ymat)))
    logging.info("samples %s" % samples)
    logging.info("Ymat %s" % Ymat)

    feature_cnt = 5
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))

    # 商品热度 购买该商品的用户/总用户数
    print("%s getting item popularity by samples..." % getCurrentTime())
    Xmat[:, 0] = get_feature_item_popularity(item_popularity_dict, samples)

    #用户在 checking date 是否有过 favorite
    print("%scalculating FAV ..." % getCurrentTime())
    Xmat[:, 1] = get_feature_behavior_on_date(BEHAVIOR_TYPE_FAV, checking_date, samples)

    #用户在 checking date 是否有过 cart
    print("%s calculating CART ..." % getCurrentTime())
    Xmat[:, 2] = get_feature_behavior_on_date(BEHAVIOR_TYPE_CART, checking_date, samples)

    # 用户 购买过商品数量/浏览过的数量
    print("%scalculating b/v ratio..." % getCurrentTime())
    Xmat[:, 3] = get_feature_buy_view_ratio(samples)

    #最后一次购买同类型的商品至 checking_date 的天数
    print("%s calculating last buy..." % getCurrentTime())
    Xmat[:, 4] = get_feature_last_buy(checking_date, samples)

    min_max_scaler = preprocessing.MinMaxScaler()

    Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    logging.info(Xmat)
    logging.info(Xmat_scaler)

    model = LogisticRegression()
    model.fit(Xmat_scaler, Ymat)

    expected = Ymat
    predicted = model.predict(Xmat)
    print(expected)
    print(predicted)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


    #outputXY(Xmat, Ymat, samples)
    return 0

def outputXY(Xmat, Ymat, samples):
    xm, xn = np.shape(Xmat)
    ym = len(Ymat)
    sample_cnt = len(samples)
    if (xm != ym):
        logging.error("ERROR: lines %d of Xmat != lines %d of Ymat" % (xm, ym))
        return
    mat_row_string = []
    for row_idx in range(xm):
        mat_row_string.append("[")
        for col_idx in range(xn):
            mat_row_string.append("%.2f " % Xmat[row_idx, col_idx])
        mat_row_string.append("] = %d %s\n" % (Ymat[row_idx], samples[row_idx]))

    logging.info(mat_row_string)



