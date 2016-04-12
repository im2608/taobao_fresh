from common import *
import numpy as np

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
        if (user_id != '100673077'):
            continue

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

    behaviors = []
    
    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
                if (behavior_consecutive[1].date() == checking_date and \
                    behavior_consecutive[0] == behavior_type):
                    return 1
    return 0

#根据购物记录，检查user 是否在 checking_date 这一天对 item 有过 behavior type
def get_feature_buy_at_date(behavior_type, checking_date, user_item_pairs):
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

# 得到用户的 购买过商品数量/浏览过的数量
def get_feature_buy_view_ratio(user_item_pairs):
    buy_view_ratio = dict()
    buy_view_ratio_list = []

    buy_view_ratio_list = np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        if (user_id not in user_buy_records or 
            item_id not in user_buy_records[user_id]):
            continue

        if (user_id in buy_view_ratio):
            buy_view_ratio_list[index][1] = buy_view_ratio[user_id]
            continue

        buy_count = 0
        for item_id, item_buy_records in g_user_buy_transection[user_id].items():
            buy_count += len(item_buy_records)

        # 没有pattern， 所有的view 都转化成了buy
        if (user_id not in g_user_behavior_patten):
            buy_view_ratio_list[index][1] = 1

        view_count = 0
        for item_id, item_patterns in g_user_behavior_patten[user_id].items():
            view_count += len(item_patterns)

        buy_view_ratio[user_id] = buy_count / (buy_count + view_count)
        buy_view_ratio_list[index][1] = buy_view_ratio[user_id]

    return does_operated

# 商品热度 购买该商品的用户/总用户数
def get_feature_item_popularity(user_item_pairs):
    item_popularity_dict = dict()
    item_popularity_list =  np.zeros((len(user_item_pairs), 1))
    for index in range(len(user_item_pairs)):
        
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        if (item_id in item_popularity_dict):
            item_popularity_list[index][0] = item_popularity_dict[item_id]
            continue

        buy_item_user_cnt = 0
        for user_id, buy_records in g_user_buy_transection.items():
            if (item_id in buy_records):
                buy_item_user_cnt += 1

        item_popularity_dict[item_id] = round(buy_item_user_cnt / 20000, 4)
        item_popularity_list[index] = item_popularity_dict[item_id]

    return item_popularity_list

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 正样本为用户购买过的商品，负样本为用户操作过但是没有购买的商品
def taking_samples(positive_samples_cnt_per_user, nag_per_pos, item_popularity):
    samples = []
    Ymat = []
    index = 0

    logging.info("taking samples, positive could be %d, nagetive could be %d" % \
                (positive_samples_cnt_per_user, positive_samples_cnt_per_user * nag_per_pos))

    #在购物记录中采集正样本
    for user_id in g_user_buy_transection:        
        item_list_user_bought = getSamplesListByUser(g_user_buy_transection, user_id, positive_samples_cnt_per_user)
        actual_pos = len(item_list_user_bought)
        for i in range(actual_pos):
            samples.append((user_id, item_list_user_bought[i]))
            Ymat.append(1)

        if (user_id not in g_user_behavior_patten):
            logging.info("%s does not have viewing record. Actual positive %d" % (user_id, actual_pos))
            continue

        #在 patterns 中采集负样本
        nagetive_cnt = actual_pos * nag_per_pos
        item_list_user_opted = getSamplesListByUser(g_user_behavior_patten, user_id, nagetive_cnt)
        actual_nag = len(item_list_user_opted)
        for i in range(actual_nag):
            samples.append((user_id, item_list_user_opted[i]))
            Ymat.append(0)

        logging.info("%s acutal positive %d, acutal nagetive %d" % (user_id, actual_pos, actual_nag))

    return samples, Ymat

# TODO: 这里需要增强，按照item的热度来采样
def getSamplesListByUser(user_behavior_dict, user_id, sample_cnt):
    item_list_user_opted = list(user_behavior_dict[user_id].keys())
    items_not_in_test_set = []
    for item_id in item_list_user_opted:
        if (item_id not in global_train_item):
            items_not_in_test_set.append(item_id)
    
    #删除没有出现在 test set 中的 item
    for item_id in items_not_in_test_set:
        item_list_user_opted.remove(item_id)

    return item_list_user_opted[0 : sample_cnt]
    
def logisticRegression():
    positive_samples_cnt_per_user = 2
    nag_per_pos = 5

    #item 的热度
    item_popularity = get_feature_item_popularity(samples)
    Xmat[:, 0] = item_popularity

    samples, Ymat = taking_samples(positive_samples_cnt_per_user, nag_per_pos, item_popularity)
    logging.info("samples %s" % samples)
    logging.info("Ymat %s" % Ymat)

    feature_cnt = 5
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))


    checking_date = datetime.datetime.strptime("2014-12-17", "%Y-%m-%d").date()
    #用户在 checking date 是否有过 favorite
    Xmat[:, 1] = get_feature_buy_at_date(BEHAVIOR_TYPE_FAV, checking_date, samples)
    #用户在 checking date 是否有过 cart
    Xmat[:, 2] = get_feature_buy_at_date(BEHAVIOR_TYPE_CART, checking_date, samples)

    # 用户 购买过商品数量/浏览过的数量
    Xmat[:, 3] = get_feature_buy_view_ratio(samples):

    print(Xmat)
    return 0