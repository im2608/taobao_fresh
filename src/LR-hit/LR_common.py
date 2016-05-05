from common import *

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

#每个用户不同操作类型的次数
g_user_behavior_count = dict()

# 每条购物记录在 redis 中都表现为字符串 
#"[ [(1, 2014-01-01 23, 35), (2, 2014-01-02 22, 1)], [(1, 2014-01-02 23, 35), (2, 2014-01-03 14, 1)] ]"
# run_for_users_cnt 跑多少个用户， 0 表示全部
def loadRecordsFromRedis(start_from, run_for_users_cnt, need_verify):

    global g_buy_record_cnt
    global g_buy_record_cnt_verify
    global g_pattern_cnt

    # 得到所有的用户
    all_users = redis_cli.get("all_users").decode()
    all_users = all_users.split(",")

    total_user = len(all_users)
    if (run_for_users_cnt > 0):
        total_user = run_for_users_cnt
    else:
        start_from = 0

    print("%s loadRecordsFromRedis, here total %d users, start from %d" % (getCurrentTime(), total_user, start_from))

    #根据用户得到用户操作过的item id
    user_index = 0
    skiped_user = 0
    for user_id in all_users:
        # if (user_id != '49818651'):
        #     continue

        if (user_index < start_from):
            user_index += 1
            continue

        if (user_index > total_user + start_from):
            break

        logging.info(user_id)

        #读取购物记录
        g_user_buy_transection[user_id] = dict()

        user_whole_info = redis_cli.hgetall(user_id)

        item_id_list = user_whole_info[bytes("item_id".encode())].decode()
        if (len(item_id_list) > 0):
            item_id_list = item_id_list.split(",")
            for item_id in item_id_list:
                item_buy_record = user_whole_info[bytes(item_id.encode())].decode()
                g_user_buy_transection[user_id][item_id] = getRecordsFromRecordString(item_buy_record)
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
        print("%d / %d users read\r" % (user_index - start_from, total_user), end="")

    print("%s total buy count %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_pattern_cnt))
    logging.info("%s total buy count %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_pattern_cnt))

def userBehaviorStatisticOnRecords(user_records):
    for user_id, opt_records in user_records.items():
        if (user_id not in g_user_behavior_count):
            g_user_behavior_count[user_id] = dict()
            g_user_behavior_count[user_id][BEHAVIOR_TYPE_BUY] = 0
            g_user_behavior_count[user_id][BEHAVIOR_TYPE_VIEW] = 0
            g_user_behavior_count[user_id][BEHAVIOR_TYPE_FAV] = 0
            g_user_behavior_count[user_id][BEHAVIOR_TYPE_CART] = 0

        for item_id, item_opt_records in opt_records.items():
            for each_record in item_opt_records:
                for behavior_cosecutive in each_record:
                    g_user_behavior_count[user_id][behavior_cosecutive[0]] += behavior_cosecutive[2]
    return 0



# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品总用户数
# 若没有用户购买过某产品，则 item_popularity 中就没有该 item 
# 只计算出现在 test set 中的 item
def calculate_item_popularity_by_records(user_records, item_popularity_dict, item_category_opt_cnt_dict):
    for user_id, item_opt_records in user_records.items():
        for item_id, records in item_opt_records.items():
            if (item_id not in global_test_item_category):
                continue

            item_category = global_test_item_category[item_id]

            if (item_category not in item_category_opt_cnt_dict):
                item_category_opt_cnt_dict[item_category] = dict()

            if (item_id not in item_popularity_dict):
                item_popularity_dict[item_id] = dict()

            behavior_type_counted = set()
            for each_record in records:
                if (len(behavior_type_counted) == 4):
                    break

                for each_behavior in each_record:
                    if (len(behavior_type_counted) == 4):
                        break

                    behavior_type = each_behavior[0]
                    if (behavior_type in behavior_type_counted):
                        continue

                    behavior_type_counted.add(behavior_type)
                    if (behavior_type not in item_category_opt_cnt_dict[item_category]):
                        item_category_opt_cnt_dict[item_category][behavior_type] = 0
                    item_category_opt_cnt_dict[item_category][behavior_type] += each_behavior[2]

                    if (behavior_type not in item_popularity_dict[item_id]):
                        item_popularity_dict[item_id][behavior_type] = 0
                    item_popularity_dict[item_id][behavior_type] += each_behavior[2]
    return 0

def calculate_item_popularity():
    #每个 item 在各个 behavior type 上的热度
    item_popularity_dict = dict()

    #在 category 上进行过各个 behavior type 操作的用户数量
    item_category_opt_cnt_dict = dict()

    calculate_item_popularity_by_records(g_user_buy_transection, item_popularity_dict, item_category_opt_cnt_dict)
    calculate_item_popularity_by_records(g_user_behavior_patten, item_popularity_dict, item_category_opt_cnt_dict)

    for item_id in item_popularity_dict:
        item_category = global_train_item_category[item_id]
        for behavior_idx in range(1, 5):
            if (behavior_idx in item_popularity_dict[item_id] and\
                behavior_idx in item_category_opt_cnt_dict[item_category]):
                item_popularity_dict[item_id][behavior_idx] = \
                    round(item_popularity_dict[item_id][behavior_idx]/item_category_opt_cnt_dict[item_category][behavior_idx], 5)

    if (len(item_popularity_dict) <= 500 ):
        logging.info("item popularity %s" % item_popularity_dict)
    return item_popularity_dict