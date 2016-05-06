from common import *
import time

#用户的购买记录
g_user_buy_transection = dict()  # 以 user id 为 key
g_user_buy_transection_item = dict()  # 以 item id 为 key


#在用户的操作记录中，从最后一条购买记录到train时间结束之间的操作行为记录，
#以它们作为patten
g_user_behavior_patten = dict() # 以 user id 为 key

g_user_behavior_patten_item = dict() # 以 item id 为 key

#总共的购买记录数
g_buy_record_cnt = 0
g_buy_record_cnt_verify = 0.0

g_pattern_cnt = 0.0

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
                if (item_id not in g_user_buy_transection_item):
                    g_user_buy_transection_item[item_id] = dict()

                item_buy_record = user_whole_info[bytes(item_id.encode())].decode()
                g_user_buy_transection[user_id][item_id] = getRecordsFromRecordString(item_buy_record)
                g_user_buy_transection_item[item_id][user_id] = g_user_buy_transection[user_id][item_id]                
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

            if (item_id not in g_user_behavior_patten_item):
                g_user_behavior_patten_item[item_id] = dict()
            g_user_behavior_patten_item[item_id][user_id] = g_user_behavior_patten[user_id][item_id]

            #logging.info("%s %s pattern is %s" % (user_id, item_id, g_user_behavior_patten[user_id][item_id]))

        user_index += 1
        print("%d / %d users read\r" % (user_index - start_from, total_user), end="")

    print("%s total buy count %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_pattern_cnt))
    logging.info("%s total buy count %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_pattern_cnt))



def calculate_item_popularity_by_records(user_records, item_popularity_dict, item_category_opt_cnt_dict):
    for user_id, item_opt_records in user_records.items():
        for item_id, records in item_opt_records.items():

            item_category = global_train_item_category[item_id]

            if (item_category not in item_category_opt_cnt_dict):
                item_category_opt_cnt_dict[item_category] = dict()

            if (item_id not in item_popularity_dict):
                item_popularity_dict[item_id] = dict()

            for each_record in records:
                for each_behavior in each_record:
                    behavior_type = each_behavior[0]
                    if (behavior_type not in item_category_opt_cnt_dict[item_category]):
                        item_category_opt_cnt_dict[item_category][behavior_type] = 0
                    item_category_opt_cnt_dict[item_category][behavior_type] += each_behavior[2]

                    if (behavior_type not in item_popularity_dict[item_id]):
                        item_popularity_dict[item_id][behavior_type] = 0
                    item_popularity_dict[item_id][behavior_type] += each_behavior[2]
    return 0

# 每个 item 在各个 behavior type 上的热度
# item 在各个 behavior type 上的次数/category 在各个 behavior type 上的次数
def calculate_item_popularity():
    #每个 item 在各个 behavior type 上的热度
    item_popularity_dict = dict()

    #在 category 上进行过各个 behavior type 操作的次数
    item_category_opt_cnt_dict = dict()

    calculate_item_popularity_by_records(g_user_buy_transection, item_popularity_dict, item_category_opt_cnt_dict)
    calculate_item_popularity_by_records(g_user_behavior_patten, item_popularity_dict, item_category_opt_cnt_dict)

    for item_id in item_popularity_dict:
        item_category = global_train_item_category[item_id]
        for behavior_type in [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART, BEHAVIOR_TYPE_BUY]:
            if (behavior_type in item_popularity_dict[item_id]):
                item_popularity_dict[item_id][behavior_type] = \
                    round(item_popularity_dict[item_id][behavior_type]/item_category_opt_cnt_dict[item_category][behavior_type], 6)
            else:
                item_popularity_dict[item_id][behavior_type] = 0

    if (len(item_popularity_dict) <= 500 ):
        logging.info("item popularity %s" % item_popularity_dict)
    return item_popularity_dict


def getBehaviorCnt(user_records, checking_date, item_behavior_cnt_dict):
    for user_id, item_opt_records in user_records.items():
        for item_id, records in item_opt_records.items():
            if (item_id not in item_behavior_cnt_dict):
                item_behavior_cnt_dict[item_id] = [0, 0, 0, 0]

            for each_record in records:
                for behavior_consecutive in each_record:
                    if (behavior_consecutive[1].date() < checking_date):
                        item_behavior_cnt_dict[item_id][behavior_consecutive[0] - 1] += behavior_consecutive[2]

#热度： 截止到checking_date（不包括）： （点击数*0.01+购买数*0.94+购物车数*0.47+收藏数*0.33）
def calculateItemPopularity(checking_date):
    logging.info("calculateItemPopularity checking_date %s" % checking_date)
    item_popularity_dict = dict()
    item_behavior_cnt_dict = dict()

    getBehaviorCnt(g_user_buy_transection, checking_date, item_behavior_cnt_dict)
    getBehaviorCnt(g_user_behavior_patten, checking_date, item_behavior_cnt_dict)
    index = 0
    total_items = len(item_behavior_cnt_dict)
    for item_id, behavior_cnt in item_behavior_cnt_dict.items():
        popularity = behavior_cnt[0]*0.01 + behavior_cnt[1]*0.33 + behavior_cnt[2]*0.47 + behavior_cnt[3]*0.94
        item_popularity_dict[item_id] = popularity

        logging.info("as of %s, %s popularity %s ==> %.1f" % (checking_date, item_id, behavior_cnt, popularity))

        index += 1
        if (index % 1000 == 0):
            print("                %d / %d popularity calculated\r" % (index, total_items), end="")

    logging.info("leaving calculateItemPopularity")

    return item_popularity_dict    



def timeElasped(start_time):
    return time.clock() - start_time    