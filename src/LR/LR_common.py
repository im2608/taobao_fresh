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
        # if (user_id != '110883802'):
        #     continue

        if (user_index < start_from):
            user_index += 1
            continue

        if (user_index > total_user + start_from):
            break

        #读取购物记录
        g_user_buy_transection[user_id] = dict()
        logging.info(user_id)

        user_whole_info = redis_cli.hgetall(user_id)

        item_id_list = user_whole_info[bytes("item_id".encode())].decode()
        if (len(item_id_list) > 0):
            item_id_list = item_id_list.split(",")
            for item_id in item_id_list:
                try:
                    item_buy_record = user_whole_info[bytes(item_id.encode())].decode()
                except KeyError as ex:
                    print("got KeyError, %s, %s" % (user_id, item_id))
                    raise ex

                g_user_buy_transection[user_id][item_id] = getRecordsFromRecordString(item_buy_record)
                #logging.info("%s %s buy record %s " % (user_id, item_id, g_user_buy_transection[user_id][item_id]))
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

    if (run_for_users_cnt == 50):
        logging.info(g_user_buy_transection.keys())

def getBehaviorCnt(user_records, item_id, checking_date, behavior_cnt):
    for user_id, item_opt_records in user_records.items():
        if (item_id not in item_opt_records):
            continue
        for each_record in item_opt_records[item_id]:
            for behavior_consecutive in each_record:
                if (behavior_consecutive[1].date() < checking_date):
                    behavior_cnt[behavior_consecutive[0] - 1] += behavior_consecutive[2]

#热度： 截止到checking_date（不包括）： （点击数*0.01+购买数*0.94+购物车数*0.47+收藏数*0.33）
def calculateItemPopularity(checking_date, items_in_nagetive_sampls):
    logging.info("calculateItemPopularity checking_date %s" % checking_date)
    item_popularity_dict = dict()
    total_items = len(items_in_nagetive_sampls)
    index = 0
    for item_id in items_in_nagetive_sampls:
        behavior_cnt = [0, 0, 0, 0]
        getBehaviorCnt(g_user_buy_transection, item_id, checking_date, behavior_cnt)
        getBehaviorCnt(g_user_behavior_patten, item_id, checking_date, behavior_cnt)
        popularity = behavior_cnt[0]*0.01 + behavior_cnt[1]*0.33 + behavior_cnt[2]*0.47 + behavior_cnt[3]*0.94
        item_popularity_dict[item_id] = popularity
        logging.info("as of %s, %s popularity %s ==> %.1f" % (checking_date, item_id, behavior_cnt, popularity))
        index += 1
        print("        %d / %d popularity calculated\r" % (index, total_items), end="")

    logging.info("leaving calculateItemPopularity")

    return item_popularity_dict