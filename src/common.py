import sys

runningPath = sys.path[0]
sys.path.append("%s\\LR-hit\\" % runningPath)
sys.path.append("%s\\RF\\" % runningPath)
sys.path.append("%s\\GBDT\\" % runningPath)
sys.path.append("%s\\features\\" % runningPath)
sys.path.append("%s\\samples\\" % runningPath)

import csv
import time
import datetime
import logging
import redis
import numpy as np

USER_ID = "user_id"
ITEM_ID = "item_id"
BEHAVEIOR_TYPE = "behavior_type"
USER_GEO = "user_geohash"
ITEM_CATE = "item_category"
TIME = "time"

algo = ""

BEHAVIOR_TYPE_VIEW = 1
BEHAVIOR_TYPE_FAV  = 2
BEHAVIOR_TYPE_CART = 3
BEHAVIOR_TYPE_BUY  = 4


tianchi_fresh_comp_train_user = "%s\\..\\input\\tianchi_fresh_comp_train_user.csv" % runningPath
tianchi_fresh_comp_train_item = "%s\\..\\input\\tianchi_fresh_comp_train_item.csv" % runningPath

# 以 user id 为 key
global_user_item_dict = dict()

# 以item category 为 key
global_item_user_dict = dict()

# 测试集数据，以 item id 为key
global_test_item_category = dict()
# 测试集数据，以 category 为key
global_test_category_item = dict()

#训练数据集， 以 item 为 key
global_train_item_category = dict()
#训练数据集， 以 category 为 key
global_train_category_item = dict()

global_totalBehaviorWeightHash = dict()
global_user_behavior_cnt = dict()

redis_cli = redis.Redis(host='10.57.14.7', port=6379, db=0)


# CRITICAL 50
# ERROR    40
# WARNING  30
# INFO     20
# DEBUG    10
# NOTSET   0
# logging.basicConfig(level=logging.INFO,\
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
#                     datefmt='%a, %d %b %Y %H:%M:%S',\
#                     filename='..\\log\\log.test.txt',\
#                     filemode='w')

def loadData(train_user_file_name = tianchi_fresh_comp_train_user):
    filehandle1 = open(train_user_file_name, encoding="utf-8", mode='r')
    user_behavior = csv.reader(filehandle1)

    index = 0
    logging.info("loadData(): loading file %s" % train_user_file_name)
    for aline in user_behavior:
        if (index == 0):
            index += 1
            continue

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        user_geohash  = aline[3]
        item_category = int(aline[4])
        behavior_time = datetime.datetime.strptime(aline[5], "%Y-%m-%d %H")

        #哪些用户对哪些物品有操作
        if (user_id not in global_user_item_dict):
            global_user_item_dict[user_id] = dict()

        if (item_category not in global_user_item_dict[user_id]):
            global_user_item_dict[user_id][item_category] = \
                {\
                    BEHAVIOR_TYPE_VIEW: {ITEM_ID:[], TIME:[]},\
                    BEHAVIOR_TYPE_FAV:  {ITEM_ID:[], TIME:[]},\
                    BEHAVIOR_TYPE_CART: {ITEM_ID:[], TIME:[]},\
                    BEHAVIOR_TYPE_BUY:  {ITEM_ID:[], TIME:[]},\
                    "w":0 # category weight for user
                }

        global_user_item_dict[user_id][item_category][behavior_type][ITEM_ID].append(item_id)
        global_user_item_dict[user_id][item_category][behavior_type][TIME].append(behavior_time)

        if (item_category not in global_item_user_dict):
            global_item_user_dict[item_category] = {ITEM_ID:set(), USER_ID:set()}

        #哪些物品分类被哪些用户操作, 该物品分类中包含了哪些item id
        global_item_user_dict[item_category][USER_ID].add(user_id)
        global_item_user_dict[item_category][ITEM_ID].add(item_id)

        bev_type = int(aline[2])
        if (bev_type not in global_user_behavior_cnt):
            global_user_behavior_cnt[bev_type] = 0

        global_user_behavior_cnt[bev_type] += 1
        index += 1

        if (index % 100000 == 0):
            print("lines read", index)

    print("total %d lines read" % index)

    logging.info("global_user_behavior_cnt is %s" % global_user_behavior_cnt)

    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_VIEW] = 1
    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_FAV] = global_user_behavior_cnt[BEHAVIOR_TYPE_VIEW] / global_user_behavior_cnt[BEHAVIOR_TYPE_FAV]
    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_CART] = global_user_behavior_cnt[BEHAVIOR_TYPE_VIEW] / global_user_behavior_cnt[BEHAVIOR_TYPE_CART]
    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_BUY] = global_user_behavior_cnt[BEHAVIOR_TYPE_VIEW] / global_user_behavior_cnt[BEHAVIOR_TYPE_BUY]

    logging.info("global_totalBehaviorWeightHash is %s" % global_totalBehaviorWeightHash)

    logging.info("loadData(): loading file %s" % train_item_file_name)

    filehandle1.close()
    return 0

def loadTestSet():
    print("%s loading %s" % (getCurrentTime(), tianchi_fresh_comp_train_item))

    filehandle2 = open(tianchi_fresh_comp_train_item, encoding="utf-8", mode='r')
    item_info  = csv.reader(filehandle2)

    index = 0
    for aline in item_info:
        if (index == 0):
            index = 1
            continue

        item_id       = aline[0]
        item_geohash  = aline[1]
        item_category = aline[2]

        global_test_item_category[item_id] = item_category

        if (item_category not in global_test_category_item):
            global_test_category_item[item_category] = set()
        global_test_category_item[item_category].add(item_id)

    filehandle2.close()

    return 0

def getCatalogByItemId(item_id):
    if (item_id in global_test_item_category):
        return global_test_item_category[item_id]

    return None

#计算用户所有操作过的item 对于用户的权值
def calItemCategoryWeight():
    logging.info("calItemCategoryWeight...")
    for user_id, user_opt in global_user_item_dict.items():

        user_behavior_weight = {BEHAVIOR_TYPE_VIEW:0, BEHAVIOR_TYPE_FAV:0, BEHAVIOR_TYPE_CART:0, BEHAVIOR_TYPE_BUY:0}

        #得到用户在所有操作过的 catgories 上各种操作类型总的数量
        for item_category, user_category_opt in user_opt.items():
            for behavior_type, behavior_opt in user_category_opt.items():
                if (behavior_type == "w"):
                    continue
                user_behavior_weight[behavior_type] += len(behavior_opt[ITEM_ID])

        if (user_behavior_weight[BEHAVIOR_TYPE_VIEW] == 0):
            user_behavior_weight[BEHAVIOR_TYPE_VIEW] = user_behavior_weight[BEHAVIOR_TYPE_FAV] + \
                                                       user_behavior_weight[BEHAVIOR_TYPE_CART] + \
                                                       user_behavior_weight[BEHAVIOR_TYPE_BUY]
        #计算每次fav, cart, buy对应着多少次view
        for behavior_type, behavior_cnt in user_behavior_weight.items():
            if (behavior_type != BEHAVIOR_TYPE_VIEW):
                user_behavior_weight[behavior_type] = safeDevision(user_behavior_weight[BEHAVIOR_TYPE_VIEW], behavior_cnt)

        user_behavior_weight[BEHAVIOR_TYPE_VIEW] = 1

        total = 0.0
        for behavior_type, behavior_cnt in user_behavior_weight.items():
            total += behavior_cnt

        #计算每种操作类型对于用户的权值
        for behavior_type, behavior_cnt in user_behavior_weight.items():
            user_behavior_weight[behavior_type] = behavior_cnt / total

        logging.info("behavior weight for user [%s] : %s" % (user_id, user_behavior_weight))

        # 计算用户操作过的 category 对于用户的权值
        # Sigma(用户在 category 上各种类型操作的数量 * 操作类型的权值) / Sigma(用户在所有 category 上各种类型操作的数量 * 操作类型的权值)
        total = 0.0
        for item_category, user_category_opt in user_opt.items():
            for behavior_type, behavior_opt in user_category_opt.items():
                if (behavior_type == "w"):
                    continue
                user_category_opt["w"] += len(behavior_opt[ITEM_ID]) * user_behavior_weight[behavior_type]

            total += user_category_opt["w"]

        for item_category, user_category_opt in user_opt.items():
            user_category_opt["w"] /= total

        #logging.info("user [%s] verifytotal %.3f / %.3f total, verifyweight %.3f" % (user_id, verifytotal, total, verifyweight))
        # if (verifyweight - 1.0 > 0.1):
        #     logging.error("user [%s] sum of category wieght(%.3f) is not 1! " % (user_id, verifyweight))

    logging.info("leaving calItemCategoryWeight...")
    return 0

def safeDevision(val1, val2):
    if (val1 == 0 or val2 == 0):
        return 0

    return val1/val2

def getMostFavCategoryOfUser(user_id):
    max_weight = 0.0
    most_fav_category = ""
    for item_category, user_category_opt in global_user_item_dict[user_id].items():
        if (max_weight < user_category_opt["w"]):
            max_weight = user_category_opt["w"]
            most_fav_category = item_category

    return most_fav_category, max_weight


def getItemsUserOpted(user_id):
    itemsUserOpted = []
    for user_opt in global_user_item_dict[user_id]:
        itemsUserOpted.append(user_opt[ITEM_CATE])

    return itemsUserOpted


def getPosOfDoubleHash(key1, key2, double_hash):
    if ( (key1 in double_hash and key2 in double_hash[key1]) or \
         (key2 in double_hash and key1 in double_hash[key2]) ):
         return None, None

    if ( key1 in double_hash):
        return key1, key2

    if (key2 in double_hash):
        return key2, key1

    double_hash[key1] = dict()

    return key1, key2

ISOTIMEFORMAT="%Y-%m-%d %X"
def getCurrentTime():
    return time.strftime(ISOTIMEFORMAT, time.localtime())

#检查test集中的某个category中的所有item， 是否都没有出现在train中
def checkItemExisting():
    for item_category, item_opt in global_item_user_dict.items():
        train_item_id_set = set(item_opt[ITEM_ID])
        test_item_id_set = set()

        if (item_category not in global_train_item):
            continue

        for test_item_id in global_train_item[item_category]:
            test_item_id_set.add(test_item_id[0])

        intersection = train_item_id_set.union(test_item_id_set) ^ (train_item_id_set ^ test_item_id_set)
        if (len(intersection) == 0):
            logging.info("all of Item of category %s in train do not exist in test!" % item_category)
        else:
            logging.info("category [%s] %s exist in both!" % (item_category, intersection))

    return 0    

def userHasOperatedItem(user_id, item_category, item_id):
    for behavior_type, behavior_opt in global_user_item_dict[user_id][item_category].items():
        if (behavior_type == "w"):
            continue

        if (item_id in behavior_opt):
            return True

    return False

def getRecordsFromRecordString(buy_records):
    all_records = []

    #去掉开头和末尾的 [[ ]]
    buy_records = buy_records[2 : len(buy_records)-2]

    #得到 购买记录 list
    #buy_records[i] = "(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)"
    buy_records = buy_records.split("], [")
    for idx  in range(len(buy_records)):
        #去掉开头和末尾的 ( )
        buy_records[idx] = buy_records[idx][1 : len(buy_records[idx])-1]

        #得到三元组list： '1, 0.0, 35'， '2, 0.0, 1'
        behavior_tuple_list =  buy_records[idx].split("), (")

        each_record = []
        current_view = None
        for behavior_tuple in behavior_tuple_list:
            behavior = behavior_tuple.split(", ")
            behavior_type  = int(behavior[0])
            #behavior_time  = float(behavior[1])
            behavior_time  =  datetime.datetime.strptime(behavior[1][1:-1], "%Y-%m-%d %H")
            behavior_count = int(behavior[2])

            each_record.append((behavior_type, behavior_time, behavior_count))

            #将连续的但是时间不同的 view 合并成一个
        #     #[ (1, time1, cnt1), (1, time2, cnt2), (2, time2, 1) ] 合并成 [ (1, time2, cnt1 + cnt2), (2, time2, 1) ]
        #     #[ (1, time1, cnt1), (2, time1, 1), (1, time2, cnt2) ] view 不连续则不合并
        #     if (current_view == None and behavior_type == BEHAVIOR_TYPE_VIEW):
        #         current_view = [behavior_time, behavior_count]
        #     elif (behavior_type == BEHAVIOR_TYPE_VIEW):
        #         current_view[1] += behavior_count
        #     else:
        #         if (current_view != None):
        #             each_record.append((BEHAVIOR_TYPE_VIEW, current_view[0], current_view[1]))
        #             each_record.append((behavior_type, behavior_time, behavior_count))
        #             current_view = None
        #         else:
        #             each_record.append((behavior_type, behavior_time, behavior_count))
        # if (current_view != None):
        #     each_record.append((BEHAVIOR_TYPE_VIEW, current_view[0], current_view[1]))

        all_records.append(each_record)

    return all_records

def loadTrainCategoryItemFromRedis():
    # 得到所有的 catrgory
    all_categories = redis_cli.get("all_category").decode()
    all_categories = all_categories.split(",")

    total_category = len(all_categories)
    print("%s loadCategoryItemFromRedis, here total %d categories in train set" % (getCurrentTime(), total_category))

    index = 0
    for category in all_categories:
        items_of_categories = redis_cli.hget("category_" + category, "item_id")
        items_of_categories = items_of_categories.decode().split(",")

        for item_id in items_of_categories:
            global_train_item_category[item_id] = category
            if (category not in global_train_category_item):
                global_train_category_item[category] = set()
            global_train_category_item[category].add(item_id)        
        index += 1

        print("%d / %d categories loaded\r" % (index, total_category), end="")

    return 0

def loadTrainCategoryItemAndSaveToRedis(train_user_file_name = tianchi_fresh_comp_train_user):
    filehandle1 = open(train_user_file_name, encoding="utf-8", mode='r')
    user_behavior = csv.reader(filehandle1)

    category_item_dict = dict()

    index = 0
    print("loadCategoryItemAndSaveToRedis(): loading file %s" % train_user_file_name)
    for aline in user_behavior:
        if (index == 0):
            index += 1
            continue

        item_id       = aline[1]
        item_category = aline[4]

        if (item_category not in category_item_dict):
            category_item_dict[item_category] = set()

        category_item_dict[item_category].add(item_id)
        index += 1

        if (index % 100000 == 0):
             print("%d lines read\r" % index, end="")

    print("%s saving category--item dict to redis..." % getCurrentTime())

    all_categories = list(category_item_dict.keys())
    redis_cli.set("all_category", ",".join(all_categories))
    total_category = len(all_categories)

    save_one_time = 1000

    idx = 0
    pipe = redis_cli.pipeline()
    for item_category, item_id_set in category_item_dict.items():
        pipe.hset("category_"+item_category, "item_id", ",".join(item_id_set))

        idx += 1
        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d categories saved to redis\r" % (idx, total_category), end="")

    if (idx % save_one_time != 0):
        pipe.execute()



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

g_users_for_alog = []

# 用户第一次接触商品到购买该商品之间的天数与用户购买该用户的可能性。天数越长，可能性越小
g_prob_bwteen_1st_days_and_buy = {1:0.0571, 2:0.032, 3:0.0221, 4:0.0164, 5:0.0138, 6:0.0098, 7:0.0089, 8:0.0077, 9:0.0062, 10:0.0055}


g_behavior_weight = {BEHAVIOR_TYPE_VIEW : 0.01,
                     BEHAVIOR_TYPE_FAV  : 0.33, 
                     BEHAVIOR_TYPE_CART : 0.47,
                     BEHAVIOR_TYPE_BUY  : 0.94}

# 每条购物记录在 redis 中都表现为字符串 
#"[ [(1, 2014-01-01 23, 35), (2, 2014-01-02 22, 1)], [(1, 2014-01-02 23, 35), (2, 2014-01-03 14, 1)] ]"
# run_for_users_cnt 跑多少个用户， 0 表示全部
def loadRecordsFromRedis(start_from, run_for_users_cnt):

    global g_buy_record_cnt
    global g_buy_record_cnt_verify
    global g_pattern_cnt

    g_user_buy_transection.clear()
    g_user_buy_transection_item.clear()
    g_user_behavior_patten.clear()

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
        # if (user_id != '50030386'):
        #     continue

        if (user_index < start_from):
            user_index += 1
            continue

        if (user_index > total_user + start_from):
            break

        g_users_for_alog.append(user_id)

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
                logging.info(" user %s buy %s: %s" % (user_id, item_id, g_user_buy_transection[user_id][item_id]))
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

    if (run_for_users_cnt > 0):
        logging.info("users for algo : %s" % g_users_for_alog)

    removeBuyCausedByDouble12()

    return

def loggingProbility(predicted_prob):
    m = np.shape(predicted_prob)[0]
    for i in range(m):
        if (predicted_prob[i][1] > predicted_prob[i][0]):
            logging.info("%.4f, %.4f" % (predicted_prob[i][0], predicted_prob[i][1]))
    return 



#用户从 1st view 到buy 的平均天数
def meanDaysFromViewToBuy():
    days_from_view_to_buy = dict()

    days = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                days += (each_record[-1][1].date() - each_record[0][1].date()).days
        days_from_view_to_buy[user_id] = round(days / len(g_user_buy_transection[user_id]))
        days = 0

    return days_from_view_to_buy

# 去掉由于双12导致的购买记录
def removeBuyCausedByDouble12():
    being_removed_user_item = dict()

    mean_days_from_view_to_buy = meanDaysFromViewToBuy()

    #从1st view 到12-12这天购买的天数少于平均天数，则认为是由于12-12导致的购买，则去掉
    december_twelve = datetime.datetime.strptime("2014-12-12", "%Y-%m-%d").date()
    for user_id, item_buy_records in g_user_buy_transection.items():
       for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                if (each_record[-1][1].date() != december_twelve):
                    continue
                days_from_view_to_buy = (each_record[-1][1].date() - each_record[0][1].date()).days
                if (days_from_view_to_buy < mean_days_from_view_to_buy[user_id]):
                    if (user_id not in being_removed_user_item):
                        being_removed_user_item[user_id] = set()

                    being_removed_user_item[user_id].add(item_id)
                    logging.info("deleting buy record %s - %s caused by 12-12, days %d, mean days %d" % 
                                 (user_id, item_id, days_from_view_to_buy, mean_days_from_view_to_buy[user_id]))
    removed_records = 0
    for user_id, item_list in being_removed_user_item.items():
        for item_id in item_list:
            del(g_user_buy_transection[user_id][item_id])
            removed_records += 1

        if (len(g_user_buy_transection[user_id]) == 0):
            del(g_user_buy_transection[user_id])
            logging.info("user %s has no buy records, delete it" % user_id)

    logging.info("total %d buy records deleted caused by 12-12" % removed_records)
    return     

def addSubFeatureMatIntoFeatureMat(sub_feature_mat, sub_feature_cnt, feature_mat, cur_total_feature_cnt):
    if (sub_feature_cnt > 0):
        feature_mat[:, cur_total_feature_cnt : cur_total_feature_cnt+sub_feature_cnt] = sub_feature_mat

    return sub_feature_cnt



# 统计用户第一次接触item到购买item之间的天数
def daysBetween1stBehaviorToBuy():
    print("%s calculating days between first behavior to buy..." % (getCurrentTime()))

    days_1st_beahvior_buy_dict = dict()
    
    total_buy = 0

    for user_id, item_id_buy in g_user_buy_transection.items():
        for item_id, buy_records in item_id_buy.items():
            for each_record in buy_records:
                timedelta = each_record[-1][1] - each_record[0][1]
                days = (each_record[-1][1] - each_record[0][1]).days
                if (days not in days_1st_beahvior_buy_dict):
                    days_1st_beahvior_buy_dict[days] = 0
                days_1st_beahvior_buy_dict[days] += 1
                if (total_buy % 1000 == 0):
                    print("    %d  buy records checkd\r" % (total_buy), end="")
                total_buy += 1

    for days, how_man_buy in days_1st_beahvior_buy_dict.items():
        logging.info("days, how_man_buy %d, %d" % (days, how_man_buy))

    days_list = days_1st_beahvior_buy_dict.keys()

    buy_vol = 0
    for days in days_list:
        buy_vol += days_1st_beahvior_buy_dict[days]
        logging.info("first %d days buy %d account for %.2f, total %d " % (days, buy_vol, buy_vol/total_buy, total_buy))

    exit(0)



def filterSamplesByProbility(samples, forecast_features, probability, min_proba):
    filtered_samples = []
    filtered_features = []
    filtered_Ymat = []
    for index, user_item in enumerate(samples):
        if (probability[index, 1] >= min_proba):
            filtered_samples.append(user_item)
            filtered_features.append(index)

    return filtered_samples, forecast_features[filtered_features, :]
