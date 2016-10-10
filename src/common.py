from sklearn.preprocessing import OneHotEncoder


import csv
import time
import datetime
import logging
import numpy as np
from sklearn import preprocessing
from global_variables import *

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
        # logging.info("getRecordsFromRecordString(): skip buy records %s" % buy_records)

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




# 每条购物记录在 redis 中都表现为字符串 
#"[[(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)], [(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)], [(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)]]"
def loadDataAndSaveToRedis(need_verify = True, user_opt_file_name = tianchi_fresh_comp_train_user):    
    print("%s loadDataAndSaveToRedis" % getCurrentTime())

    total_buy_record_cnt = 0
    total_pattern_cnt = 0

    filehandle1 = open(user_opt_file_name, encoding="utf-8", mode='r')

    user_behavior_csv = csv.reader(filehandle1)
    index = 0

    user_behavior_record = dict()
    skiped_buy_cnt = 0
    logging.info("loading file %s" % user_opt_file_name)
    for aline in user_behavior_csv:
        if (index == 0):
            index += 1
            continue

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        user_geohash  = aline[3]
        item_category = aline[4]
        behavior_time = datetime.datetime.strptime(aline[5], "%Y-%m-%d %H")

        if (index % 100000 == 0):
            print("%d lines read\r" % index,  end="")

        if (user_id not in user_behavior_record):
            user_behavior_record[user_id] = dict()

        if (item_id not in user_behavior_record[user_id]):
            user_behavior_record[user_id][item_id] = []

        #用户在某个 item id 上的每个操作以（操作类型，操作时间, 操作次数） 的元组表示
        #user_behavior_seq 为该元组序列，按照时间排序，相同时间的按照1，2，3，4排序
        user_behavior_seq = user_behavior_record[user_id][item_id]
        seq_len = len(user_behavior_seq)
        #按照操作时间生成操作序列
        if (seq_len == 0):
            user_behavior_seq.append((behavior_type, behavior_time))
        else:
            if (behavior_time > user_behavior_seq[seq_len - 1][1]):
                user_behavior_seq.insert(seq_len, (behavior_type, behavior_time))
                continue

            for idx in range(seq_len):
                if (behavior_time <= user_behavior_seq[idx][1]):
                    user_behavior_seq.insert(idx, (behavior_type, behavior_time))
                    break
        index += 1        

    print("\r\ntotal %d lines read" % index)

    #根据操作序列得到用户的购买记录，以及pattern
    print("%s getting user buy records" % getCurrentTime())
    index = 0
    total_user = len(user_behavior_record)
    for user_id, item_id_list in user_behavior_record.items(): #用户， 该用户在哪些 item 上有操作

        for item_id, behavior_seq in item_id_list.items(): 
            #用户在某个 item 上只有一次非购物操作，略过
            if (len(behavior_seq) == 1 and behavior_seq[0][0] != BEHAVIOR_TYPE_BUY):
                continue

            #用户购买记录，按照时间排序，相同时间的话，1，2，3在前，4在后
            #将连续的行为压缩成 {behavior：count}的形式
            #[1,1,1,2,3,4] => [((1, time), 3), ((2, time), 1), ((3, time), 1), ((4, time), 1)]
            sorted_seq = sortAndCompressBuyRecord(behavior_seq)

            user_buy_record = []
            #logging.debug("user: %s, item_id: %s, behavior seq: %s, sorted seq: %s" % (user_id, item_id, behavior_seq, sorted_seq))
            for behavior_consecutive in sorted_seq:
                behavior_type = behavior_consecutive[0][0]
                behavior_time = behavior_consecutive[0][1]
                user_buy_record.append(behavior_consecutive)

                if (behavior_type != BEHAVIOR_TYPE_BUY):
                    continue

                #有些购物记录没有任何浏览记录，跳过
                if (len(user_buy_record) == 0):
                    total_buy_record_cnt += 1
                    skiped_buy_cnt += 1
                    continue

                for idx in range(len(user_buy_record)):
                    #重新生成新的三元组 (操作类型， 操作时间 2014-01-23， 操作次数)
                    user_buy_record[idx] = (user_buy_record[idx][0][0], \
                                            convertDatatimeToStr(user_buy_record[idx][0][1]),\
                                            user_buy_record[idx][1])


                if (user_id not in g_user_buy_transection):
                    g_user_buy_transection[user_id] = dict()
            
                if (item_id not in g_user_buy_transection[user_id]):
                    g_user_buy_transection[user_id][item_id] = []

                # #如果有连续的购买，则为每个购买行为生成一条购物记录
                # buy_cnt = behavior_consecutive[1]
                # for each_buy in range(buy_cnt):
                #     g_user_buy_transection[user_id][item_id].append(user_buy_record.copy())
                #     total_buy_record_cnt += 1

                total_buy_record_cnt += behavior_consecutive[1]

                user_buy_record.clear()

            if (len(user_buy_record) > 0):
                if (user_id not in g_user_behavior_patten):
                    g_user_behavior_patten[user_id] = dict()

                if (item_id not in g_user_behavior_patten[user_id]):
                    g_user_behavior_patten[user_id][item_id] = []

                for idx in range(len(user_buy_record)):                
                    #重新生成新的三元组
                    user_buy_record[idx] = (user_buy_record[idx][0][0], \
                                            convertDatatimeToStr(user_buy_record[idx][0][1]),\
                                            user_buy_record[idx][1])

                g_user_behavior_patten[user_id][item_id].append(user_buy_record.copy())
                total_pattern_cnt += 1
        index += 1
        print("%d /%d users checked\r" % (index, total_user), end="")

    saveRecordstoRedis()

    # logging.info("g_user_behavior_patten %s" % g_user_behavior_patten)
    # logging.info("g_user_buy_transection %s" % g_user_buy_transection)

    #logginBuyRecords()

    filehandle1.close()

    return 0

#用户购买记录，按照时间排序，相同时间的情况下，1，2，3排在前，4在后
def sortAndCompressBuyRecord(user_buy_record):
    sorted_compressed_behavior = []

    for user_behavior in user_buy_record: #user_behavior is [behavior, time]
        sorted_len = len(sorted_compressed_behavior)
        if (sorted_len == 0):
            sorted_compressed_behavior.append([user_behavior, 1])
            continue

        if (user_behavior[1] > sorted_compressed_behavior[sorted_len - 1][0][1]):
            sorted_compressed_behavior.append([user_behavior, 1])
            continue

        if (user_behavior[1] < sorted_compressed_behavior[0][0][1]):
            sorted_compressed_behavior.insert(0, [user_behavior, 1])
            continue

        inserted = False
        idx = 0
        for behavior_consecutive in sorted_compressed_behavior:
            if (behavior_consecutive[0] == user_behavior):
                behavior_consecutive[1] += 1
                inserted = True
                break
            if (user_behavior[1] < behavior_consecutive[0][1] or \
                (user_behavior[1] == behavior_consecutive[0][1] and user_behavior[0] < behavior_consecutive[0][0])):
                break
            idx += 1

        if (not inserted):
            sorted_compressed_behavior.insert(idx, [user_behavior, 1])

    return sorted_compressed_behavior


def convertDatatimeToStr(opt_datatime):
    return "%04d-%02d-%02d %02d" % (opt_datatime.year, opt_datatime.month, opt_datatime.day, opt_datatime.hour)

def logginBuyRecords():
    print("%s logginBuyRecords" % (getCurrentTime()))
    for user_id, item_id_buy in g_user_buy_transection.items():
        buy_cnt = 0
        for item_id, buy_records in item_id_buy.items():
            for each_record in buy_records:
                logging.info("user %s item id %s : %s" % (user_id, item_id, each_record))
            buy_cnt += len(buy_records)
        logging.info("user %s -- %d buy records" % (user_id, buy_cnt))

    for user_id, item_id_opt in g_user_behavior_patten.items():
        for item_id, behavior_pattern in item_id_opt.items():
            logging.info("user %s item category %s  behavior pattern: %s" % (user_id, item_id, behavior_pattern))

    print("%s logginBuyRecords Done" % (getCurrentTime()))
    return 0

def saveRecordstoRedis():
    print("%s saveBuyRecordstoRedis()" % getCurrentTime())
    all_users = list(g_user_buy_transection.keys())
    redis_cli.set("all_users", ",".join(all_users))
    total_user = len(all_users)

    save_one_time = 1000

    idx = 0
    pipe = redis_cli.pipeline()
    for user_id, item_id_buy in g_user_buy_transection.items():        
        item_id_str = list(item_id_buy.keys())
        pipe.hset(user_id, "item_id", ",".join(item_id_str))

        for item_id, buy_records in item_id_buy.items():
            pipe.hset(user_id, item_id, buy_records)

        idx += 1
        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d saved to redis\r" % (idx, total_user), end="")

    if (idx % save_one_time != 0):
        pipe.execute()

    print("")

    all_users_verify = list(g_user_buy_transection_verify.keys())
    total_verify_user = len(all_users_verify)

    redis_cli.set("all_users_verify", ",".join(all_users_verify))

    idx = 0
    for user_id, item_id_buy in g_user_buy_transection_verify.items():
        item_id_str = list(item_id_buy.keys())
        pipe.hset(user_id, "item_id_verify", ",".join(item_id_str))

        for item_id, buy_records in item_id_buy.items():
            pipe.hset(user_id, item_id+"_verify", buy_records)
        idx += 1
        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d verify user saved to redis\r" % (idx, total_verify_user), end="")

    if (idx % save_one_time != 0):
        pipe.execute()

    print("")

    print("%s save patterns to redis" % getCurrentTime())
    idx = 0
    total_user = len(g_user_behavior_patten)
    for user_id, item_id_opt in g_user_behavior_patten.items():        
        item_id_str = list(item_id_opt.keys())
        pipe.hset(user_id, "item_id_pattern", ",".join(item_id_str))

        for item_id, item_pattern in item_id_opt.items():
            pipe.hset(user_id, item_id+"_pattern", item_pattern)
        idx += 1

        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d user pattern saved to redis\r" % (idx, total_user), end="")

    print("")
    if (idx % save_one_time != 0):
        pipe.execute()

    print("%s saveBuyRecordstoRedis() Done!" % getCurrentTime())

    return 0



# 每条购物记录在 redis 中都表现为字符串 
#"[ [(1, 2014-01-01 23, 35), (2, 2014-01-02 22, 1)], [(1, 2014-01-02 23, 35), (2, 2014-01-03 14, 1)] ]"
# run_for_users_cnt 跑多少个用户， 0 表示全部
def loadRecordsFromRedis(start_from, run_for_users_cnt):
    total_buy_record_cnt = 0
    total_pattern_cnt = 0

    g_user_buy_transection.clear()
    g_user_buy_transection_item.clear()
    g_user_behavior_patten.clear()
    g_users_for_alog.clear()

    # 得到所有的用户
    all_users = redis_cli.get("all_users").decode()
    all_users = all_users.split(",")

    total_user = len(all_users)
    if (run_for_users_cnt > 0):
        total_user = run_for_users_cnt
    else:
        start_from = 0

    print("%s loadRecordsFromRedis, start from %d read %d users" % (getCurrentTime(), start_from, total_user))

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

        user_whole_info = redis_cli.hgetall(user_id)

        if (user_id not in g_user_buy_transection):
            g_user_buy_transection[user_id] = dict()

        item_id_list = user_whole_info[bytes("item_id".encode())].decode()
        if (len(item_id_list) > 0):
            item_id_list = item_id_list.split(",")
            for item_id in item_id_list:

                item_buy_record = user_whole_info[bytes(item_id.encode())].decode()

                #读取购物记录
                g_user_buy_transection[user_id][item_id] = getRecordsFromRecordString(item_buy_record)

                logging.info(" user %s buy %s: %s" % (user_id, item_id, g_user_buy_transection[user_id][item_id]))

                if (item_id not in g_user_buy_transection_item):
                    g_user_buy_transection_item[item_id] = dict()

                g_user_buy_transection_item[item_id][user_id] = g_user_buy_transection[user_id][item_id]
                total_buy_record_cnt += len(g_user_buy_transection[user_id][item_id])
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
        total_pattern_cnt += len(item_pattern_list)

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

    print("%s total buy count %d, pattern count %d " % (getCurrentTime(), total_buy_record_cnt, total_pattern_cnt))
    logging.info("%s total buy count %d, pattern count %d " % (getCurrentTime(), total_buy_record_cnt, total_pattern_cnt))

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




def filterSamplesByProbility(samples, forecast_features, probability, min_proba):
    filtered_samples = []
    filtered_features = []
    filtered_Ymat = []
    for index, user_item in enumerate(samples):
        if (probability[index, 1] >= min_proba):
            filtered_samples.append(user_item)
            filtered_features.append(index)

    return filtered_samples, forecast_features[filtered_features, :]

# 从按照降序排好序的tuple中返回次序
# 如： [(id1, 100), (id2, 100), (id3, 90), (id4, 90), (id5, 80), (id6, 70)] =>
# [1, 1, 2, 2, 3, 4]
def getRankFromSortedTuple(sorted_tuple):
    rank = 1
    cur_val = sorted_tuple[0][1]
    sorted_rank = []
    for sorted_val in sorted_tuple:
        if (sorted_val[1] < cur_val):
            cur_val = sorted_val[1]
            rank += 1
        sorted_rank.append(rank)
    return sorted_rank


# 对onehot 编码，这里假设排序名次最多为 max_rank
# g_rank_onehot_enc = OneHotEncoder()
# max_rank = 150

# tmp = [x for x in range(max_rank)]
# tmp = np.array(tmp).reshape(-1, 1)
# g_rank_onehot_enc.fit(tmp)

# def oneHotEncodeRank(rank):
#     rank = np.array(rank).reshape(-1, 1)
#     rank_onehot = g_rank_onehot_enc.transform(rank).toarray()

#     return rank_onehot


def getOnehotEncoder(slide_windows_models, Xmat_weight, Xmat_forecast, n_estimators):
    onehot_val = [0 for x in range(n_estimators)]
    max_apply = 0

    for X_useful_mat_clf_model in slide_windows_models:
        X_useful_mat = X_useful_mat_clf_model[0]
        clf_model = X_useful_mat_clf_model[1]
        X_train_lr_enc = clf_model.apply(Xmat_weight)[:, :, 0]
        X_forecast_lr_enc = clf_model.apply(Xmat_forecast)[:, :, 0]
        for i in range(n_estimators):
            onehot_val[i] = max(onehot_val[i], X_train_lr_enc[:, i].max(), X_forecast_lr_enc[:, i].max())
            max_apply = max(max_apply, X_train_lr_enc[:, i].max(), X_forecast_lr_enc[:, i].max())

    onehot_mat = np.zeros((max_apply + 1, n_estimators))
    for i in range(n_estimators):
        for j in range(int(onehot_val[i]) + 1):
            onehot_mat[j, i] = j

    logging.info("max_apply is %d, columns of onehot_mat is %d, sum(onehot_val) %d" % (max_apply, onehot_mat.shape[1], sum(onehot_val)))
    np.savetxt("%s\\..\log\\onehot_mat.txt" % runningPath, onehot_mat, fmt="%.4f", newline="\n")

    onehot_enc = OneHotEncoder()
    onehot_enc.fit(onehot_mat)

    return onehot_enc

def calculateUserActivity(window_start_date, window_end_date, user_records, user_activity_score_dict, user_item_pairs):

    if ("total_activity" not in user_activity_score_dict):
        user_activity_score_dict["total_activity"] = 0

    for user_item in user_item_pairs:
        user_id = user_item[0]
        item_id = user_item[1]

        if (user_id not in user_records or 
            item_id not in user_records[user_id]):        
            continue

        if (user_id not in user_activity_score_dict):
            user_activity_score_dict[user_id] = dict()
            user_activity_score_dict[user_id]["activity_on_item"] = []
            user_activity_score_dict[user_id]["activity"] = 0            

        user_activiey_on_item = 0
        for each_record in user_records[user_id][item_id]:
            for behavior in each_record:
                if (behavior[1].date() >= window_start_date and 
                    behavior[1].date() < window_end_date):
                    days = (window_end_date - behavior[1].date()).days
                    # 用户在item上的行为次数 * 行为的权值 * 行为日至购买日之间的天数导致购买的可能性 = 用户在item上的分数， 分数越高，用户对item越感兴趣
                    user_activiey_on_item += behavior[2] * g_behavior_weight[behavior[0]] * g_prob_bwteen_1st_days_and_buy[days]

        user_activity_score_dict[user_id]["activity_on_item"].append((item_id, user_activiey_on_item))
        user_activity_score_dict[user_id]["activity"] += user_activiey_on_item
        user_activity_score_dict["total_activity"] += user_activiey_on_item

    # 按照用户对item的分数从高到低排序
    for user_id, item_score in user_activity_score_dict.items():
        if (user_id == "total_activity"):
            continue

        user_activity_score_dict[user_id]["activity_on_item"] = sorted(item_score["activity_on_item"], key=lambda item:item[1], reverse=True)
        # logging.info("user activity %s %.2f activity %s" % 
        #              (user_id, 
        #               user_activity_score_dict[user_id]["activity"], 
        #               user_activity_score_dict[user_id]["activity_on_item"]))

    return


def modelBlending_iterate(logisticReg, X_train_features, min_proba):
    iterating_count = 3
    for iterator in range(iterating_count):
        Ymat_iter = []

        pretected_prob = logisticReg.predict_proba(X_train_features)
        for i, prob in enumerate(pretected_prob):
            if (pretected_prob[i, 1] >= min_proba):
                Ymat_iter.append(1)
            else:
                Ymat_iter.append(0)

        logisticReg.fit(X_train_features, Ymat_iter)

    return logisticReg


def sigmoid(inX):
    return 1.0/(1.0 + np.exp(-inX))

def ridgeRegress(Xmat, Ymat, lam):
    print("ridgeRegress Xmat (%d, %d), Ymat (%d)" %(Xmat.shape[0], Xmat.shape[1], len(Ymat)))
    xTx = Xmat.T * Xmat
    demon = xTx + np.eye(np.shape(Xmat)[1]) * lam
    if (np.linalg.det(demon) == 0.0):
        print("Xmat is singular, can not do inverse")
        return None
    tmp = Xmat.T * np.mat(Ymat).T
    ws = demon.I * tmp
    return ws
