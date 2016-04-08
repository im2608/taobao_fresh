import sys
import csv
import time
import datetime
import logging
import redis


USER_ID = "user_id"
ITEM_ID = "item_id"
BEHAVEIOR_TYPE = "behavior_type"
USER_GEO = "user_geohash"
ITEM_CATE = "item_category"
TIME = "time"


BEHAVIOR_TYPE_VIEW = 1
BEHAVIOR_TYPE_FAV  = 2
BEHAVIOR_TYPE_CART = 3
BEHAVIOR_TYPE_BUY  = 4

runningPath = sys.path[0]
tianchi_fresh_comp_train_user = "%s\\..\\input\\tianchi_fresh_comp_train_user.csv" % runningPath
tianchi_fresh_comp_train_item = "%s\\..\\input\\tianchi_fresh_comp_train_item.csv" % runningPath

# 以 user id 为 key
global_user_item_dict = dict()

# 以item category 为 key
global_item_user_dict = dict()


global_train_item = set()

global_totalBehaviorWeightHash = dict()
global_user_behavior_cnt = dict()

redis_cli = redis.Redis(host='10.57.14.3', port=6379, db=0)

# CRITICAL 50
# ERROR    40
# WARNING  30
# INFO     20
# DEBUG    10
# NOTSET   0
logging.basicConfig(level=logging.INFO,\
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                    datefmt='%a, %d %b %Y %H:%M:%S',\
                    filename='..\\log\\log.test.txt',\
                    filemode='w')

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

def loadTrainItem():
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

        global_train_item.add(item_id)

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
