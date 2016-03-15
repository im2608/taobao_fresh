import sys
import csv
import time
import datetime

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

global_totalBehaviorCntHash = dict()
global_totalBehaviorWeightHash = dict()


def loadData(train_user_file_name = tianchi_fresh_comp_train_user, train_item_file_name = tianchi_fresh_comp_train_item):
    filehandle1 = open(train_user_file_name, encoding="utf-8", mode='r')
    user_behavior = csv.reader(filehandle1)

    filehandle2 = open(train_item_file_name, encoding="utf-8", mode='r')
    item_info  = csv.reader(filehandle2)
    index = 0
    for aline in user_behavior:
        if (index == 0):
            index += 1
            continue

        #哪些用户对哪些物品有操作
        if (aline[0] not in global_user_item_dict):
            global_user_item_dict[aline[0]] = dict()

        item_category = int(aline[4])
        if (item_category not in global_user_item_dict[aline[0]]):
            global_user_item_dict[aline[0]][item_category] = dict()
            global_user_item_dict[aline[0]][item_category][ITEM_ID] = []
            global_user_item_dict[aline[0]][item_category][BEHAVEIOR_TYPE] = []
            global_user_item_dict[aline[0]][item_category][TIME] = []

        global_user_item_dict[aline[0]][item_category][ITEM_ID] = aline[1]
        global_user_item_dict[aline[0]][item_category][BEHAVEIOR_TYPE] = int(aline[2])
        global_user_item_dict[aline[0]][item_category][TIME] = datetime.datetime.strptime(aline[5], "%Y-%m-%d %H")

        if (item_category not in global_item_user_dict):
            global_item_user_dict[item_category] = set()

        #哪些物品分类被哪些用户操作
        global_item_user_dict[item_category].add(aline[0])

        bev_type = int(aline[2])
        if (bev_type not in global_totalBehaviorCntHash):
            global_totalBehaviorCntHash[bev_type] = 0

        global_totalBehaviorCntHash[bev_type] += 1

    print("global_totalBehaviorCntHash is ", global_totalBehaviorCntHash)

    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_VIEW] = 1
    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_FAV] = global_totalBehaviorCntHash[BEHAVIOR_TYPE_VIEW] / global_totalBehaviorCntHash[BEHAVIOR_TYPE_FAV]
    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_CART] = global_totalBehaviorCntHash[BEHAVIOR_TYPE_VIEW] / global_totalBehaviorCntHash[BEHAVIOR_TYPE_CART]
    global_totalBehaviorWeightHash[BEHAVIOR_TYPE_BUY] = global_totalBehaviorCntHash[BEHAVIOR_TYPE_VIEW] / global_totalBehaviorCntHash[BEHAVIOR_TYPE_BUY]

    print("global_totalBehaviorWeightHash is ", global_totalBehaviorWeightHash)

    filehandle1.close()
    filehandle2.close()

    return 0


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
