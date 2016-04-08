from common import *
import userCF
import itemCF
import apriori

def splitHistoryData(fileName, splited_files):
    print(" reading data file ", fileName)
    dataHistory = {}
    historyDataFile = open(fileName, encoding="utf-8", mode='r')
    print("splited_files is  %d" % splited_files)

    splitedFileHandle = []
    for fileIdx in range(splited_files):
        splitedFileName = "%s\\..\\input\\splitedInput\\datafile.%03d" % (runningPath, fileIdx)
        dataFile = open(splitedFileName, "w")
        splitedFileHandle.append(dataFile)
        print("%s created" % splitedFileName)

    lineIdx = 0

    for aline in historyDataFile.readlines():
        if (lineIdx == 0):
            lineIdx = 1
            continue

        userId = aline.split(",")[0]
        fileIdx = hash(userId) % splited_files
        splitedFileHandle[fileIdx].write(aline)

    print("history data file is read")

    for fileIdx in range(splited_files):
        splitedFileHandle[fileIdx].close()

    historyDataFile.close()

    return 0


#检查所有用户操作过的物品id是否都在 sub item 表中
def checkItem(train_user_file_name, train_item_file_name):
    operated_item_cats = dict()
    sub_item_cats = dict()

    user_behavior = csv.reader(open(train_user_file_name, encoding="utf-8", mode='r'))
    sub_item_info = csv.reader(open(train_item_file_name, encoding="utf-8", mode='r'))
    index = 0
    for aline in user_behavior:
        if (index == 0):
            index += 1
            continue

        operated_item_cats[aline[2]] = 1

    index = 0
    for aline in sub_item_info:
        if (index == 0):
            index += 1
            continue
        sub_item_cats[aline[0]] = 1

    missed_cnt = 0
    for item_cat in sub_item_cats.keys():
        if (item_cat not in operated_item_cats):
            missed_cnt += 1

    print("total %d sum item missed in operated item table!" % missed_cnt)

    return 0

def directBuy():
    directly_buy_users = 0
    total_users = 0
    for user_id, item_categories in global_user_item_dict.items():
        total_users += 1
        directly_bought_cate = dict()
        for category, operation_info in item_categories.items():
            viewing = False
            buy = False
            for behavior_idx in range(len(operation_info[BEHAVEIOR_TYPE])):
                if (operation_info[BEHAVEIOR_TYPE][behavior_idx] == 4):
                    buy = True                    
                else:
                    viewing = True

                if (viewing and buy):
                    break

            if ((not viewing) and buy):
                if (operation_info[TIME][behavior_idx] not in directly_bought_cate):
                    directly_bought_cate[operation_info[TIME][behavior_idx]] = []
                directly_bought_cate[operation_info[TIME][behavior_idx]].append(category)

        if (len(directly_bought_cate) > 0):
            directly_buy_users += 1
            logging.info("user %s directly bought\n%s" % (user_id, directly_bought_cate))

    logging.info("total %d / %d user directly buoght without viewing!" % (directly_buy_users, total_users))

    return 0

def getUserItemCatalogCnt(filename):
    user_cnt = set()
    item_catelog_cnt = set()

    filehandle = open(filename, encoding="utf-8", mode='r')
    user_behavior = csv.reader(filehandle)
    index = 0
    for aline in user_behavior:
        if (index == 0):
            index += 1
            continue

        user_cnt.add(aline[0])
        item_catelog_cnt.add(aline[4])

    print("here are %d users and %d item catelogies in file %s" % (len(user_cnt), len(item_catelog_cnt), filename))

    filehandle.close()

    return 0



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
file_idx = 0
data_file = "%s\\..\\input\\splitedInput\\datafile.%03d" % (runningPath, file_idx)

need_verify = True
factor = 0.1
#apriori.loadDataAndSaveToRedis(need_verify)
print("factor is %.4f, need verify %d" % (factor, need_verify))
#loadTrainItem()
apriori.loadRecordsFromRedis(factor, need_verify)
apriori.Bayes(need_verify)

#apriori.loadFrequentItemsFromRedis()
#L = apriori.aprioriAlgorithm()
#apriori.matchPatternAndFrequentItem(L, factor, need_verify)
if (need_verify):
    apriori.verificationForecast()

#apriori.saveFrequentItemToRedis(L)


#getUserItemCatalogCnt(data_file)


# loadData()
# checkItemExisting()
#calItemCategoryWeight()
#directBuy()
#userCF.UserCollaborativeFiltering()
#userCF.recommendationUserCF(5)
#itemCF.ItemCollaborativeFiltering()



#splitHistoryData(tianchi_fresh_comp_train_user, 100)
#checkItem(tianchi_fresh_comp_train_user, tianchi_fresh_comp_train_item)
