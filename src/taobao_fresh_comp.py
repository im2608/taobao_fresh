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

    

#"[('41209588', '326492304'), ('25286173', '187742447'), ('129020685', '248639479')]"
def loadTestingFeaturesFromRedis():    
    samples_test_str = redis_cli.get("testing_samples").decode()

    #去掉首尾的 [(' and ')]
    samples_test_str = samples_test_str[3 : len(samples_test_str)-3]
    # 分割成元组数组
    samples_test_list = samples_test_str.split("'), ('")

    for index in range(len(samples_test_list)):
        # 去掉每个元组首尾的( and ):  ('41209588', '326492304')
        samples_test_list[index] = samples_test_list[index][0 : len(samples_test_list[index]) - 1]
        user_item = samples_test_list[index].split("', '")
        samples_test_list[index] = (user_item[0], user_item[1])

    print("load %d tesing samples from redis" % len(samples_test_list))
    return samples_test_list

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
sys.path.append("%s\\LR-hit\\" % runningPath)
sys.path.append("%s\\RF\\" % runningPath)
sys.path.append("%s\\GBDT\\" % runningPath)
sys.path.append("%s\\features\\" % runningPath)
sys.path.append("%s\\samples\\" % runningPath)

import LR
import RF
import GBDT

file_idx = 53
data_file = "%s\\..\\input\\splitedInput\\datafile.%03d" % (runningPath, file_idx)

need_verify = False
factor = 0.1

#apriori.loadDataAndSaveToRedis(need_verify)
#loadTrainCategoryItemAndSaveToRedis()

print("--------------------------------------------------")
print("--------------- Starting... ----------------------")
print("--------------------------------------------------")
loadTrainCategoryItemFromRedis()
loadTestSet()

need_output = 1
print("%s ************** output is %d ************" % (getCurrentTime(), need_output))

if (algo == "Apriori"):
    print("%s ============ Algorithm is Apriori ============" % (getCurrentTime()))

    apriori.loadRecordsFromRedis(factor, need_verify)
    # apriori.Bayes(need_verify)

    #apriori.loadFrequentItemsFromRedis()
    #L = apriori.aprioriAlgorithm()
    #apriori.matchPatternAndFrequentItem(L, factor, need_verify)
    if (need_verify):
        apriori.verificationForecast()
    #apriori.saveFrequentItemToRedis(L)

elif (algo == "LR"):
    print("%s ============ Algorithm is Logistic Regression ============" % (getCurrentTime()))
    if (need_output == 1):
        start_from = 0
        user_cnt = 0
        checking_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()
        forecast_date = checking_date + datetime.timedelta(1)
        loadRecordsFromRedis(start_from, user_cnt)
        LR.logisticRegression(user_cnt, checking_date, forecast_date, need_output)
    else:
        start = 0
        user_cnt = 0
        checking_date = datetime.datetime.strptime("2014-12-05", "%Y-%m-%d").date()
        forecast_date = checking_date + datetime.timedelta(1)
        loadRecordsFromRedis(start, user_cnt)
        LR.logisticRegression(user_cnt, checking_date, forecast_date, need_output)
elif (algo == "RF"):
    print("%s ============ Algorithm is Random Forest ============" % (getCurrentTime()))
    if (need_output == 1):
        start_from = 0
        user_cnt = 0
        checking_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()
        forecast_date = checking_date + datetime.timedelta(1)
        loadRecordsFromRedis(start_from, user_cnt)
        RF.randomForest(user_cnt, checking_date, forecast_date, need_output)
    else:
        start = 0
        user_cnt = 2000
        checking_date = datetime.datetime.strptime("2014-12-05", "%Y-%m-%d").date()
        forecast_date = checking_date + datetime.timedelta(1)
        loadRecordsFromRedis(start, user_cnt)
        RF.randomForest(user_cnt, checking_date, forecast_date, need_output)
elif (algo == "GBDT")        :
    print("%s ============ Algorithm is GBDT ============" % (getCurrentTime()))
    start = 0
    user_cnt = 0
    checking_date = datetime.datetime.strptime("2014-12-05", "%Y-%m-%d").date()
    forecast_date = checking_date + datetime.timedelta(1)
    loadRecordsFromRedis(start, user_cnt)
    GBDT.GradientBoostingRegressionTree(checking_date, forecast_date, need_output)



print("=====================================================================")
print("=========================  training...   ============================")
print("=====================================================================")

start_from = 0
user_cnt = 200
window_start_date = datetime.datetime.strptime("2014-11-18", "%Y-%m-%d").date()
window_end_date = datetime.datetime.strptime("2014-11-27", "%Y-%m-%d").date()
final_end_date = datetime.datetime.strptime("2014-12-01", "%Y-%m-%d").date()
features_importance = None
loadRecordsFromRedis(start_from, user_cnt)

while (window_end_date <= final_end_date):    
    if (features_importance is None):
        features_importance = GBDT.GBDT_slideWindows(window_start_date, window_end_date, True)
    else:
        features_importance += GBDT.GBDT_slideWindows(window_start_date, window_end_date, True)

    window_end_date += datetime.timedelta(1)
    window_start_date += datetime.timedelta(1)    

useful_features = features_importance[features_importance > 0.0]
for feature_idx in useful_features:
    for feature_name in g_feature_info:
        if (g_feature_info[feature_name] == feature_idx):
            g_useful_feature_info[feature_name] = feature_idx
            break

logging.info("After split window, features_importance is " % features_importance)    

print("=====================================================================")
print("=========================  forecasting...============================")
print("=====================================================================")

window_start_date = datetime.datetime.strptime("2014-11-18", "%Y-%m-%d").date()
window_end_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()
forecast_date = datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()
nag_per_pos = 10

print("        %s checking date %s" % (getCurrentTime(), checking_date))

# #item 的热度
print("        %s calculating popularity..." % getCurrentTime())
#item_popularity_dict = calculate_item_popularity()
item_popularity_dict = calculateItemPopularity(window_start_date, window_end_date)
print("        %s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))

samples, Ymat = takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos, item_popularity_dict)

print("        %s taking samples for forecasting (%s, %d)" % (getCurrentTime(), window_end_date, nag_per_pos))
samples, Ymat = takingSamplesForTesting(window_start_date, window_end_date, nag_per_pos, item_popularity_dict)
print("        %s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))

Xmat = createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, item_popularity_dict, False)

Xmat, Ymat = shuffle(Xmat, Ymat, random_state=13)

params = {'n_estimators': 100, 
          'max_depth': 4,
          'min_samples_split': 1,
          'learning_rate': 0.01, 
          'loss': 'ls'
          }

clf = GradientBoostingRegressor(**params)
clf.fit(Xmat, Ymat)

X_leaves = clf.apply(Xmat)

samples_test = takingSamplesForTesting(forecast_date)
