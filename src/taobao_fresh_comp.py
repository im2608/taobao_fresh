from common import *
import userCF
import itemCF
import apriori
import os
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier)

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

import LR
import RF
import GBDT
from taking_sample import *
import verify

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
user_cnt = 0
slide_windows_days = 10

if (user_cnt == 0):
    window_start_date = datetime.datetime.strptime("2014-11-18", "%Y-%m-%d").date()
    final_end_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()
else:
    window_start_date = datetime.datetime.strptime("2014-12-02", "%Y-%m-%d").date()
    final_end_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()

window_end_date = window_start_date + datetime.timedelta(slide_windows_days)

features_importance = None
loadRecordsFromRedis(start_from, user_cnt)
slide_windows_models = []

#滑动窗口 11-18 -- 12-17 得到特征的重要性， 并保留每个滑动窗口训练出的的model
while (window_end_date < final_end_date):    
    slide_window_clf = GBDT.GBDT_slideWindows(window_start_date, window_end_date, True)
    if (slide_window_clf == None):
        window_end_date += datetime.timedelta(1)
        window_start_date += datetime.timedelta(1)
        continue

    slide_windows_models.append(slide_window_clf)
    print("type(slide_window_clf)", type(slide_window_clf))

    if (features_importance is None):
        features_importance = slide_window_clf.feature_importances_
    else:
        features_importance += slide_window_clf.feature_importances_

    window_end_date += datetime.timedelta(1)
    window_start_date += datetime.timedelta(1)

logging.info("After split window, features_importance is %s" % features_importance)

updateUsefulFeatures(features_importance)

nag_per_pos = 10

# 根据滑动窗口的结果，使用重要性 > 0 的特征从 12-08 -- 12-17 生成特征矩阵以及12-18 的购买记录，交给滑动窗口
# 训练出的model，生成叶节点，传给 LR 再进行训练， 最后使用 LR 从 12-09 -- 12-18 生成特征矩阵进行预测
print("=====================================================================")
print("==============  generating weights %s - %s ==========" % (window_start_date, window_end_date))
print("=====================================================================")

# item 的热度
print("        %s calculating popularity..." % getCurrentTime())
#item_popularity_dict = calculate_item_popularity()
item_popularity_dict = calculateItemPopularity(window_start_date, window_end_date)
print("        %s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))

samples_weight, Ymat_weight = takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos, item_popularity_dict)

# 使用重要性 > 0 的特征从 12-08 -- 12-17 生成特征矩阵
print("        %s forecasting, reading feature matrix from %s -- %s" % (getCurrentTime(), window_start_date, window_end_date))
Xmat_weight = GBDT.createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples_weight, item_popularity_dict, True)
Xmat_weight = preprocessing.scale(Xmat_weight)
m, n = np.shape(Xmat_weight)
print("        %s matrix for generating weights (%d, %d)" % (getCurrentTime(), m, n))

Xmat_weight, Ymat_weight = shuffle(Xmat_weight, Ymat_weight, random_state=13)

X_train, X_train_lr, Y_train, Y_train_lr = train_test_split(Xmat_weight,
                                                            Ymat_weight,
                                                            test_size=0.5)

slide_windows_onehot = []
# 滑动窗口训练出的model分别对12-08 -- 12-17的数据生成叶节点， 与feature weight 矩阵合并后，生成一个大的特征矩阵，然后交给LR进行训练
X_train_features = Xmat_weight
for clf_model in slide_windows_models:
    # grd_enc = OneHotEncoder()
    # grd_enc.fit(clf_model.apply(Xmat_weight)[:, :, 0])
    # slide_windows_onehot.append(grd_enc)
    #X_train_lr_enc = grd_enc.transform(clf_model.apply(Xmat_weight)).toarray()

    X_train_lr_enc = clf_model.apply(Xmat_weight)[:, :, 0]
    X_train_features = np.column_stack((X_train_features, X_train_lr_enc))

m, n = X_train_features.shape
print("        %s X_train_features by split window models %d, %d " % (getCurrentTime(), m, n))

# EnsembleModel
logisticReg = LogisticRegression()
logisticReg.fit(X_train_features, Ymat_weight)

params = {'n_estimators': 500, 
          'max_depth': 4,
          'min_samples_split': 1,
          'learning_rate': 0.01, 
          #'loss': 'ls'
          'loss': 'deviance'
          }

# # 使用GBDT 算法
# gbdtRegressor = GradientBoostingRegressor(**params)
gbdtRegressor = GradientBoostingClassifier(**params)
gbdtRegressor.fit(X_train_features, Ymat_weight)


rfcls = RandomForestClassifier(n_estimators=500)
rfcls.fit(X_train_features, Ymat_weight)

# 采样 12-09 - 12-18 的数据生成特征矩阵
forecast_date = window_end_date + datetime.timedelta(1)
print("=====================================================================")
print("==============forecasting %s  ==============================" % forecast_date)
print("=====================================================================")

window_start_date = forecast_date - datetime.timedelta(slide_windows_days)

samples_forecast = takingSamplesForForecasting(window_start_date, forecast_date)

Xmat_forecast = GBDT.createTrainingSet(window_start_date, forecast_date, nag_per_pos, samples_forecast, item_popularity_dict, True)
Xmat_forecast = preprocessing.scale(Xmat_forecast)

m, n = np.shape(Xmat_forecast)
print("        %s matrix for generating forecast matrix (%d, %d)" % (getCurrentTime(), m, n))

# 滑动窗口训练出的model分别对12-09 -- 12-18的数据生成叶节点， 与feature weight 矩阵合并后，生成一个大的特征矩阵，然后交给LR进行训练
X_forecast_features = Xmat_forecast
for i, clf_model in enumerate(slide_windows_models):
    # X_forecast_enc = slide_windows_onehot[i].transform(clf_model.apply(Xmat_forecast)).toarray()

    X_forecast_enc =clf_model.apply(Xmat_forecast)[:, :, 0]    
    X_forecast_features = np.column_stack((X_forecast_features, X_forecast_enc))

m, n =  X_forecast_features.shape
print("        %s forecasting feature matrix %d, %d, sample forecasting %d " % (getCurrentTime(), m, n, len(samples_forecast)))

np.savetxt("%s\\..\log\\X_forecast_features.txt" % runningPath, X_forecast_features, fmt="%.4f", newline="\n")

findal_predicted_prob = logisticReg.predict_proba(X_forecast_features)
findal_predicted_prob += gbdtRegressor.predict_proba(X_forecast_features)
findal_predicted_prob += rfcls.predict_proba(X_forecast_features)

topK = 1000

if (forecast_date == datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()):

    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    if (len(samples_forecast) > 1000):
        topK = 1000
    else:
        topK = round(len(samples_forecast) / 2)

    print("%s probility of top1 = %.4f, top%d = %.4f" % 
          (getCurrentTime(), findal_predicted_prob[prob_desc[0], 1], topK, findal_predicted_prob[prob_desc[topK], 1] ))

    min_proba = 0.6

    file_idx = 0
    output_file_name = "%s\\..\\output\\forecast.GBDT.LR.%s.%d.csv" % (runningPath, datetime.date.today(), file_idx)
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = "%s\\..\\output\\forecast.GBDT.LR.%s.%d.csv" % (runningPath, datetime.date.today(), file_idx)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    outputFile.write("user_id,item_id\n")
    # for index in range(len(predicted_prob)):
    #     if (predicted_prob[index][1] >= min_proba):
    #         outputFile.write("%s,%s\n" % (samples_forecast[index][0], samples_forecast[index][1]))
    for index in range(topK):
        outputFile.write("%s,%s\n" %
            (samples_forecast[prob_desc[index]][0], samples_forecast[prob_desc[index]][1]))
        logging.info("prediction probability (%s,%s) =  %.4f" % 
            (samples_forecast[prob_desc[index]][0], samples_forecast[prob_desc[index]][1], findal_predicted_prob[prob_desc[index], 1]))

    outputFile.close()

else:
    verify_user_start = start_from + user_cnt
    verify_user_cnt = user_cnt
    verify.verifyPredictionEnsembleModel(window_start_date, forecast_date, nag_per_pos, verify_user_start, verify_user_cnt, topK,
                                         slide_windows_models, logisticReg, gbdtRegressor, rfcls)