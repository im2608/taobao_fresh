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
from feature_selection import *


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



#滑动窗口 11-18 -- 12-17 得到并累加特征的重要性， 并保留每个滑动窗口训练时使用的特征矩阵
def trainModelWithSlideWindow(window_start_date, final_end_date, slide_windows_days, cal_feature_importance):
    window_end_date = window_start_date + datetime.timedelta(slide_windows_days)

    slide_feature_mat_dict = dict()
    feature_importances = None

    #滑动窗口 11-18 -- 12-17 得到特征的重要性， 并保留每个滑动窗口训练是使用的特征矩阵
    while (window_end_date < final_end_date):    
        Xmat, Ymat, slide_window_clf = GBDT.GBDT_slideWindows(window_start_date, window_end_date, cal_feature_importance, n_estimators)
        if (slide_window_clf == None):
            window_end_date += datetime.timedelta(1)
            window_start_date += datetime.timedelta(1)
            continue

        slide_feature_mat_dict[(window_start_date, window_end_date)] = (Xmat, Ymat)
        if (feature_importances is None):
            feature_importances = slide_window_clf.feature_importances_
        else:
            feature_importances += slide_window_clf.feature_importances_

        window_end_date += datetime.timedelta(1)
        window_start_date += datetime.timedelta(1)

    return slide_feature_mat_dict, feature_importances


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

import LR
import RF
import GBDT
from taking_sample import *
import verify
from feature_selection import *


# CRITICAL 50
# ERROR    40
# WARNING  30
# INFO     20
# DEBUG    10
# NOTSET   0


#apriori.loadDataAndSaveToRedis(need_verify)
#loadTrainCategoryItemAndSaveToRedis()



print("--------------------------------------------------")
print("--------------- Starting... ----------------------")
print("--------------------------------------------------")


loadTrainCategoryItemFromRedis()
loadTestSet()


start_from = int(sys.argv[1].split("=")[1])
user_cnt = int(sys.argv[2].split("=")[1])
slide_windows_days = int(sys.argv[3].split("=")[1])
topK = int(sys.argv[4].split("=")[1])
run_for_test = int(sys.argv[5].split("=")[1])
min_proba = float(sys.argv[6].split("=")[1])

# print("start_from %d, user_cnt %d" % (start_from, user_cnt))

# start_from = 4000
# user_cnt = 0
# slide_windows_days = 4
# topK = 1000
# min_proba = 0.5

n_estimators = 300


log_file = '..\\log\\log.%d.%d.txt' % (start_from, user_cnt)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file,
                    filemode='w')

print("=====================================================================")
print("=========================  training...   ============================")
print("======== user from %d, count %d, slide window size %d ========" % (start_from, user_cnt, slide_windows_days))
print("=====================================================================")



loadRecordsFromRedis(start_from, user_cnt)

# daysBetween1stBehaviorToBuy()


if (run_for_test == 1):
    window_start_date = datetime.datetime.strptime("2014-11-18", "%Y-%m-%d").date()
    final_end_date = datetime.datetime.strptime("2014-12-17", "%Y-%m-%d").date()
else:
    window_start_date = datetime.datetime.strptime("2014-11-18", "%Y-%m-%d").date()
    final_end_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()

# 若 cal_feature_importance = True， 则累加特征的重要性， 忽略 final_feature_importance
# 若为 False，则 final_feature_importance 为累加好的特征重要性， 根据特征的重要性在滑动窗口内重新训练model
slide_feature_mat_dict, feature_importance = trainModelWithSlideWindow(window_start_date, final_end_date, slide_windows_days, True)

logging.info("After split window, feature_importance is : ")
useful_features_idx = []
for feature_name, idx in g_feature_info.items():
    logging.info("%s : %f" % (feature_name, feature_importance[idx]))
    if (feature_importance[idx] >= g_min_inportance):
        useful_features_idx.append(idx)

logging.info("useful_features_idx %s" % useful_features_idx)

print("=====================================================================")
print(" training with feature importance slide days (useful features: %d) ==" % len(useful_features_idx))
print("=====================================================================")

slide_windows_models = []
# 用满足 min importance 的特征重新训练模型, 训练好的模型保存在 slide_windows_models 中
for window_start_end_date, X_Y_mat in slide_feature_mat_dict.items():
    window_start_date = window_start_end_date[0]
    window_end_date = window_start_end_date[1]
    X_mat = X_Y_mat[0]
    Y_mat = X_Y_mat[1]

    X_useful_mat = X_mat[:, useful_features_idx]

    params = {'n_estimators': n_estimators, 
          'max_depth': 4,
          'min_samples_split': 1,
          'learning_rate': 0.01, 
          #'loss': 'ls'
          'loss': 'deviance'
          }

    #clf = GradientBoostingRegressor(**params)
    clf = GradientBoostingClassifier(**params)

    print("%s Using useful features for split windows (%s, %s)\r" % (getCurrentTime(), window_start_date, window_end_date), end ="")
    clf.fit(X_useful_mat, Y_mat)

    slide_windows_models.append(clf)

nag_per_pos = 10

window_end_date = final_end_date
window_start_date = window_end_date - datetime.timedelta(slide_windows_days)


# 根据滑动窗口的结果，使用重要性 > 0 的特征从 12-08 -- 12-17 生成特征矩阵以及12-18 的购买记录，交给滑动窗口
# 训练出的model，生成叶节点，传给 LR 再进行训练， 最后使用 LR 从 12-09 -- 12-18 生成特征矩阵进行预测
print("=====================================================================")
print("==============  generating weights %s - %s (%d)=======" % (window_start_date, window_end_date, slide_windows_days))
print("=====================================================================")


# samples_weight, Ymat_weight = takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos)
samples_weight, Ymat_weight = takeSamples(window_start_date, window_end_date, False)

# 使用重要性 > 0 的特征从 12-08 -- 12-17 生成特征矩阵
print("        %s forecasting, reading feature matrix from %s -- %s" % (getCurrentTime(), window_start_date, window_end_date))

params = {'window_start_date' : window_start_date, 
         'window_end_date' : window_end_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_weight, 
         'cal_feature_importance' : False}


Xmat_weight = GBDT.createTrainingSet(**params)
Xmat_weight = preprocessing.scale(Xmat_weight)
Xmat_weight = Xmat_weight[:, useful_features_idx]

m, n = np.shape(Xmat_weight)
print("        %s matrix for generating weights (%d, %d)" % (getCurrentTime(), m, n))

# Xmat_weight, Ymat_weight = shuffle(Xmat_weight, Ymat_weight, random_state=13)

# X_train, X_train_lr, Y_train, Y_train_lr = train_test_split(Xmat_weight, Ymat_weight, test_size=0.5)

slide_windows_onehot = []
# 滑动窗口训练出的model分别对12-08 -- 12-17的数据生成叶节点， 与feature weight 矩阵合并后，生成一个大的特征矩阵，然后交给LR进行训练
X_train_features = Xmat_weight

models_cnt = len(slide_windows_models)
for model_idx, clf_model in enumerate(slide_windows_models):
    # grd_enc = OneHotEncoder()
    # grd_enc.fit(clf_model.apply(Xmat_weight)[:, :, 0])
    # slide_windows_onehot.append(grd_enc)
    #X_train_lr_enc = grd_enc.transform(clf_model.apply(Xmat_weight)).toarray()
    X_train_lr_enc = clf_model.apply(Xmat_weight)[:, :, 0]
    X_train_features = np.column_stack((X_train_features, X_train_lr_enc))

m, n = X_train_features.shape
print("        %s X_train_features by split window models %d, %d " % (getCurrentTime(), m, n))

# EnsembleModel
# GBDT 算法
params = {'n_estimators': n_estimators, 
          'max_depth': 4,
          'min_samples_split': 1,
          'learning_rate': 0.01, 
          #'loss': 'ls'
          'loss': 'deviance'
          }


# gbdtRegressor = GradientBoostingRegressor(**params)
print("        %s running GBDT..." % (getCurrentTime()))
gbdtRegressor = GradientBoostingClassifier(**params)
gbdtRegressor.fit(X_train_features, Ymat_weight)


# 逻辑回归算法
print("        %s running LR..." % (getCurrentTime()))
logisticReg = LogisticRegression()
logisticReg.fit(X_train_features, Ymat_weight)


# 随机森林算法
print("        %s running RF..." % (getCurrentTime()))
rfcls = RandomForestClassifier(n_estimators=n_estimators)
rfcls.fit(X_train_features, Ymat_weight)

# 采样 12-09 - 12-18 的数据生成特征矩阵
forecast_date = window_end_date + datetime.timedelta(1)
print("=====================================================================")
print("==============  forecasting %s  ============================" % forecast_date)
print("=====================================================================")

window_start_date = forecast_date - datetime.timedelta(slide_windows_days)

# samples_forecast, _ = takingSamplesForForecasting(window_start_date, forecast_date, True)
samples_forecast, _ = takeSamples(window_start_date, forecast_date, True)

params = {'window_start_date' : window_start_date, 
         'window_end_date' : forecast_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_forecast, 
         'cal_feature_importance' : False}
Xmat_forecast = GBDT.createTrainingSet(**params)
Xmat_forecast = preprocessing.scale(Xmat_forecast)
Xmat_forecast = Xmat_forecast[:, useful_features_idx]

m, n = np.shape(Xmat_forecast)
print("        %s matrix for generating forecast matrix (%d, %d)" % (getCurrentTime(), m, n))

# 滑动窗口训练出的model分别对12-09 -- 12-18的数据生成叶节点， 与feature weight 矩阵合并后，生成一个大的特征矩阵，然后交给LR进行训练
X_forecast_features = Xmat_forecast
for model_idx, clf_model in enumerate(slide_windows_models):
    # X_forecast_enc = slide_windows_onehot[i].transform(clf_model.apply(Xmat_forecast)).toarray()
    X_forecast_enc =clf_model.apply(Xmat_forecast)[:, :, 0]    
    X_forecast_features = np.column_stack((X_forecast_features, X_forecast_enc))

m, n =  X_forecast_features.shape
print("        %s forecasting feature matrix %d, %d, sample forecasting %d " % (getCurrentTime(), m, n, len(samples_forecast)))

np.savetxt("%s\\..\log\\X_forecast_features.txt" % runningPath, X_forecast_features, fmt="%.4f", newline="\n")

#  用 RF 过滤 samples
findal_predicted_prob = rfcls.predict_proba(X_forecast_features)
filtered_sampls, X_filtered_features = filterSamplesByProbility(samples_forecast, X_forecast_features, findal_predicted_prob, min_proba)
m, n = np.shape(X_filtered_features)
print("%s After filtering by Random Forest, shape(X_filtered_features) = (%d, %d), samples = %d" % 
     (getCurrentTime(), m, n, len(filtered_sampls)))
if (len(filtered_sampls) == 0):
    print("        %s No samples after filtering by Random Forest" % (getCurrentTime()))
    filtered_sampls = samples_forecast
    X_filtered_features = X_forecast_features

# 用gbdt过滤 samples
findal_predicted_prob = gbdtRegressor.predict_proba(X_filtered_features)
filtered_sampls_gbdt, X_filtered_features_gbdt = filterSamplesByProbility(filtered_sampls, X_filtered_features, findal_predicted_prob, min_proba)
m, n = np.shape(X_filtered_features_gbdt)
print("%s After filtering by GBDT, shape(X_filtered_features_gbdt) = (%d, %d), samples = %d" % 
      (getCurrentTime(), m, n, len(filtered_sampls_gbdt)))
if (len(filtered_sampls_gbdt) == 0):
    print("        %s No samples after filtering by GBDT")
    filtered_sampls_gbdt = filtered_sampls
    X_filtered_features_gbdt = X_filtered_features

samples_forecast = filtered_sampls_gbdt

# 用逻辑回归预测
findal_predicted_prob = logisticReg.predict_proba(X_filtered_features_gbdt)

if (forecast_date == datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()):

    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    if (len(samples_forecast) < topK):
        topK = round(len(samples_forecast) / 2)

    print("%s probility of top1 = %.4f, top%d = %.4f" % 
          (getCurrentTime(), findal_predicted_prob[prob_desc[0], 1], topK, findal_predicted_prob[prob_desc[topK], 1] ))

    file_idx = 0
    output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    # outputFile.write("user_id,item_id\n")    
    for index in range(topK):
        prob = findal_predicted_prob[prob_desc[index], 1]
        if (prob < min_proba):
            print("        %s findal_predicted_prob[%d] %.4f < min_proba %.4f, breaking..." % (getCurrentTime(), index, prob, min_proba))
            break

        outputFile.write("%s,%s,%.4f\n" %
            (samples_forecast[prob_desc[index]][0], samples_forecast[prob_desc[index]][1], findal_predicted_prob[prob_desc[index], 1] ))
        logging.info("prediction probability (%s,%s) =  %.4f" % 
            (samples_forecast[prob_desc[index]][0], samples_forecast[prob_desc[index]][1], findal_predicted_prob[prob_desc[index], 1]))

    outputFile.close()

else:
    verify_user_start = start_from + user_cnt
    verify_user_cnt = user_cnt

    params = {'window_start_date' : window_start_date,
              'forecast_date': forecast_date, 
              'nag_per_pos' : nag_per_pos, 
              'verify_user_start' : verify_user_start, 
              'verify_user_cnt' : verify_user_cnt, 
              'topK' : topK, 
              'min_proba' : min_proba, 
              'slide_windows_models' : slide_windows_models, 
              'logisticReg' : logisticReg, 
              'gbdtRegressor' : gbdtRegressor, 
              'rfcls' : rfcls, 
              'useful_features_idx' : useful_features_idx}

    verify.verifyPredictionEnsembleModel(**params)
