#! python taobao_fresh_comp.py start_from= user_cnt= slide= topk= min_proba=0.5 start=2014-11-12 end=2014-12-10 output=
from common import *
# import userCF
# import itemCF
# import apriori
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
from sklearn.cross_validation import StratifiedKFold  
from sklearn import linear_model

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
def trainModelWithSlideWindow(window_start_date, final_end_date, slide_windows_days, start_from, user_cnt, 
                              n_estimators, nag_per_pos, max_depth, learning_rate, min_samples_split):

    window_end_date = window_start_date + datetime.timedelta(slide_windows_days)

    slide_feature_mat_dict = dict()
    feature_importances = None

    #滑动窗口 11-18 -- 12-17 得到特征的重要性， 并保留每个滑动窗口训练是使用的特征矩阵
    while (window_end_date < final_end_date):    
        params = {
            'window_start_date' : window_start_date, 
            'window_end_date' : window_end_date, 
            'nag_per_pos' : nag_per_pos, 
            'n_estimators' : n_estimators, 
            'max_depth' : max_depth, 
            'learning_rate' : learning_rate,
            'min_samples_split' : min_samples_split,
            'start_from' : start_from, 
            'user_cnt' : user_cnt 
        }
        Xmat, Ymat, slide_window_clf = GBDT.GBDT_slideWindows(**params)
        if (slide_window_clf == None):
            window_end_date += datetime.timedelta(1)
            window_start_date += datetime.timedelta(1)
            continue

        slide_feature_mat_dict[(window_start_date, window_end_date)] = (Xmat, Ymat, slide_window_clf)
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
from recommendation_same_day import *

# CRITICAL 50
# ERROR    40
# WARNING  30
# INFO     20
# DEBUG    10
# NOTSET   0


#loadDataAndSaveToRedis()
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
min_proba = float(sys.argv[5].split("=")[1])
start_date = sys.argv[6].split("=")[1]
end_date = sys.argv[7].split("=")[1]
does_output = int(sys.argv[8].split("=")[1])
print("start_from %d, user_cnt %d, slide window %d, topK %d, min prob %.2f, (%s, %s), output %d " % (start_from,
      user_cnt, slide_windows_days, topK, min_proba, start_date, end_date, does_output))

# start_from = 4000
# user_cnt = 0
# slide_windows_days = 4
# topK = 1000
# min_proba = 0.5

n_estimators = 50
max_depth = 4
learning_rate = 0.5
min_samples_split = 2

nag_per_pos = 10

log_file = '..\\log\\log.%d.%d.txt' % (start_from, user_cnt)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file,
                    filemode='w')


print("=====================================================================")
print("  training...  user from %d, count %d, slide window size %d (%s -- %s)" % 
      (start_from, user_cnt, slide_windows_days, start_date, end_date))
print("=====================================================================")

window_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
final_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
forecast_date = final_end_date + datetime.timedelta(1)

loadRecordsFromRedis(start_from, user_cnt)

# countUserCategoryOnSameDay(window_start_date, final_end_date)
# countUserBuyItemInDay()


params = {
    'window_start_date' : window_start_date, 
    'final_end_date' : final_end_date, 
    'slide_windows_days' : slide_windows_days, 
    'start_from' : start_from, 
    'user_cnt' : user_cnt,
    'nag_per_pos' : nag_per_pos,    
    'n_estimators' : n_estimators, 
    'max_depth' : max_depth, 
    'learning_rate' : learning_rate,
    'min_samples_split' : min_samples_split,
}

slide_feature_mat_dict, feature_importance = trainModelWithSlideWindow(**params)

logging.info("After slide window, feature_importance is : ")
useful_features_idx = []

for idx, importance in enumerate(feature_importance):
    if (feature_importance[idx] >= g_min_inportance):
        useful_features_idx.append(idx)
logging.info("useful_features_idx %s" % useful_features_idx)

print("=====================================================================")
print(" training with feature importance  (useful features: %d)  user from %d, count %d, slide window size %d " % 
      (len(useful_features_idx), start_from, user_cnt, slide_windows_days))
print("=====================================================================")

slide_windows_models = []
# 用满足 min importance 的特征重新训练模型, 训练好的模型保存在 slide_windows_models 中
for window_start_end_date, X_Y_mat_clf in slide_feature_mat_dict.items():
    window_start_date = window_start_end_date[0]
    window_end_date = window_start_end_date[1]
    X_mat = X_Y_mat_clf[0]
    Y_mat = X_Y_mat_clf[1]
    clf = X_Y_mat_clf[2]

    X_useful_mat = X_mat[:, useful_features_idx]

    params = {
        'n_estimators': n_estimators, 
        'max_depth': max_depth,
        'min_samples_split': 2,
        'learning_rate': learning_rate, 
        #'loss': 'ls'
        'loss': 'deviance'
    }

    # clf = GradientBoostingClassifier(**params)

    # print("%s Using useful features for slide windows (%s, %s)\r" % (getCurrentTime(), window_start_date, window_end_date), end ="")
    # clf.fit(X_useful_mat, Y_mat)

    slide_windows_models.append((X_useful_mat, clf, window_start_end_date))

window_end_date = final_end_date
window_start_date = window_end_date - datetime.timedelta(slide_windows_days)


# 根据滑动窗口的结果，使用重要性 > 0 的特征从 12-08 -- 12-17 生成特征矩阵以及12-18 的购买记录，交给滑动窗口
# 训练出的model，生成叶节点，传给 LR 再进行训练， 最后使用 LR 从 12-09 -- 12-18 生成特征矩阵进行预测
print("=====================================================================")
print("==============  generating weights %s - %s, user from %d, count %d, slide window size %d" % 
      (window_start_date, window_end_date, start_from, user_cnt, slide_windows_days))
print("=====================================================================")
samples_weight, Ymat_weight = takeSamples(window_start_date, window_end_date, nag_per_pos, start_from, user_cnt)

np.savetxt("%s\\..\log\\samples_weight(%s,%s).txt" % (runningPath, window_start_date, window_end_date), list(samples_weight), fmt="%s", newline="\n")
np.savetxt("%s\\..\log\\Ymat_(%s,%s)(%d,%d).txt" % (runningPath, window_start_date, window_end_date, start_from, user_cnt), Ymat_weight, fmt="%d", newline="\n")

params = {'window_start_date' : window_start_date, 
         'window_end_date' : window_end_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_weight, 
         }

Xmat_weight = createFeatureMatrix(**params)

m, n = np.shape(Xmat_weight)
print("        %s matrix for generating weights matrix (%s, %s) (%d, %d)" % (getCurrentTime(), window_start_date, window_end_date, m, n))

window_start_date = forecast_date - datetime.timedelta(slide_windows_days)

if (does_output != 1):  # 如果是需要验证, 则将用于训练的用户清空，读取新的一批用户数据
    verify_user_start = start_from + user_cnt
    print("%s reloading verifying users..." % (getCurrentTime()))
    verify_users = user_cnt * 2
    logging.info("reloading verifying users %d ..." % (verify_users))
    loadRecordsFromRedis(verify_user_start, verify_users)


print("%s taking samples for forecasting %s " % (getCurrentTime(), forecast_date))
during_forecasting = False
if (forecast_date == ONLINE_FORECAST_DATE):
    during_forecasting = True

params = {'window_start_date' : window_start_date,
          'window_end_date' : forecast_date,
          'user_records' : g_user_behavior_patten,
          'during_forecasting' : during_forecasting
}
samples_pattern = takeSamplesByUserBehavior(**params)
# samples_pattern = set()

if (forecast_date != ONLINE_FORECAST_DATE):
    params = {'window_start_date' : window_start_date,
              'window_end_date' : forecast_date,
              'user_records' : g_user_buy_transection,
              'during_forecasting' : during_forecasting
    }
    samples_buy = takeSamplesByUserBehavior(**params)
    print("%s samples count from buy %d / %d pattern" % (getCurrentTime(), len(samples_buy), len(samples_pattern)))

    samples_forecast = samples_pattern.union(samples_buy)    
    samples_forecast = list(samples_forecast)
else:
    samples_forecast = list(samples_pattern)    

print("%s samples forecast %d" % (getCurrentTime(), len(samples_forecast)))

params = {'window_start_date' : window_start_date, 
         'window_end_date' : forecast_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_forecast, 
         }

Xmat_forecast = createFeatureMatrix(**params)

m, n = np.shape(Xmat_forecast)
print("        %s matrix for generating forecast matrix (%s, %s), (%d, %d)" % (getCurrentTime(), window_start_date, forecast_date, m, n))

onehot_enc = getOnehotEncoder(slide_windows_models, Xmat_weight, Xmat_forecast, n_estimators)

# 滑动窗口训练出的model分别对12-08 -- 12-17的数据生成叶节点， 与feature weight 矩阵合并后，生成一个大的特征矩阵，然后交给LR进行训练
X_train_features = Xmat_weight

for i, X_useful_mat_clf_model in enumerate(slide_windows_models):
    X_useful_mat = X_useful_mat_clf_model[0]
    clf_model = X_useful_mat_clf_model[1]
    slide_windows_start = X_useful_mat_clf_model[2][0]
    slide_windows_end = X_useful_mat_clf_model[2][1]
  
    #  GBDT 得到叶节点
    X_train_lr_enc = clf_model.apply(Xmat_weight)[:, :, 0]
    # X_train_lr_enc = onehot_enc.transform(X_train_lr_enc).toarray()
    if (X_train_features is None):
        X_train_features = X_train_lr_enc
    else:
        X_train_features = np.column_stack((X_train_features, X_train_lr_enc))

m, n = X_train_features.shape
print("        %s X_train_features by slide window models %d, %d " % (getCurrentTime(), m, n))

# EnsembleModel
# 逻辑回归算法
print("        %s running LR..." % (getCurrentTime()))
logisticReg = LogisticRegression()
logisticReg.fit(X_train_features, Ymat_weight)

# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(X_train_features, Ymat_weight)

# logisticReg = modelBlending_iterate(logisticReg, X_train_features, min_proba)

# 滑动窗口训练出的model分别对特征矩阵的数据生成叶节点， 与feature weight 矩阵合并后，生成一个大的特征矩阵，然后交给LR进行训练
X_forecast_features = Xmat_forecast

for X_useful_mat_clf_model in slide_windows_models:
    X_useful_mat = X_useful_mat_clf_model[0]
    clf_model = X_useful_mat_clf_model[1]
    slide_windows_start = X_useful_mat_clf_model[2][0]
    slide_windows_end = X_useful_mat_clf_model[2][1]

    X_forecast_enc =clf_model.apply(Xmat_forecast)[:, :, 0]    
    # X_forecast_enc = onehot_enc.transform(X_forecast_enc).toarray()
    if (X_forecast_features is None):
        X_forecast_features = X_forecast_enc
    else:
        X_forecast_features = np.column_stack((X_forecast_features, X_forecast_enc))
    # blend_forecast[:, i] = clf_model.predict_proba(Xmat_forecast)[:, 1]

m, n =  X_forecast_features.shape
print("        %s forecasting feature matrix %d, %d, samples for forecasting %d " % (getCurrentTime(), m, n, len(samples_forecast)))

# 用逻辑回归预测
findal_predicted_prob = logisticReg.predict_proba(X_forecast_features)

# findal_predicted_prob = clf.predict(X_forecast_features)
# Y_prob = np.zeros((X_forecast_features.shape[0], 2))
# for i in range(len(findal_predicted_prob)):
#     Y_prob[i, 1] = findal_predicted_prob[i]
# findal_predicted_prob = Y_prob

if (does_output == 1):
    print("=====================================================================")
    print(" forecasting %s  user from %d, count %d, slide window size %d" % 
         (forecast_date, start_from, user_cnt, slide_windows_days))
    print("=====================================================================")

    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    file_idx = 0
    output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    for i, prob_index in enumerate(prob_desc):
        prob = findal_predicted_prob[prob_index, 1]
        if (prob < min_proba):
            print("        %s findal_predicted_prob[%d] %.4f < min_proba %.4f, breaking..." % (getCurrentTime(), i, prob, min_proba))
            break

        outputFile.write("%s,%s,%.4f\n" %
            (samples_forecast[prob_index][0], samples_forecast[prob_index][1], findal_predicted_prob[prob_index, 1] ))

    outputFile.close()

else:
    verify_user_start = start_from + user_cnt
    verify_user_cnt = user_cnt

    print("=====================================================================")
    print("== verifyPredictionEnsembleModel (%s, %s) esitmators %d, max depth %d, learning rate %.2f, min split %d" %
          (window_start_date, forecast_date, n_estimators, max_depth, learning_rate, min_samples_split))
    print("=====================================================================")

    if (len(samples_forecast) < topK):
        topK = round(len(samples_forecast) / 2)

    params = {'forecast_date': forecast_date, 
              'findal_predicted_prob' : findal_predicted_prob,
              'verify_samples' : samples_forecast,
              'topK' : topK, 
              'min_proba' : min_proba, 
             }

    # verify.verifyPredictionEnsembleModelWithRule(**params)
    verify.verifyPredictionEnsembleModel(**params)
