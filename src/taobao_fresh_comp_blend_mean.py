#! python taobao_fresh_comp_mean.py start_from= user_cnt= slide= topk= test=1 min_proba=0.5 start=2014-11-12 end=2014-12-10
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


n_estimators = 100
max_depth = 7
learning_rate = 0.01
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

    X_useful_mat = X_mat#[:, useful_features_idx]

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
samples_weight, Ymat_weight = takeSamples(window_start_date, window_end_date, nag_per_pos, True, start_from, user_cnt)

params = {'window_start_date' : window_start_date, 
         'window_end_date' : window_end_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_weight, 
         }

Xmat_weight = createFeatureMatrix(**params)
if (len(samples_weight) < topK):
    topK = round(len(samples_weight) / 2)

Xmat_weight, Ymat_weight, samples_weight = shuffle(Xmat_weight, Ymat_weight, samples_weight, random_state=13)

m, n = np.shape(Xmat_weight)
print("        %s matrix for generating weights matrix (%s, %s) (%d, %d)" % (getCurrentTime(), window_start_date, window_end_date, m, n))

slide_windows_precession = dict()

# 滑动窗口训练出的model分别使用12-08 -- 12-17的数据对12-18进行预测，得到预测的准确率
for i, X_useful_mat_clf_model in enumerate(slide_windows_models):
    X_useful_mat = X_useful_mat_clf_model[0]
    clf_model = X_useful_mat_clf_model[1]
    slide_windows_start = X_useful_mat_clf_model[2][0]
    slide_windows_end = X_useful_mat_clf_model[2][1]

    predicted_prob = clf_model.predict_proba(Xmat_weight)

    params = {'forecast_date': window_end_date, 
              'findal_predicted_prob' : predicted_prob,
              'verify_samples' : samples_weight,
              'topK' : topK, 
              'min_proba' : min_proba, 
             }
    p, r, f1 = verify.verifyPredictionEnsembleModel(**params)
    print("%s model for slide window (%s, %s), precission is %.4f" % (getCurrentTime(), slide_windows_start, slide_windows_end, p))

    slide_windows_precession[(slide_windows_start, slide_windows_end)] = p

if (does_output == 0):
    verify_user_start = start_from + user_cnt
    verify_users = user_cnt * 2
    print("%s reloading verifying users... %d" % (getCurrentTime(), verify_users))
    loadRecordsFromRedis(verify_user_start, verify_users, None)

#取得用于预测的特征矩阵
window_start_date = forecast_date - datetime.timedelta(slide_windows_days)
samples_forecast, _ = takeSamples(window_start_date, forecast_date, nag_per_pos, True, start_from, user_cnt)

params = {'window_start_date' : window_start_date, 
         'window_end_date' : forecast_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_forecast, 
         }

Xmat_forecast = createFeatureMatrix(**params)

m, n = np.shape(Xmat_forecast)
print("        %s matrix for generating forecast matrix (%s, %s), (%d, %d)" % (getCurrentTime(), window_start_date, forecast_date, m, n))

findal_predicted_prob = np.zeros((Xmat_forecast.shape[0], 2))

# 滑动窗口训练出的model分别对特征矩阵进行预测，预测出的概率 * model的准确率作为该model的输出, 将所有modle的输出累加作为最终的输出，取 topK
for X_useful_mat_clf_model in slide_windows_models:
    X_useful_mat = X_useful_mat_clf_model[0]
    clf_model = X_useful_mat_clf_model[1]
    slide_windows_start = X_useful_mat_clf_model[2][0]
    slide_windows_end = X_useful_mat_clf_model[2][1]

    predicted_prob = clf_model.predict_proba(Xmat_forecast)
    findal_predicted_prob += predicted_prob * slide_windows_precession[(slide_windows_start, slide_windows_end)]

findal_predicted_prob[:, 1] = (findal_predicted_prob[:, 1] - findal_predicted_prob[:, 1].min())/(findal_predicted_prob[:, 1].max() - findal_predicted_prob[:, 1].min())

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
    print("=====================================================================")
    print("== verifyPredictionEnsembleModel (%s, %s) esitmators %d, max depth %d, learning rate %.2f, min split %d" %
          (window_start_date, forecast_date, n_estimators, max_depth, learning_rate, min_samples_split))
    print("=====================================================================")

    n_folds = 5
    skf = list(StratifiedKFold(samples_forecast, n_folds))  

    for i, (train, test) in enumerate(skf):  
        for X_useful_mat_clf_model in slide_windows_models:
                X_useful_mat = X_useful_mat_clf_model[0]
                clf_model = X_useful_mat_clf_model[1]

    if (len(samples_forecast) < topK):
        topK = round(len(samples_forecast) / 2)

    params = {'forecast_date': forecast_date, 
              'findal_predicted_prob' : findal_predicted_prob,
              'verify_samples' : samples_forecast,
              'topK' : topK, 
              'min_proba' : min_proba, 
             }

    verify.verifyPredictionEnsembleModel(**params)
