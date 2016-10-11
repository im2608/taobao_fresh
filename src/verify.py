from common import *
import apriori
import GBDT
from taking_sample import *
from sklearn import preprocessing
import logging
from utils import *
from sklearn.cross_validation import StratifiedKFold  
from sklearn.linear_model import LogisticRegression

buy_records_mysql = dict()
buy_records_python = dict()


def verifyBuyRecords():
    file_user_buy_record_mysql = open("%s\\..\\input\\buy_record_mysql.csv" % runningPath, encoding="utf-8", mode='r')
    user_buy_records_csv = csv.reader(file_user_buy_record_mysql)

    file_buy_records_python = open("%s\\..\\input\\buy_record_python.csv" % runningPath, encoding="utf-8", mode='r')
    buy_records_python_csv = csv.reader(file_buy_records_python)

    index = 0

    for aline in user_buy_records_csv:
        if (index == 0):
            index += 1
            continue
        user_id = aline[0]
        buy_records_cnt = int(aline[1])

        buy_records_mysql[user_id] = buy_records_cnt;

    index = 0
    for aline in buy_records_python_csv:
        if (index == 0):
            index += 1
            continue

        user_id = aline[0]
        buy_records_cnt = int(aline[1])

        buy_records_python[user_id] = buy_records_cnt;

    logging.info("users %d from mysql" % len(buy_records_mysql))
    logging.info("users %d from python" % len(buy_records_python))

    for user_id, buy_records_cnt in buy_records_mysql.items():
        if (user_id not in buy_records_python):
            logging.info("Error: %s not in Buy-Records-Python" % user_id)
            continue

        if (buy_records_cnt != buy_records_python[user_id]):
            logging.info("Error: user %s Mysql (%d) != Python (%d)" % (user_id, buy_records_cnt, buy_records_python[user_id]))

    return 0


# file_idx = 37
# data_file = "%s\\..\\input\\splitedInput\\datafile.%03d" % (runningPath, file_idx)

# #apriori.loadData(True)
# apriori.loadRecordsFromRedis()
# apriori.saveRecordstoRedis()
# #verifyBuyRecords()


def calcuatingF1(forecast_date, predicted_user_item, actual_user_item):
    logging.info("calcuatingF1 forecast_date %s" % forecast_date)

    actual_count = len(actual_user_item)
    predicted_count = len(predicted_user_item)

    hit_count = 0
    user_hit_list = []

    for index, user_item_prob in enumerate(predicted_user_item):
        user_item = user_item_prob[0] 
        prob = user_item_prob[1]
        # index = user_item_prob[1]
        logging.info("predicted (%s, %s), %d" % (user_item[0], user_item[1], index))
        if (user_item in actual_user_item):
            hit_count += 1
            user_hit_list.append((prob, index, hit_count))

    logging.info("%s acutal buy: " % forecast_date)
    for user_item in actual_user_item:
        logging.info("acutal buy %s, %s" % (user_item[0], user_item[1]))

    print("%s forecasting date %s, positive count %d, predicted count %d, hit count %d" %\
          (getCurrentTime(), forecast_date, actual_count, predicted_count, hit_count))

    if (predicted_count != 0):
        p = hit_count / predicted_count
    else:
        p = 0

    if (actual_count != 0):
        r = hit_count / actual_count
    else:
        r = 0

    if (p != 0 or r != 0):        
        f1 = (2 * p * r) / (p + r)
    else:
        f1 = 0

    print("%s precission: %.4f, recall %.4f, F1 %.4f" % (getCurrentTime(), p, r, f1))

    return p, r, f1

def verifyPredictionEnsembleModel(forecast_date, findal_predicted_prob, verify_samples, topK, min_proba, actual_user_item):
    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    print("%s probility of top1 = %.4f, top%d = %.4f" % 
          (getCurrentTime(), findal_predicted_prob[prob_desc[0], 1], topK, findal_predicted_prob[prob_desc[topK], 1] ))

    predicted_user_item = []

    for index in range(topK):
        if (findal_predicted_prob[prob_desc[index], 1] < min_proba):
            print("        %s probability %.4f < min_proba %.4f, breaking..." %
                  (getCurrentTime(), findal_predicted_prob[prob_desc[index], 1], min_proba))
            break
        user_item = verify_samples[prob_desc[index]]
        predicted_user_item.append((user_item, findal_predicted_prob[prob_desc[index], 1]))

    return calcuatingF1(forecast_date, predicted_user_item, actual_user_item)

def verifyPredictionEnsembleModelWithRule(forecast_date, findal_predicted_prob, verify_samples, topK, min_proba):
    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    print("%s probility of top1 = %.4f, top%d = %.4f" % 
          (getCurrentTime(), findal_predicted_prob[prob_desc[0], 1], topK, findal_predicted_prob[prob_desc[topK], 1] ))

    predicted_user_item = []

    for index in range(topK):
        if (findal_predicted_prob[prob_desc[index], 1] < min_proba):
            print("        %s probability %.4f < min_proba %.4f, breaking..." %
                  (getCurrentTime(), findal_predicted_prob[prob_desc[index], 1], min_proba))
            break
        user_item = verify_samples[prob_desc[index]]
        predicted_user_item.append((user_item, findal_predicted_prob[prob_desc[index], 1]))

    prediction_rule = rule_12_18_cart(forecast_date)

    predicted_user_item = list((set(predicted_user_item).union(set(prediction_rule))))

    actual_user_item = takingPositiveSamplesOnDate(forecast_date, during_verifying=True)

    return calcuatingF1(forecast_date, predicted_user_item, actual_user_item)


def getModelByCV(Xmat, Ymat, X_forecast_features, verify_samples, min_proba, topK, folds, verify_date):
    print("%s getModelByCV, verify date %s" % (getCurrentTime(), verify_date))

    logisticReg = LogisticRegression()
    actual_user_item = takingPositiveSamplesOnDate(verify_date, during_verifying=True)

    skf = list(StratifiedKFold(Ymat[:, 0], folds))

    best_f1 = 0.0
    best_model = logisticReg

    for i, (train, test) in enumerate(skf): 
        print("%s        verifying fold %d ..." % (getCurrentTime(), i))
        logisticReg.fit(Xmat[train], Ymat[train, 0])

        findal_predicted_prob = logisticReg.predict_proba(X_forecast_features)

        params = {'forecast_date': verify_date, 
                  'findal_predicted_prob' : findal_predicted_prob,
                  'verify_samples' : verify_samples,
                  'topK' : topK, 
                  'min_proba' : min_proba, 
                  'actual_user_item' : actual_user_item,
                 }

        # verify.verifyPredictionEnsembleModelWithRule(**params)
        p, r, f1 = verify.verifyPredictionEnsembleModel(**params)
        if (best_f1 < f1):
            best_f1 = f1
            best_model = logisticReg

    print("        %s getModelByCV, best f1 is %.4f" % (getCurrentTime(), best_f1))
    return best_model

# r = f*p/(2*p-f)