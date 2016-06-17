from common import *
import apriori
import GBDT
from taking_sample import *
from sklearn import preprocessing


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
    actual_count = len(actual_user_item)
    predicted_count = len(predicted_user_item)

    hit_count = 0
    user_hit_list = set()

    for user_item in predicted_user_item:
        logging.info("predicted %s, %s" % (user_item[0], user_item[1]))
        if (user_item in actual_user_item):
            hit_count += 1

    for user_item in predicted_user_item:
        for user_item2 in actual_user_item:
            if (user_item[0] == user_item2[0]):
                user_hit_list.add(user_item[0])

    print("hit user: %s" % user_hit_list)

    for user_item in actual_user_item:
        logging.info("acutal buy %s, %s" % (user_item[0], user_item[1]))

    print("forecasting date %s, positive count %d, predicted count %d, hit count %d" %\
          (forecast_date, actual_count, predicted_count, hit_count))

    if (predicted_count != 0):
        p = hit_count / predicted_count
    else:
        p = 0

    if (actual_count != 0):
        r = hit_count / actual_count
    else:
        r = 0

    if (p != 0 or r != 0):        
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0

    print("precission: %.4f, recall %.4f, F1 %.4f" % (p, r, f1))

    return


def verifyPrediction(window_start_date, forecast_date, min_proba, nag_per_pos, verify_user_start, verify_user_cnt, clf, grd_enc, logisticReg):
    print("=====================================================================")
    print("=========================  verifying...  ============================")
    print("=====================================================================")

    g_user_buy_transection.clear()

    print("%s reloading verifying users..." % (getCurrentTime()))
    loadRecordsFromRedis(verify_user_start, verify_user_cnt)

    item_popularity_dict = calculateItemPopularity(window_start_date, forecast_date)

    verify_samples, _ = takingSamplesForForecasting(window_start_date, forecast_date)
    
    print("%s creating verifying feature matrix..." % (getCurrentTime()))
    Xmat_verify = GBDT.createTrainingSet(window_start_date, forecast_date, nag_per_pos, verify_samples, item_popularity_dict, False)
    Xmat_verify = preprocessing.scale(Xmat_verify)

    X_leaves_verify = grd_enc.transform(clf.apply(Xmat_verify))

    predicted_prob = logisticReg.predict_proba(X_leaves_verify)

    predicted_user_item = []
    for index in range(len(predicted_prob)):
        if (predicted_prob[index][1] >= min_proba):
            predicted_user_item.append(verify_samples[index])

    actual_user_item = takingPositiveSamples(forecast_date)

    calcuatingF1(forecast_date, predicted_user_item, actual_user_item)
    return


def verifyPredictionEnsembleModel(window_start_date, forecast_date, nag_per_pos, verify_user_start, verify_user_cnt, topK, min_proba,
                                  slide_windows_models, logisticReg, gbdtRegressor, rfcls, final_feature_importance):
    print("=====================================================================")
    print("============verifyPredictionEnsembleModel %s, %s ===============" % (window_start_date, forecast_date))
    print("=====================================================================")

    g_user_buy_transection.clear()

    print("%s reloading verifying users..." % (getCurrentTime()))
    loadRecordsFromRedis(verify_user_start, verify_user_cnt)

    item_popularity_dict = calculateItemPopularity(window_start_date, forecast_date)

    verify_samples, _ = takingSamplesForForecasting(window_start_date, forecast_date)

    print("%s creating verifying feature matrix..." % (getCurrentTime()))
    Xmat_verify = GBDT.createTrainingSet(window_start_date, forecast_date, nag_per_pos, verify_samples, item_popularity_dict, False, final_feature_importance)
    Xmat_verify = preprocessing.scale(Xmat_verify)

    X_verify_features = Xmat_verify
    for clf_model in slide_windows_models:    
        # grd_enc.fit(clf_model.apply(Xmat_verify))
        # X_verify_enc = grd_enc.transform(clf_model.apply(Xmat_verify)).toarray()
        X_verify_enc = clf_model.apply(Xmat_verify)[:, :, 0]
        X_verify_features = np.column_stack((X_verify_features, X_verify_enc))

    m, n = np.shape(X_verify_features)

    print("Verify featurs shape (%d, %d)" % (m, n))

    findal_predicted_prob = logisticReg.predict_proba(X_verify_features)
    findal_predicted_prob += gbdtRegressor.predict_proba(X_verify_features)
    findal_predicted_prob += rfcls.predict_proba(X_verify_features)

    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    if (len(verify_samples) > 1000):
        topK = 1000
    else:
        topK = round(len(verify_samples) / 2)

    print("%s probility of top1 = %.4f, top%d = %.4f" % 
          (getCurrentTime(), findal_predicted_prob[prob_desc[0], 1], topK, findal_predicted_prob[prob_desc[topK], 1] ))

    predicted_user_item = []

    for index in range(topK):
        if (findal_predicted_prob[prob_desc[index], 1] < min_proba):
            print("        %s probability %.4f < min_proba %.4f, breaking..." %
                  (getCurrentTime(), findal_predicted_prob[prob_desc[index], 1], min_proba))
            break
        user_item = verify_samples[prob_desc[index]]
        predicted_user_item.append(user_item)

    actual_user_item = takingPositiveSamples(forecast_date)

    calcuatingF1(forecast_date, predicted_user_item, actual_user_item)
    return 