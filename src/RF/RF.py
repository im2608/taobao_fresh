from common import *
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from features import *
from user_features import *
from item_features import *
from taking_sample import *
import os
import csv

###############################################################################################################
###############################################################################################################
###############################################################################################################

def createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict):
    #预先分配足够的features
    feature_cnt = 100
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))

    feature_cnt = 0

    behavior_type_list = [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART, BEHAVIOR_TYPE_BUY]

    ##################################################################################################
    #######################################商品属性####################################################
    ##################################################################################################

    # 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 返回 4 个特征
    print("%s behavior count in last 14 days..." % getCurrentTime())
    begin_date = checking_date - datetime.timedelta(14)
    Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    # item 第一次behavior 距离checking date 的天数, 返回 4 个特征
    print("%s days from first behavior..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_days_from_1st_behavior(checking_date, samples); feature_cnt += 4

    # item 最后一次behavior 距离checking date 的天数, 返回 4 个特征
    print("%s days from last behavior..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_days_from_last_behavior(checking_date, samples); feature_cnt += 4

    print("%s getting item popularity..." % getCurrentTime())
    # 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品总用户数
    #Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_BUY, item_popularity_dict, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_item_popularity2(item_popularity_dict, samples); feature_cnt += 1

    ##################################################################################################
    #######################################用户 - 商品交互属性##########################################
    ##################################################################################################
    #用户在 checking date 的前 1 天是否对 item id 有过 favorite
    verify_date = checking_date - datetime.timedelta(1)
    print("%s calculating %s FAV ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples); feature_cnt += 1

    #用户在 checking date 的前 1 天是否有过 cart
    print("%s calculating %s CART ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_CART, verify_date, samples); feature_cnt += 1

    #截止到checking date(不包括)， 用户在category 上的购买浏览转化率 购买过的category数量/浏览过的category数量
    print("%s calculating user-item b/v ratio..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_buy_view_ratio(checking_date, samples); feature_cnt += 1

    # #最后一次操作同类型的商品至 checking_date 的天数
    print("%s days of last behavior of category... " % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_last_opt_category(checking_date, samples); feature_cnt += 4

    #截止到 checking_date（不包括）， 用户一共购买过多少同类型的商品
    print("%s how many category did user buy... " % getCurrentTime())
    Xmat[:, feature_cnt] = feature_how_many_buy(checking_date, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 n 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量
    for pre_days in [14, 7, 3]:        
        print("%s get behavior in last %d days" % (getCurrentTime(), pre_days))
        Xmat[:, feature_cnt : feature_cnt+4] = feature_user_item_behavior_ratio(checking_date, pre_days, samples); feature_cnt += 4

    #用户第一次购买 item 前， 在 item 上的的各个 behavior 的数量, 3个特征
    print("%s behavior count before first buy..." % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+3] = feature_behavior_cnt_before_1st_buy(checking_date, samples); feature_cnt += 3

    # 用户最后一次操作 item 至 checking_date(包括）) 的天数，
    print("%s days of last behavior" % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+4] = feature_last_opt_item(checking_date, samples); feature_cnt += 4

    #截止到checking date（不包括）， user 对 category 购买间隔的平均天数以及方差
    print("%s mean and variance days that user buy category" % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+2] = feature_mean_days_between_buy_user_category(checking_date, samples); feature_cnt += 2

    ##################################################################################################
    #######################################   用户属性   ##############################################
    ##################################################################################################
    
    # 在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车的比率
    for last_days in [14, 7, 3]:
        print("%s user's behavior count in last %d days" % (getCurrentTime(), last_days))
        begin_date = checking_date - datetime.timedelta(last_days)
        Xmat[:, feature_cnt:feature_cnt+7] = feature_how_many_behavior(begin_date, checking_date, samples); feature_cnt += 7

    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    print("%s Total features %d" % (getCurrentTime(), feature_cnt))


    #去掉没有用到的features
    Xmat = Xmat[:, 0:feature_cnt]
    m, n = np.shape(Xmat)
    print("%s shape of Xmap (%d, %d)" % (getCurrentTime(), m, n))
    Xmat = np.mat(Xmat)
    logging.info(Xmat)

    return Xmat

def randomForest(user_cnt, checking_date, forecast_date, need_output):    
    nag_per_pos = 10
    print("%s checking date %s" % (getCurrentTime(), checking_date))

    # #item 的热度
    print("%s calculating popularity..." % getCurrentTime())
    #item_popularity_dict = calculate_item_popularity()
    item_popularity_dict = calculateItemPopularity(checking_date)
    print("%s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))
    logging.info("item popularity len is %d" % len(item_popularity_dict))

    print("%s taking samples for training (%s, %d)" % (getCurrentTime(), checking_date, nag_per_pos))
    samples, Ymat = takingSamplesForTraining(checking_date, nag_per_pos, item_popularity_dict)
    print("%s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))

    Xmat = createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    rfcls = RandomForestClassifier(n_estimators=100)
    rfcls.fit(Xmat, Ymat)
    expected = Ymat

    predicted = rfcls.predict(Xmat)
    logging.info("=========== expected =========")
    logging.info(expected)

    logging.info("=========== predicted =========")
    logging.info(predicted)

    # 每种分类在每个sample 上的概率 predicted_prob[index][0] -- 0 的概率， predicted_prob[index][1] -- 1 的概率
    predicted_prob = rfcls.predict_proba(Xmat)
    logging.info("=========== predicted Probability (%d, %d)=========" % (np.shape(predicted_prob)[0], np.shape(predicted_prob)[1]))
    logging.info(predicted_prob)

    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    min_proba = 0.5
    print("=====================================================================")
    print("=========================  forecasting...(%.1f) =====================" % min_proba)
    print("=====================================================================")

    print("%s taking samples for forecasting (%s, %d)" % (getCurrentTime(), forecast_date, nag_per_pos))
    samples_test = takingSamplesForTesting(forecast_date)

    Xmat_forecast = createTrainingSet(forecast_date, nag_per_pos, samples_test, item_popularity_dict)
    # Xmat_forecast_scaler =min_max_scaler.fit_transform(Xmat_forecast)

    predicted_prob = rfcls.predict_proba(Xmat_forecast)
    loggingProbility(predicted_prob)
    
    if (need_output == 1):
        file_idx = 0
        output_file_name = "%s\\..\\output\\forecast.RF.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)
        while (os.path.exists(output_file_name)):
            file_idx += 1
            output_file_name = "%s\\..\\output\\forecast.RF.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)

        print("%s outputting %s" % (getCurrentTime(), output_file_name))
        outputFile = open(output_file_name, mode='w')
        outputFileWriter = csv.writer(outputFile, delimiter=',', quoting=csv.QUOTE_NONE)
        #outputFileWriter.writerow(["user_id", "item_id"])
        outputFile.write("\"user_id\",\"item_id\"\n")
        for index in range(len(predicted_prob)):
            if (predicted_prob[index][1] >= min_proba):
                outputFile.write("%s,%s\n" % (samples_test[index][0], samples_test[index][1]))
                #outputFileWriter.writerow([samples_test[index], samples_test[index]])

        outputFile.close()
    else:
        verifyPrediction(forecast_date, samples_test, predicted_prob, min_proba)

    return 0

def verifyPrediction(forecast_date, samples_test, predicted_prob, min_proba):
    predicted_user_item = []
    for index in range(len(predicted_prob)):
        if (predicted_prob[index][1] >= min_proba):
            predicted_user_item.append(samples_test[index])

    actual_user_item = takingPositiveSamples(forecast_date)

    actual_count = len(actual_user_item)
    predicted_count = len(predicted_user_item)

    hit_count = 0
    user_hit_list = set()

    for user_item in predicted_user_item:
        logging.info("predicted %s , %s" % (user_item[0], user_item[1]))
        if (user_item in actual_user_item):
            hit_count += 1

    for user_item in predicted_user_item:
        for user_item2 in actual_user_item:
            if (user_item[0] == user_item2[0]):
                user_hit_list.add(user_item[0])

    if (len(user_hit_list) > 0):
        logging.info("hit user: %s" % user_hit_list)

    print("hit user: %s" % user_hit_list)

    for user_item in actual_user_item:
        logging.info("acutal buy %s , %s" % (user_item[0], user_item[1]))

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



def saveTesingFeaturesToRedis():

    loadRecordsFromRedis(0, 50)
    forecast_date = datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()
    samples_test = takingSamplesForTesting(forecast_date)
    # redis_cli.set("testing_samples", samples_test)
    # print("%s saving %d testing samples to reids" % (getCurrentTime(), len(samples_test)))

    print("%s getting item popularity..." % getCurrentTime())
    verify_date = forecast_date - datetime.timedelta(1)
    print("%s calculating %s FAV ..." % (getCurrentTime(), verify_date))
    features = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples_test)
    redis_cli.set("feature_behavior_on_date", features.A)

    return 0