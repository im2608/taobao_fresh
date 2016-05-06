from common import *
from LR_common import *
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from features import *
from user_features import *
from item_features import *
from taking_sample import *
import os

#35ebe50


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
    Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_BUY, item_popularity_dict, samples); feature_cnt += 1

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
    for behavior_type in behavior_type_list:
        print("%s calculating last behavior %d " % (getCurrentTime(), behavior_type))
        Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, behavior_type, samples); feature_cnt += 1

    #截止到 checking_date（不包括）， 用户一共购买过多少同类型的商品
    Xmat[:, feature_cnt] = feature_how_many_buy(checking_date, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 n 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量
    for pre_days in [14, 7, 3]:        
        for behavior_type in behavior_type_list:
            print("%s get behavior %d ratios in last %d days" % (getCurrentTime(), behavior_type, pre_days))
            Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, behavior_type, pre_days, samples); feature_cnt += 1

    #用户第一次购买前的各个 behavior 数
    for behavior_type in [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART]:
        print("%s behavior %d count before first buy..." % (getCurrentTime(), behavior_type))

    # 用户最后一次操作 item 至 checking_date(包括）) 的天数，
    for behavior_type in behavior_type_list:
        print("%s days of last behavior %d" % (getCurrentTime(), behavior_type))
        Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, behavior_type, samples); feature_cnt += 1

    #截止到checking date（不包括）， user 对 category 购买间隔的平均天数以及方差
    print("%s mean and variance days that user buy category" % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+2] = feature_mean_days_between_buy_user_category(checking_date, samples); feature_cnt += 2

    ##################################################################################################
    #######################################   用户属性   ##############################################
    ##################################################################################################
    
    #在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车的比率
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

def logisticRegression(user_cnt, checking_date, forecast_date, need_output):    
    nag_per_pos = 10
    print("%s checking date %s" % (getCurrentTime(), checking_date))

    # #item 的热度
    print("%s calculating popularity..." % getCurrentTime())
    item_popularity_dict = calculate_item_popularity()
    #item_popularity_dict = calculateItemPopularity(checking_date)
    print("%s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))
    logging.info("item popularity len is %d" % len(item_popularity_dict))

    print("%s taking samples for training (%s, %d)" % (getCurrentTime(), checking_date, nag_per_pos))
    samples, Ymat = takingSamplesForTraining(checking_date, nag_per_pos, item_popularity_dict)
    print("%s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))

    Xmat = createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict)

    min_max_scaler = preprocessing.MinMaxScaler()
    Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    logging.info(Xmat_scaler)

    model = LogisticRegression()
    # model = DecisionTreeClassifier()

    model.fit(Xmat_scaler, Ymat)
    expected = Ymat
    predicted = model.predict(Xmat_scaler)
    logging.info("=========== expected =========")
    logging.info(expected)

    logging.info("=========== predicted =========")
    logging.info(predicted)

    # 每种分类在每个sample 上的概率
    # predicted_prob = model.predict_proba(Xmat_scaler)
    # logging.info("=========== predicted Probability (%d, %d)=========" % (np.shape(predicted_prob)[0], np.shape(predicted_prob)[1]))
    # logging.info(predicted_prob)

    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("=====================================================================")
    print("=========================  forecasting... ===========================")
    print("=====================================================================")

    print("%s taking samples for forecasting (%s, %d, False)" % (getCurrentTime(), checking_date, nag_per_pos))
    samples_test = takingSamplesForTesting(checking_date)

    Xmat_forecast = createTrainingSet(forecast_date, nag_per_pos, samples_test, item_popularity_dict)

    Xmat_forecast_scaler =min_max_scaler.fit_transform(Xmat_forecast)
    predicted = model.predict(Xmat_forecast_scaler)

    if (need_output == 1):
        file_idx = 0
        output_file_name = "%s\\..\\output\\forecast.LR.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)
        while (os.path.exists(output_file_name)):
            file_idx += 1
            output_file_name = "%s\\..\\output\\forecast.LR.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)

        print("outputting %s" % output_file_name)
        outputFile = open(output_file_name, encoding="utf-8", mode='w')
        outputFile.write("user_id,item_id\n")
        for index in range(len(predicted)):
            if (predicted[index] == 1):
                outputFile.write("%s,%s\r" % (samples_test[index][0], samples_test[index][1]))

        outputFile.close()
    else:
        verifyPrediction(forecast_date, samples_test, predicted)

    return 0



def verifyPrediction(forecast_date, samples_test, predicted):
    predicted_user_item = []
    for index in range(len(predicted)):
        if (predicted[index] == 1):
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
