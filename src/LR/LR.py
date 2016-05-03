from common import *
from LR_common import *
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
#from user_features import *
from user_item_features import *
from user_features import *
from item_features import *
from taking_sample import *
import os



###############################################################################################################
###############################################################################################################
###############################################################################################################

def createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict):
    #预先分配足够的features
    feature_cnt = 100
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))

    feature_cnt = 0

    ##################################################################################################
    #######################################商品属性####################################################
    ##################################################################################################
    
    print("%s getting item popularity..." % getCurrentTime())

    # 商品热度 
    Xmat[:, feature_cnt] = feature_item_popularity(item_popularity_dict, samples); feature_cnt += 1

    print("%s getting features..." % getCurrentTime())

    # 过去14, 7, 3 天， 各个 behavior 在 item 上的总次数, 各有4 个特征
    # print("%s behavior count in last 14 days..." % getCurrentTime())
    # begin_date = checking_date - datetime.timedelta(14)
    # Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    # print("%s behavior count in last 10 days..." % getCurrentTime())
    # begin_date = checking_date - datetime.timedelta(7)
    # Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    print("%s behavior count in last 3 days..." % getCurrentTime())
    begin_date = checking_date - datetime.timedelta(3)
    Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    ##################################################################################################
    #######################################用户 - 商品交互属性##########################################
    ##################################################################################################
    #用户在 checking date 的前 1 天是否对 item id 有过 favorite
    verify_date = checking_date - datetime.timedelta(1)
    # print("%s calculating %s FAV ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples); feature_cnt += 1

    #用户在 checking date 的前 1 天是否有过 cart
    # print("%s calculating %s CART ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_CART, verify_date, samples); feature_cnt += 1

    # 得到用户在某商品上的购买浏览转化率 购买过的某商品数量/浏览过的商品数量
    # print("%s calculating user-item b/v ratio..." % getCurrentTime())
    # Xmat[:, feature_cnt] = feature_buy_view_ratio(samples); feature_cnt += 1

    # #最后一次操作同类型的商品至 checking_date 的天数
    # print("%s calculating last buy..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_CART, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_BUY, samples); feature_cnt += 1

    #截止到 checking_date（不包括）， 用户一共购买过多少同类型的商品
    Xmat[:, feature_cnt] = feature_how_many_buy(checking_date, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 30 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
    pre_days = 14
    # print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 7 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量      
    pre_days = 7
    # print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 3 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
    pre_days = 3
    # print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    #用户第一次购买前的各个 behavior 数
    # print("%s behavior count before 1st buy..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_CART, samples); feature_cnt += 1

    #最后一次操作 item 至 checking_date 的天数的倒数
    # print("%s days from last operation..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_CART, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_BUY, samples); feature_cnt += 1

    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    # print("%s Total features %d" % (getCurrentTime(), feature_cnt))


    #去掉没有用到的features
    Xmat = Xmat[:, 0:feature_cnt]

    m, n = np.shape(Xmat)
    print("shape of Xmap (%d, %d)" % (m, n))
    Xmat = np.mat(Xmat)
    logging.info(Xmat)

    return Xmat

def logisticRegression(user_cnt, checking_date, forecast_date, need_output):    
    nag_per_pos = 15
    print("%s checking date %s" % (getCurrentTime(), checking_date))

    print("%s taking samples..." % getCurrentTime())
    samples, Ymat, item_popularity_dict = takingSamples(checking_date, nag_per_pos)
    print("%s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))
    if (len(samples) <= 500):
        logging.info("samples %s" % samples)
        logging.info("Ymat %s" % Ymat)

    Xmat = createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    model = LogisticRegression()
    model.fit(Xmat, Ymat)
    expected = Ymat
    predicted = model.predict(Xmat)
    logging.info("=========== expected =========")
    logging.info(expected)

    logging.info("=========== predicted =========")
    logging.info(predicted)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    print("=====================================================================")
    print("=========================  forecasting... ===========================")
    print("=====================================================================")
    Xmat_forecast = createTrainingSet(forecast_date, nag_per_pos, samples, item_popularity_dict)

    #Xmat_forecast_scaler =min_max_scaler.fit_transform(Xmat_forecast)
    predicted = model.predict(Xmat_forecast)

    if (need_output == 1):
        file_idx = 0
        output_file_name = "%s\\..\\output\\forecast.LR.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)
        while (os.path.exists(output_file_name)):
            file_idx += 1
            output_file_name = "%s\\..\\output\\forecast.LR.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)

        outputFile = open(output_file_name, encoding="utf-8", mode='w')
        outputFile.write("user_id,item_id\n")
        for index in range(len(predicted)):
            if (predicted[index] == 1):
                outputFile.write("%s,%s\r" % (samples[index][0], samples[index][1]))

        outputFile.close()
    else:
        verifyPrediction(forecast_date, samples, predicted)

    return 0


def verifyPrediction(forecast_date, samples, predicted):
    predicted_user_item = []
    for index in range(len(predicted)):
        if (predicted[index] == 1):
            predicted_user_item.append(samples[index])

    actual_user_item, items_in_postive = takingPositiveSamples(forecast_date)

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
