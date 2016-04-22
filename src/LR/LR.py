from common import *
from LR_common import *
import numpy as np
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from user_item_features import *
from item_features import *
from user_features import *
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
    ####################################   商品属性   #################################################
    ##################################################################################################

    # 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品总用户数    
    print("%s getting item popularity..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_VIEW, item_popularity_dict, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_FAV, item_popularity_dict, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_CART, item_popularity_dict, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_BUY, item_popularity_dict, samples); feature_cnt += 1


    # 过去30, 10, 3 天， 各个 behavior 在 item 上的总次数, 各有4 个特征
    print("%s behavior count in last 30 days..." % getCurrentTime())
    begin_date = checking_date - datetime.timedelta(30)
    Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    print("%s behavior count in last 10 days..." % getCurrentTime())
    begin_date = checking_date - datetime.timedelta(10)
    Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    print("%s behavior count in last 3 days..." % getCurrentTime())
    begin_date = checking_date - datetime.timedelta(3)
    Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(begin_date, checking_date, samples); feature_cnt += 4

    # item 第一次behavior 距离checking date 的天数, 返回 4 个特征
    print("%s days from 1st behavior..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_days_from_1st_behavior(checking_date, samples); feature_cnt += 4

    ##################################################################################################
    ####################################   用户 - 商品交互属性    s######################################
    ##################################################################################################
    #用户在 checking date 的前 1 天是否对 item id 有过 favorite
    verify_date = checking_date - datetime.timedelta(1)
    print("%s calculating %s FAV ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples); feature_cnt += 1

    #用户在 checking date 的前 1 天是否有过 cart
    print("%s calculating %s CART ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_CART, verify_date, samples); feature_cnt += 1

    # 得到用户在某商品上的购买浏览转化率 购买过的某商品数量/浏览过的商品数量
    print("%s calculating user-item b/v ratio..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_buy_view_ratio(samples); feature_cnt += 1

    # #最后一次操作同类型的商品至 checking_date 的天数
    print("%s calculating last buy..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_CART, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(checking_date, BEHAVIOR_TYPE_BUY, samples); feature_cnt += 1

    #截止到 checking_date（不包括）， 用户一共购买过多少同类型的商品
    Xmat[:, feature_cnt] = feature_how_many_buy_item(checking_date, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 30 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
    pre_days = 30
    print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 7 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量      
    pre_days = 7
    print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    # 用户checking_date（不包括）之前 3 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
    pre_days = 3
    print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    #用户第一次购买 item 前的各个 behavior 数
    print("%s behavior count before 1st buy..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_CART, samples); feature_cnt += 1

    #最后一次操作 item 至 checking_date 的天数的倒数
    print("%s days from last operation..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_CART, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_BUY, samples); feature_cnt += 1

    # user 对 item 购买间隔的平均天数
    print("%s mean days between buy ..." % getCurrentTime())
    Xmat[:, feature_cnt] = mean_days_between_buy_user_item(samples); feature_cnt += 1


    ##################################################################################################
    #######################################    用户属性   #############################################
    ##################################################################################################
    #用户一共购买过多少商品
    print("%s how many items user has bought in whole data set..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_how_many_buy(samples); feature_cnt += 1

    #在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车
    # 这里会产生 7 个特征
    begin_date = datetime.datetime.strptime("2014-01-01", "%Y-%m-%d").date()
    print("%s user total behavior count, b/v, b/f and b/c ratio..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+7] = feature_how_many_behavior(begin_date, checking_date, True, samples); feature_cnt += 7

    # 用户在 checking date（不包括） 之前每次购买日期距 checking date 的天数的平均值和方差
    # 这里会产生 2 个特征
    print("%s days from each buy to %s, mean and vairance..." % (getCurrentTime(), checking_date))
    Xmat[:, feature_cnt : feature_cnt+2] = feature_mean_days_between_buy_user(checking_date, samples); feature_cnt += 2

    # # 用户最后一次购买至 checking date（不包括）的天数
    print("%s last buy to %s." % (getCurrentTime(), checking_date))
    Xmat[:, feature_cnt] = feature_last_buy_user(checking_date, samples); feature_cnt += 1

    # 截止到checking_date（不包括），
    # 用户A有1周购买的商品有多少种
    # 用户A有2周购买的商品有多少种
    # 用户A有3周购买的商品有多少种
    # 用户A有4周购买的商品有多少种
    # 返回 4 个特征
    print("%s how many buy in weeks..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt + 4] = feature_how_many_buy_in_weeks(checking_date, samples); feature_cnt += 4

    #截止到checking_date（不包括）， 用户有多少天进行了各种类型的操作
    # 返回 4 个特征
    print("%s how many days for behavior..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt + 4] = feature_how_many_days_for_behavior(checking_date, samples); feature_cnt += 4    

    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    print("%s Total features %d" % (getCurrentTime(), feature_cnt))


    #去掉没有用到的features
    Xmat = Xmat[:, 0:feature_cnt]

    Xmat = np.mat(Xmat)

    return Xmat

def logisticRegression(user_cnt, checking_date, forecast_date, need_forecast, need_output, need_verify):    
    # userBehaviorStatisticOnRecords(g_user_buy_transection)
    # userBehaviorStatisticOnRecords(g_user_behavior_patten)

    print("=====================================================================")
    print("=========================  trainning...  ============================")
    print("=====================================================================")

    nag_per_pos = 15

    print("%s checking date %s, nagetive samples per positive ones %d" % (getCurrentTime(), checking_date, nag_per_pos))

    #item 的热度
    print("%s calculating popularity..." % getCurrentTime())
    item_popularity_dict = calculate_item_popularity()
    print("%s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))
    logging.info("item popularity len is %d" % len(item_popularity_dict))

    print("%s taking samples..." % getCurrentTime())
    samples, Ymat = takingSamples(checking_date, nag_per_pos, item_popularity_dict)
    print("%s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))
    if (len(samples) <= 500):
        logging.info("samples %s" % samples)
        logging.info("Ymat %s" % Ymat)

    Xmat = createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    # logging.info(Xmat_scaler)

    
    # summarize the selection of the attributes
    model = LogisticRegression()
    model.fit(Xmat, Ymat)
    # rfe = RFE(model, np.shape(Xmat)[1])
    # rfe = rfe.fit(Xmat, Ymat)
    # print(rfe.support_)
    # print(rfe.ranking_)

    expected = Ymat
    predicted = model.predict(Xmat)
    logging.info("=========== expected =========")
    logging.info(expected)

    logging.info("=========== predicted =========")
    logging.info(predicted)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print("confusion matrix: ")
    print(metrics.confusion_matrix(expected, predicted))

    if (need_forecast == 0):
        return

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

    if (need_verify == 1):
        verifyPrediction(forecast_date, samples, predicted)

    return 0


def verifyPrediction(forecast_date, samples, predicted):
    predicted_user_item = []
    for index in range(len(predicted)):
        if (predicted[index] == 1):
            predicted_user_item.append(samples[index])

    actual_user_item = takingPositiveSamples(forecast_date)

    actual_count = len(actual_user_item)
    predicted_count = len(predicted_user_item)

    hit_count = 0

    for user_item in predicted_user_item:
        if (user_item in actual_user_item):
            hit_count += 1

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
