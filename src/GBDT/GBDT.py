from common import *
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.utils import shuffle

from features import *
from user_features import *
from item_features import *
from user_category_features import *
from taking_sample import *

import os




def createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, item_popularity_dict):
    #预先分配足够的features
    feature_cnt = 100
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))

    feature_cnt = 0

    days_in_windows = (window_end_date - window_start_date).days

    ##################################################################################################
    #######################################商品属性####################################################
    ##################################################################################################

    # 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 返回 4 个特征
    Xmat[:, feature_cnt : feature_cnt+4] = feature_beahvior_cnt_on_item(window_start_date, window_end_date, samples); feature_cnt += 4

    # item 第一次behavior 距离checking date 的天数, 返回 4 个特征
    print("        %s days from first behavior..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_days_from_1st_behavior(window_start_date, window_end_date, samples); feature_cnt += 4

    # item 最后一次behavior 距离checking date 的天数, 返回 4 个特征
    print("        %s days from last behavior..." % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_days_from_last_behavior(window_start_date, window_end_date, samples); feature_cnt += 4

    print("        %s getting item popularity..." % getCurrentTime())
    # 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品总用户数
    #Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_BUY, item_popularity_dict, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_item_popularity2(item_popularity_dict, samples); feature_cnt += 1

    ##################################################################################################
    #######################################用户 - 商品交互属性##########################################
    ##################################################################################################
    #用户在 checking date 的前 1 天是否对 item id 有过 favorite
    verify_date = window_end_date - datetime.timedelta(1)
    print("        %s calculating %s FAV ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples); feature_cnt += 1

    #用户在 checking date 的前 1 天是否有过 cart
    print("        %s calculating %s CART ..." % (getCurrentTime(), verify_date))
    Xmat[:, feature_cnt] = feature_behavior_on_date(BEHAVIOR_TYPE_CART, verify_date, samples); feature_cnt += 1

    # 在 [windw_start_date, window_end_dat) 范围内， 
    # 用户checking_date（不包括）之前 n 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量
    for pre_days in [days_in_windows, round(days_in_windows/2), round(days_in_windows/3)]:
        print("        %s get behavior in last %d days" % (getCurrentTime(), pre_days))
        Xmat[:, feature_cnt : feature_cnt+4] = feature_user_item_behavior_ratio(window_end_date, pre_days, samples); feature_cnt += 4

    #在 [windw_start_date, window_end_dat) 范围内， 用户第一次购买 item 前， 在 item 上的的各个 behavior 的数量, 3个特征
    print("        %s behavior count before first buy..." % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+3] = feature_behavior_cnt_before_1st_buy(window_start_date, window_end_date, samples); feature_cnt += 3

    #在 [windw_start_date, window_end_dat) 范围内，  用户最后一次操作 item 至 checking_date(包括）) 的天数，
    print("        %s days of last behavior" % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+4] = feature_last_opt_item(window_start_date, window_end_date, samples); feature_cnt += 4

    ##################################################################################################
    ###################################### 用户 - category交互属性  ###################################
    ##################################################################################################
    #在 [windw_start_date, window_end_dat) 范围内， ， 用户在category 上的购买浏览转化率 购买过的category数量/浏览过的category数量
    print("        %s calculating user-item b/v ratio..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_buy_view_ratio(window_start_date, window_end_date, samples); feature_cnt += 1

    #在 [windw_start_date, window_end_dat) 范围内， 用户最后一次操作同类型的商品至 checking_date 的天数
    print("        %s days of last behavior of category... " % getCurrentTime())
    Xmat[:, feature_cnt : feature_cnt+4] = feature_last_opt_category(window_start_date, window_end_date, samples); feature_cnt += 4

    # 在 [windw_start_date, window_end_dat) 范围内， ， 用户一共购买过多少同类型的商品
    print("        %s how many category did user buy... " % getCurrentTime())
    Xmat[:, feature_cnt] = feature_how_many_buy_category(window_start_date, window_end_date, samples); feature_cnt += 1

    #在 [windw_start_date, window_end_dat) 范围内， user 对 category 购买间隔的平均天数以及方差
    print("        %s mean and variance days that user buy category" % (getCurrentTime()))
    Xmat[:, feature_cnt : feature_cnt+2] = feature_mean_days_between_buy_user_category(window_start_date, window_end_date, samples); feature_cnt += 2

    ##################################################################################################
    #######################################   用户属性   ##############################################
    ##################################################################################################
    #在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车的比率
    for last_days in [days_in_windows, round(days_in_windows/2), round(days_in_windows/3)]:
        print("        %s user's behavior count in last %d days" % (getCurrentTime(), last_days))
        begin_date = window_end_date - datetime.timedelta(last_days)
        Xmat[:, feature_cnt:feature_cnt+7] = feature_how_many_behavior_user(begin_date, window_end_date, samples); feature_cnt += 7

    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    print("        %s Total features %d" % (getCurrentTime(), feature_cnt))

    #去掉矩阵中没有用到的列
    Xmat = Xmat[:, 0:feature_cnt]
    m, n = np.shape(Xmat)
    print("        %s shape of Xmap (%d, %d)" % (getCurrentTime(), m, n))
    Xmat = np.mat(Xmat)

    np.savetxt("%s\\..\log\\X_mat.txt" % runningPath, Xmat, fmt="%.4f", newline="\n")

    return Xmat

# 滑动窗口， window_end_date 为 Y， 从 [window_start_date, window_end_date -1] 范围内得到特征矩阵， 通过
# 训练得到特征的 importance
def GBDT_slideWindows(window_start_date, window_end_date):
    nag_per_pos = 10
    print("%s slide windows from %s to %s" % (getCurrentTime(), window_start_date, window_end_date))

    # #item 的热度
    print("        %s calculating popularity..." % getCurrentTime())
    #item_popularity_dict = calculate_item_popularity()
    item_popularity_dict = calculateItemPopularity(window_start_date, window_end_date)
    print("        %s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))

    print("        %s taking samples for training (%s, %d)" % (getCurrentTime(), window_end_date, nag_per_pos))
    samples, Ymat = takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos, item_popularity_dict)
    print("        %s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))

    Xmat = createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, item_popularity_dict)

    Xmat, Ymat = shuffle(Xmat, Ymat, random_state=13)

    params = {'n_estimators': 100, 
              'max_depth': 4,
              'min_samples_split': 1,
              'learning_rate': 0.01, 
              'loss': 'ls'
              }

    clf = GradientBoostingRegressor(**params)
    clf.fit(Xmat, Ymat)

    feature_importance = clf.feature_importances_

    logging.info("slide window [%s, %s], features importance %s " % (window_start_date, window_end_date, feature_importance))

    return feature_importance


def GradientBoostingRegressionTree(checking_date, forecast_date, need_output):
    nag_per_pos = 10
    print("        %s checking date %s" % (getCurrentTime(), checking_date))

    # #item 的热度
    print("        %s calculating popularity..." % getCurrentTime())
    #item_popularity_dict = calculate_item_popularity()
    item_popularity_dict = calculateItemPopularity(checking_date)
    print("        %s item popularity len is %d" % (getCurrentTime(), len(item_popularity_dict)))
    logging.info("item popularity len is %d" % len(item_popularity_dict))

    print("        %s taking samples for training (%s, %d)" % (getCurrentTime(), checking_date, nag_per_pos))
    samples, Ymat = takingSamplesForTraining(checking_date, nag_per_pos, item_popularity_dict)
    print("        %s samples count %d, Ymat count %d" % (getCurrentTime(), len(samples), len(Ymat)))

    Xmat = createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict)

    Xmat, Ymat = shuffle(Xmat, Ymat, random_state=13)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    params = {'n_estimators': 100, 
              'max_depth': 4,
              'min_samples_split': 1,
              'learning_rate': 0.01, 
              'loss': 'ls'
              }

    clf = GradientBoostingRegressor(**params)
    clf.fit(Xmat, Ymat)
    X_leaves = clf.apply(Xmat)
    # print("X_leaves shape ", np.shape(X_leaves))
    # np.savetxt("%s\\..\log\\X_leaves.txt" % runningPath, X_leaves, fmt="%.4f", newline="\n")

    expected = Ymat
    predicted = clf.predict(Xmat)
    feature_importance = clf.feature_importances_
    print("feature_importance shape :", np.shape(feature_importance))
    print(feature_importance)

    print("=========== expected =========")
    print(expected)

    print("=========== predicted =========")
    print(predicted)

'''
    min_proba = 0.7
    print("=====================================================================")
    print("=========================  forecasting...(%.1f) ==================" % min_proba)
    print("=====================================================================")

    print("        %s taking samples for forecasting (%s, %d)" % (getCurrentTime(), forecast_date, nag_per_pos))
    samples_test = takingSamplesForTesting(forecast_date)

    Xmat_forecast = createTrainingSet(forecast_date, nag_per_pos, samples_test, item_popularity_dict)

    # Xmat_forecast_scaler =min_max_scaler.fit_transform(Xmat_forecast)
    # predicted = clf.predict(Xmat_forecast)
    # 每种分类在每个sample 上的概率 predicted_prob[index][0] -- 0 的概率， predicted_prob[index][1] -- 1 的概率
    predicted_prob = clf.predict_proba(Xmat_forecast)

    if (need_output == 1):
        file_idx = 0
        output_file_name = "%s\\..\\output\\forecast.LR.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)
        while (os.path.exists(output_file_name)):
            file_idx += 1
            output_file_name = "%s\\..\\output\\forecast.LR.%d.%s.%d.csv" % (runningPath, np.shape(Xmat)[1], datetime.date.today(), file_idx)

        print("        %s outputting %s" % (getCurrentTime(), output_file_name))
        outputFile = open(output_file_name, encoding="utf-8", mode='w')
        outputFile.write("user_id,item_id\n")
        for index in range(len(predicted_prob)):
            if (predicted_prob[index][1] >= min_proba):
                outputFile.write("%s,%s\n" % (samples_test[index][0], samples_test[index][1]))

        outputFile.close()
    else:
        verifyPrediction(forecast_date, samples_test, predicted_prob, min_proba)

    return 0
'''
