from common import *
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from features import *
from user_features import *
from item_features import *
from category_features import *
from user_category_features import *
from taking_sample import *
    
import os




def createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, item_popularity_dict, during_training):
    logging.info("entered createTrainingSet %s - %s " % (window_start_date, window_end_date))
    slide_start_time = time.clock()
    #预先分配足够的features
    feature_cnt_buf = 200
    Xmat = np.mat(np.zeros((len(samples), feature_cnt_buf)))

    total_feature_cnt = 0

    days_in_window = (window_end_date - window_start_date).days
    ##################################################################################################
    #######################################商品属性####################################################
    ##################################################################################################

    # 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 平均每天的点击数以及方差
    # 返回 12 个特征
    print("        %sbehavior count..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_beahvior_cnt_on_item(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_beahvior_cnt_on_item takes %d seconds" % (getCurrentTime(), time_end - time_start))


    # item 第一次behavior 距离checking date 的天数, 返回 4 个特征
    print("        %s days from first behavior..." % getCurrentTime())
    time_start = time.clock()
    feature_mat, feature_cnt = feature_days_from_1st_behavior(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_days_from_1st_behavior takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # item 最后一次behavior 距离checking date 的天数, 返回 4 个特征
    print("        %s days from last behavior..." % getCurrentTime())
    time_start = time.clock()
    feature_mat, feature_cnt = feature_days_from_last_behavior(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_days_from_last_behavior takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # print("        %s getting item popularity..." % getCurrentTime())
    # time_start = time.clock()
    # # 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品总用户数
    # #Xmat[:, feature_cnt] = feature_item_popularity(BEHAVIOR_TYPE_BUY, item_popularity_dict, samples); feature_cnt += 1
    # feature_mat, feature_cnt = feature_item_popularity2(item_popularity_dict, samples, during_training, total_feature_cnt)
    # total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    # time_end = time.clock()
    # print("        %s feature_item_popularity2 takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # [begin date, end date) 期间，总共有多少用户购买了该 item
    print("        %s how much users buy this item..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_users_bought(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_users_bought takes %d seconds" % (getCurrentTime(), time_end - time_start))

    ##################################################################################################
    #######################################用户 - 商品交互属性##########################################
    ##################################################################################################
    #用户在 checking date 的前 1 天是否对 item id 有过 favorite
    verify_date = window_end_date - datetime.timedelta(1)
    print("        %s calculating %s FAV ..." % (getCurrentTime(), verify_date))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_behavior_on_date(BEHAVIOR_TYPE_FAV) takes %d seconds" % (getCurrentTime(), time_end - time_start))

    #用户在 checking date 的前 1 天是否有过 cart
    print("        %s calculating %s CART ..." % (getCurrentTime(), verify_date))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_behavior_on_date(BEHAVIOR_TYPE_CART, verify_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_behavior_on_date(BEHAVIOR_TYPE_CART) takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # 在 [window_start_date, window_end_dat) 范围内， 
    # 用户checking_date（不包括）之前 n 天购买（浏览， 收藏， 购物车）该商品的次数/该用户购买（浏览， 收藏， 购物车）所有商品的总数量    
    for pre_days in [days_in_window, round(days_in_window/2), round(days_in_window/3)]:
        print("        %s get behavior in last %d days" % (getCurrentTime(), pre_days))
        time_start = time.clock()
        feature_mat, feature_cnt = feature_user_item_behavior_ratio(window_end_date, pre_days, samples, during_training, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
        time_end = time.clock()
        print("        %s feature_user_item_behavior_ratio(%d) takes %d seconds" % (getCurrentTime(), pre_days, time_end - time_start))


    #在 [window_start_date, window_end_dat) 范围内， 用户第一次购买 item 前， 在 item 上的的各个 behavior 的数量, 3个特征
    print("        %s behavior count before first buy..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_behavior_cnt_before_1st_buy(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_behavior_cnt_before_1st_buy takes %d seconds" % (getCurrentTime(), time_end - time_start))

    #在 [window_start_date, window_end_dat) 范围内，  用户最后一次操作 item 至 checking_date(包括）) 的天数，
    print("        %s days of last behavior" % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_last_opt_item(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_last_opt_item takes %d seconds" % (getCurrentTime(), time_end - time_start))


    print("        %s mean days user buy item..." % (getCurrentTime()))
    feature_mat, feature_cnt = feature_mean_days_between_buy_user_item(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s mean days user buy item takes %d seconds" % (getCurrentTime(), time_end - time_start))

    ##################################################################################################
    ###################################### 用户 - category交互属性  ###################################
    ##################################################################################################
    #在 [window_start_date, window_end_dat) 范围内， ， 用户在category 上的购买浏览转化率 购买过的category数量/浏览过的category数量
    print("        %s calculating user-item b/v ratio..." % getCurrentTime())
    time_start = time.clock()
    feature_mat, feature_cnt = feature_buy_view_ratio(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_buy_view_ratio takes %d seconds" % (getCurrentTime(), time_end - time_start))

    #在 [window_start_date, window_end_dat) 范围内， 用户最后一次操作同类型的商品至 checking_date 的天数
    print("        %s days of last behavior of category... " % getCurrentTime())
    time_start = time.clock()
    feature_mat, feature_cnt = feature_last_opt_category(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_last_opt_category takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # 在 [window_start_date, window_end_dat) 范围内， ， 用户一共购买过多少同类型的商品
    print("        %s how many category did user buy... " % getCurrentTime())
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_buy_category(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_buy_category takes %d seconds" % (getCurrentTime(), time_end - time_start))

    #在 [window_start_date, window_end_dat) 范围内， user 对 category 购买间隔的平均天数以及方差
    print("        %s mean and variance days that user buy category" % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_mean_days_between_buy_user_category(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_mean_days_between_buy_user_category takes %d seconds" % (getCurrentTime(), time_end - time_start))

    ##################################################################################################
    #######################################   用户属性   ##############################################
    ##################################################################################################
    #在 [begin_date, end_date)时间段内， 用户总共有过多少次浏览，收藏，购物车，购买的行为以及 购买/浏览， 购买/收藏， 购买/购物车的比率
    for pre_days in [days_in_window, round(days_in_window/2), round(days_in_window/3)]:
        print("        %s user's behavior count in last %d days" % (getCurrentTime(), pre_days))
        time_start = time.clock()
        feature_mat, feature_cnt = feature_how_many_behavior_user(pre_days, window_end_date, samples, during_training, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
        time_end = time.clock()
        print("        %s feature_how_many_behavior_user takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # 用户在 checking date（不包括） 之前每次购买间隔的天数的平均值和方差, 返回两个特征
    print("        %s days between buy of user..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_mean_days_between_buy_user(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_mean_days_between_buy_user takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # [window_start_date, window_end_date) 期间， 用户最后一次行为至 window_end_date （不包括）的天数, 没有该行为则为 0, 返回4个特征
    print("        %s days from user's last behavior ..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_last_behavior_user(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_last_behavior_user takes %d seconds" % (getCurrentTime(), time_end - time_start))

    #截止到checking_date（不包括）， 用户有多少天进行了各种类型的操作
    # 返回 4 个特征
    print("        %s how many days on which user has behavior..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_days_for_behavior(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_days_for_behavior takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # [start date, end date) 范围内，用户购买过 1/2/3/4 ... ... /slide window days 次的item有多少， 返回 slide window days 个特征
    # 用户在同一天内多次购买同一个item算一次
    # 例如 用户在 第1天购买了item1，item2， item3， 然后在第5天又购买了该item1, 第6 天购买了 item2， 第7 天购买了item3，第 8 天有购买了item3
    # 用户购买过item1， item2两次，购买过item3 三次，则buy_in_days_list[2] = 2， buy_in_days_list[3] = 1
    print("        %s feature_how_many_buy_in_days..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_buy_in_days(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_buy_in_days takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # ##################################################################################################
    # #######################################   category 属性 ###########################################
    # ##################################################################################################
    # category 第一次, 最后一次 behavior 距离checking date 的天数, 返回 8 个特征
    print("        %s feature_days_from_1st_last_behavior_category..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_days_from_1st_last_behavior_category(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_days_from_1st_last_behavior_category takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # # 在 [begin_date, checking_date) 期间， 各个 behavior 在 item 上的总次数, 平均每天的点击数以及方差
    # # 返回 12 个特征
    print("        %s feature_beahvior_cnt_on_category..." % (getCurrentTime()))
    time_start = time.clock()
    feature_mat, feature_cnt = feature_beahvior_cnt_on_category(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_beahvior_cnt_on_category takes %d seconds" % (getCurrentTime(), time_end - time_start))

    # [begin date, end date) 期间，总共有多少用户购买了该 category
    print("        %s how many users buy category..." % (getCurrentTime()))
    feature_how_many_users_bought_category(window_start_date, window_end_date, samples, during_training, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s how many users buy category takes %d seconds" % (getCurrentTime(), time_end - time_start))

    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    print("        %s Total features %d" % (getCurrentTime(), total_feature_cnt))

    #去掉矩阵中没有用到的列
    Xmat = Xmat[:, 0:total_feature_cnt]
    m, n = np.shape(Xmat)

    slide_end_time = time.clock()
    print("        %s shape of Xmap (%d, %d), slide window(%s, %s) took %d seconds" %
         (getCurrentTime(), m, n, window_start_date, window_end_date, slide_end_time - slide_start_time))

    Xmat = np.mat(Xmat)

    # np.savetxt("%s\\..\log\\X_mat.txt" % runningPath, Xmat, fmt="%.4f", newline="\n")

    logging.info("leaving createTrainingSet")

    return Xmat

# 滑动窗口， window_end_date 为 Y， 从 [window_start_date, window_end_date -1] 范围内得到特征矩阵， 通过
# 训练得到特征的 importance
def GBDT_slideWindows(window_start_date, window_end_date, during_training):
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

    Xmat = createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, item_popularity_dict, during_training)
    if (len(samples) == 0):
        print("%s No buy records from %s to %s, returning..." % (getCurrentTime(), window_start_date, window_end_date))
        return None

    Xmat = preprocessing.scale(Xmat)
    m, n = np.shape(Xmat)

    Xmat, Ymat = shuffle(Xmat, Ymat, random_state=13)

    params = {'n_estimators': 500, 
              'max_depth': 4,
              'min_samples_split': 1,
              'learning_rate': 0.01, 
              #'loss': 'ls'
              'loss': 'deviance'
              }

    #clf = GradientBoostingRegressor(**params)
    clf = GradientBoostingClassifier(**params)

    clf.fit(Xmat, Ymat)

    feature_importance = clf.feature_importances_

    logging.info("slide window [%s, %s], features (%d, %d) importance %s " % (window_start_date, window_end_date, m, n, feature_importance))

    # if (during_training):
    #     plotDeviance(window_start_date, window_end_date, nag_per_pos, params, clf)

    return clf


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

    Xmat = createTrainingSet(checking_date, nag_per_pos, samples, item_popularity_dict, False)

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

