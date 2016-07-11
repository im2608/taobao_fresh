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
from cross_features import *

import os




def createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, cal_feature_importance, final_feature_importance):
    logging.info("entered createTrainingSet %s - %s " % (window_start_date, window_end_date))
    slide_start_time = time.clock()
    #预先分配足够的features
    feature_cnt_buf = 300
    Xmat = np.mat(np.zeros((len(samples), feature_cnt_buf)))

    total_feature_cnt = 0

    days_in_window = (window_end_date - window_start_date).days

    pre_days_list = [days_in_window, round(days_in_window/2), round(days_in_window/4)]
    ##################################################################################################
    #######################################商品属性####################################################
    ##################################################################################################


    # item 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
    print("        %s 1st, last behavior and days between them...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_days_from_1st_last_behavior_item(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_days_from_1st_last_behavior_item takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # [begin date, end date) 期间，总共有多少用户在该 item 上进行了各种操作，按照操作数量进行加权，得到 item 上的加权在 category 中的排序
    print("        %s how much users buy this item...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_users_behavior_item(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    user_cnt_buy_item = feature_mat[:, 3]
    print("        %s feature_how_many_users_bought takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # [begin date, end date) 期间， item 的销量
    print("        %s item sales volume...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_item_sals_volume(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    item_sales_vol = feature_mat[:, 0]
    print("        %s feature_item_sals_volume takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    ##################################################################################################
    #######################################用户 - 商品交互属性##########################################
    ##################################################################################################
    #用户在 checking date 的前 1 天是否对 item id 有过 favorite
    verify_date = window_end_date - datetime.timedelta(1)
    print("        %s calculating %s FAV ...   \r" % (getCurrentTime(), verify_date), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_behavior_on_date(BEHAVIOR_TYPE_FAV, verify_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_behavior_on_date(BEHAVIOR_TYPE_FAV) takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    #用户在 checking date 的前 1 天是否有过 cart
    print("        %s calculating %s CART ...   \r" % (getCurrentTime(), verify_date), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_behavior_on_date(BEHAVIOR_TYPE_CART, verify_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_behavior_on_date(BEHAVIOR_TYPE_CART) takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # 用户checking_date（不包括）之前 pre_days 天购买（浏览， 收藏， 购物车）该商品的次数, 这些次数占该用户购买（浏览， 收藏， 购物车）所有商品的总次数的比例,
    # 用户在item上pre_days 天购买/浏览，  
    # 用户在item上pre_days 天购买/购物车， 
    # 用户在item上 pre_days 天购买/收藏， 
    # 用户在item上 pre_days 天购物车/收藏， 
    # 用户在item上pre_days 天购物车/浏览, 
    # 返回 13 个特征
    user_behavior_cnt_on_item = []
    for pre_days in pre_days_list:
        print("        %s get behavior in last %d days   \r" % (getCurrentTime(), pre_days), end="")
        time_start = time.clock()
        feature_mat, feature_cnt = feature_user_item_behavior_ratio(window_end_date, pre_days, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
        time_end = time.clock()
        user_behavior_cnt_on_item.append(feature_mat[:, 0:4]) # 保留用户在item上各个behavior的次数
        print("        %s feature_user_item_behavior_ratio(%d) takes %d seconds, total featurs %d   \r" % (getCurrentTime(), pre_days, time_end - time_start, total_feature_cnt), end="")

    #在 [window_start_date, window_end_dat) 范围内， 用户第一次购买 item 前， 在 item 上的的各个 behavior 的数量, 3个特征
    print("        %s behavior count before first buy...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_behavior_cnt_before_1st_buy(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_behavior_cnt_before_1st_buy takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # 用户最后一次操作 item 至 checking_date(不包括) 的天数，以及在item上最后一次cart 至最后一次buy之间的天数, 返回5个特征
    print("        %s days of last behavior   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_last_opt_item(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_last_opt_item takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    #在 [windw_start_date, window_end_dat) 范围内， user 对 item 购买间隔的平均天数
    print("        %s mean days user buy item...   \r" % (getCurrentTime()), end="")
    feature_mat, feature_cnt = feature_mean_days_between_buy_user_item(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s mean days user buy item takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    ##################################################################################################
    ###################################### 用户 - category交互属性  ###################################
    ##################################################################################################
    #在 [window_start_date, window_end_dat) 范围内， ， 用户在category 上的购买浏览转化率 购买过的category数量/浏览过的category数量
    print("        %s calculating user-item b/v ratio...\r" % getCurrentTime(), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_buy_view_ratio(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_buy_view_ratio takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    #在 [window_start_date, window_end_dat) 范围内， 用户最后一次操作同类型的商品至 checking_date 的天数
    print("        %s days of last behavior of category... \r" % getCurrentTime(), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_last_opt_category(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_last_opt_category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # 在 [window_start_date, window_end_dat) 范围内， ， 用户一共购买过多少同类型的商品
    print("        %s how many category did user buy...   \r" % getCurrentTime(), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_buy_category(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_buy_category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    #在 [window_start_date, window_end_dat) 范围内， user 对 category 购买间隔的平均天数以及方差
    print("        %s mean and variance days that user buy category   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_mean_days_between_buy_user_category(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_mean_days_between_buy_user_category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # user 在 category 上各个行为的次数以及在item上各个行为的次数占category上次数的比例
    # 返回 8 个特征
    print("        %s user behavior count on category   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    for i, pre_days in enumerate(pre_days_list):
        feature_mat, feature_cnt = feature_user_behavior_cnt_on_category(pre_days, window_end_date, samples, user_behavior_cnt_on_item[i], cal_feature_importance, final_feature_importance, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s user behavior count on category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    ##################################################################################################
    #######################################   用户属性   ##############################################
    ##################################################################################################
    # 距离 end_date pre_days 天内， 用户总共有过多少次浏览，收藏，购物车，购买的行为, 购买/浏览， 购买/收藏， 购买/购物车, 购物车/收藏， 购物车/浏览的比率,
    # 返回 9 个特征
    print("        %s how many behavior of user...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    for i, pre_days in enumerate(pre_days_list):
        feature_mat, feature_cnt = feature_how_many_behavior_user(pre_days, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()

    # 用户在 checking date（不包括） 之前每次购买间隔的天数的平均值和方差, 返回两个特征
    print("        %s days between buy of user...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_mean_days_between_buy_user(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_mean_days_between_buy_user takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # [window_start_date, window_end_date) 期间， 用户最后一次行为至 window_end_date （不包括）的天数, 没有该行为则为 0, 返回4个特征
    print("        %s days from user's last behavior ...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_last_behavior_user(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_last_behavior_user takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    #截止到checking_date（不包括）， 用户有多少天进行了各种类型的操作
    # 返回 4 个特征
    print("        %s how many days on which user has behavior...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_days_for_behavior(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_days_for_behavior takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # [start date, end date) 范围内，用户购买过 1/2/3/4 ... ... /slide window days 次的item有多少， 返回 slide window days 个特征
    # 用户在同一天内多次购买同一个item算一次
    # 例如 用户在 第1天购买了item1，item2， item3， 然后在第5天又购买了该item1, 第6 天购买了 item2， 第7 天购买了item3，第 8 天有购买了item3
    # 用户购买过item1， item2两次，购买过item3 三次，则buy_in_days_list[2] = 2， buy_in_days_list[3] = 1
    print("        %s feature_how_many_buy_in_days...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_buy_in_days(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_how_many_buy_in_days takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # ##################################################################################################
    # #######################################   category 属性 ###########################################
    # ##################################################################################################
    # category 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
    print("        %s feature_days_from_1st_last_behavior_category...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_days_from_1st_last_behavior_category(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_days_from_1st_last_behavior_category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # # 在 [begin_date, checking_date) 期间， 各个 behavior 在 category 上的总次数, 平均每天的点击数以及方差
    # # 返回 12 个特征
    print("        %s feature_beahvior_cnt_on_category...   \r" % (getCurrentTime()), end="")
    category_behavior_cnt = []
    time_start = time.clock()
    for i, pre_days in enumerate(pre_days_list):
        feature_mat, feature_cnt = feature_beahvior_cnt_on_category(pre_days, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
        category_behavior_cnt.append(feature_mat[:, 0:4]) # 取出 各个 behavior 在 item 上的总次数
    time_end = time.clock()

    print("        %s feature_beahvior_cnt_on_category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # [begin date, end date) 期间，总共有多少用户购买了该 category
    print("        %s how many users buy category...   \r" % (getCurrentTime()), end="")
    time_start = time.clock()
    feature_mat, feature_cnt = feature_how_many_users_bought_category(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    user_cnt_buy_category = feature_mat
    print("        %s how many users buy category takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    # [begin date, end date) 期间， category 的销量
    print("        %s category sales volume...   \r" % (getCurrentTime()), end="")
    feature_mat, feature_cnt = feature_category_sals_volume(window_start_date, window_end_date, samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    category_sales_vol = feature_mat
    print("        %s category sales volume takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    ##################################################################################################
    #######################################  交叉特征  ################################################
    ##################################################################################################
    
    # item 的销量占 category 的销量的比例
    print("        %s ratio of sales volume, item/category   \r" % (getCurrentTime()), end="")    
    feature_mat, feature_cnt = feature_sales_ratio_itme_category(item_sales_vol, category_sales_vol, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    print("        %s ratio of sales volume, item/category, total featurs %d   \r" % (getCurrentTime(), total_feature_cnt))

    # 购买 item 的用户数量占购买 category 的用户数的比例
    print("        %s ratio of user cnt buy item/category   \r" % (getCurrentTime()), end="")    
    feature_mat, feature_cnt = feature_buyer_ratio_item_category(user_cnt_buy_item, user_cnt_buy_category, cal_feature_importance, final_feature_importance, total_feature_cnt)
    total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    print("        %s ratio of user cnt buy item/category, total featurs %d   \r" % (getCurrentTime(), total_feature_cnt))
    
    print("        %s category behavior count...   \r" % (getCurrentTime()), end="")
    item_behavior_cnt = []
    time_start = time.clock()
    for i, pre_days in enumerate(pre_days_list):
        # 各个 behavior 在 item 上的总次数, 平均每天的点击数, 方差以及用户在item上behavior的次数占总次数的比例
        # 返回 16 个特征
        feature_mat, feature_cnt = feature_beahvior_cnt_on_item(pre_days, window_end_date, user_behavior_cnt_on_item[i],
                                                                samples, cal_feature_importance, final_feature_importance, total_feature_cnt)
        total_feature_cnt += addSubFeatureMatIntoFeatureMat(feature_mat, feature_cnt, Xmat, total_feature_cnt)
    time_end = time.clock()
    print("        %s feature_beahvior_cnt_on_item takes %d seconds, total featurs %d   \r" % (getCurrentTime(), time_end - time_start, total_feature_cnt), end="")

    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    print("        %s Total features %d   \r" % (getCurrentTime(), total_feature_cnt))

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
def GBDT_slideWindows(window_start_date, window_end_date, cal_feature_importance, final_feature_importance, n_estimators):
    nag_per_pos = 10
    print("%s slide windows from %s to %s   \r" % (getCurrentTime(), window_start_date, window_end_date))

    print("        %s taking samples for training (%s, %d)   \r" % (getCurrentTime(), window_end_date, nag_per_pos))
    samples, Ymat = takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos)
    print("        %s samples count %d, Ymat count %d   \r" % (getCurrentTime(), len(samples), len(Ymat)))

    Xmat = createTrainingSet(window_start_date, window_end_date, nag_per_pos, samples, cal_feature_importance, final_feature_importance)
    if (len(samples) == 0):
        print("%s No buy records from %s to %s, returning...   \r" % (getCurrentTime(), window_start_date, window_end_date))
        return None

    Xmat = preprocessing.scale(Xmat)
    m, n = np.shape(Xmat)

    # Xmat, Ymat = shuffle(Xmat, Ymat, random_state=13)

    params = {'n_estimators': n_estimators, 
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

    # if (cal_feature_importance):
    #     plotDeviance(window_start_date, window_end_date, nag_per_pos, params, clf)

    return clf