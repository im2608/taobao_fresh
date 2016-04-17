from common import *
from LR_common import *
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from features import *
from taking_sample import *




###############################################################################################################
###############################################################################################################
###############################################################################################################

def createTrainingSet(checking_date, positive_samples_cnt_per_user, nag_per_pos, samples, item_popularity_dict):
    #预先分配足够的features
    feature_cnt = 100
    Xmat = np.mat(np.zeros((len(samples), feature_cnt)))

    feature_cnt = 0

    ##################################################################################################
    #######################################商品属性####################################################
    ##################################################################################################
    # 商品热度 购买该商品的用户/总用户数
    print("%s getting item popularity by samples..." % getCurrentTime())
    
    Xmat[:, feature_cnt] = feature_item_popularity(item_popularity_dict, samples); feature_cnt += 1
    print("%s Total %d samples taken" % (getCurrentTime(), len(samples)))
    logging.info("%s Total %d samples taken" % (getCurrentTime(), len(samples)))

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

    # 得到用户在某商品上的购买浏览转化率 购买过的某商品数量/浏览过的商品数量
    print("%s calculating user-item b/v ratio..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_buy_view_ratio(samples); feature_cnt += 1

    #最后一次操作同类型的商品至 checking_date 的天数的倒数
    print("%s calculating last buy..." % getCurrentTime())
    Xmat[:, feature_cnt] = feature_last_opt_category(g_user_buy_transection, checking_date, BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(g_user_buy_transection, checking_date, BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_category(g_user_buy_transection, checking_date, BEHAVIOR_TYPE_CART, samples); feature_cnt += 1
     Xmat[:, feature_cnt] = feature_last_opt_category(g_user_buy_transection, checking_date, BEHAVIOR_TYPE_BUY, samples); feature_cnt += 1

    #用户一共购买过多少同类型的商品 * 最后一天购买至 checking_date 的天数的倒数
    Xmat[:, feature_cnt] = feature_how_many_buy(Xmat[:, 4], checking_date, samples); feature_cnt += 1

    # 用户在过去 30 天对 item id 各个behavior type 的次数 / 用户相应的behavior tpye 的次数    
    pre_days = 30
    print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    # 用户在过去 7 天对 item id 各个behavior type 的次数 / 用户相应的behavior tpye 的次数    
    pre_days = 7
    print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    # 用户在过去 3 天对 item id 各个behavior type 的次数 / 用户相应的behavior tpye 的次数    
    pre_days = 3
    print("%s get behavior ratios...%d days" % (getCurrentTime(), pre_days))
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_VIEW, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_FAV, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_CART, pre_days, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_user_item_behavior_ratio(checking_date, BEHAVIOR_TYPE_BUY, pre_days, samples); feature_cnt += 1

    #用户第一次购买前的各个 behavior 数
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_behavior_cnt_before_1st_buy(BEHAVIOR_TYPE_CART, samples); feature_cnt += 1

    #最后一次操作 item 至 checking_date 的天数的倒数
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_VIEW, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_FAV, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_CART, samples); feature_cnt += 1
    Xmat[:, feature_cnt] = feature_last_opt_item(checking_date, BEHAVIOR_TYPE_BUY, samples); feature_cnt += 1



    ##################################################################################################
    #######################################    feature end  ##########################################
    ##################################################################################################

    print("Total features %d" % feature_cnt)


    #去掉没有用到的features
    Xmat = Xmat[:, 0:feature_cnt]

    m, n = np.shape(Xmat)
    print("shape of Xmap (%d, %d)" % (m, n))
    logging.info("shape of Xmap (%d, %d" % (m, n))
    Xmat = np.mat(Xmat)
    logging.info(Xmat)

    return Xmat

def logisticRegression(user_cnt):    
    userBehaviorStatisticOnRecords(g_user_buy_transection)
    userBehaviorStatisticOnRecords(g_user_behavior_patten)

    logging.info("user behavior count:")
    logging.info(g_user_behavior_count)

    positive_samples_cnt_per_user = 10
    nag_per_pos = 5
    checking_date = datetime.datetime.strptime("2014-12-17", "%Y-%m-%d").date()
    print("%s checking date %s" % (getCurrentTime(), checking_date))

    #item 的热度
    print("%s calculating popularity..." % getCurrentTime())
    item_popularity_dict = calculate_item_popularity()

    print("%s taking samples..." % getCurrentTime())
    samples, Ymat = takingSamples(positive_samples_cnt_per_user, checking_date, nag_per_pos, item_popularity_dict)
    print("samples count %d, Ymat count %d" % (len(samples), len(Ymat)))
    logging.info("samples %s" % samples)
    logging.info("Ymat %s" % Ymat)

    Xmat = createTrainingSet(checking_date, positive_samples_cnt_per_user, nag_per_pos, samples, item_popularity_dict)

    min_max_scaler = preprocessing.MinMaxScaler()
    Xmat_scaler = min_max_scaler.fit_transform(Xmat)

    logging.info(Xmat_scaler)

    model = LogisticRegression()
    model.fit(Xmat_scaler, Ymat)
    expected = Ymat
    predicted = model.predict(Xmat_scaler)
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

    forecast_date = datetime.datetime.strptime("2014-12-18", "%Y-%m-%d").date()
    Xmat_forecast = createTrainingSet(forecast_date, positive_samples_cnt_per_user, nag_per_pos, samples, item_popularity_dict)
    logging.info("================================")
    logging.info(Xmat_forecast)

    Xmat_forecast_scaler =min_max_scaler.fit_transform(Xmat_forecast)
    predicted = model.predict(Xmat_forecast_scaler)

    output_file_name = "%s\\..\\output\\forecast.LR.%d.csv" % (runningPath, user_cnt)

    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    outputFile.write("user_id,item_id\n")
    for index in range(len(predicted)):
        if (predicted[index] == 1):
            outputFile.write("%s,%s\r" % (samples[index][0], samples[index][1]))

    outputFile.close()

    return 0

def outputXY(Xmat, Ymat, samples):
    xm, xn = np.shape(Xmat)
    ym = len(Ymat)
    sample_cnt = len(samples)
    if (xm != ym):
        logging.error("ERROR: lines %d of Xmat != lines %d of Ymat" % (xm, ym))
        return
    mat_row_string = []
    for row_idx in range(xm):
        mat_row_string.append("[")
        for feature_cnt in range(xn):
            mat_row_string.append("%.2f " % Xmat[row_idx, feature_cnt])
        mat_row_string.append("] = %d %s\n" % (Ymat[row_idx], samples[row_idx]))

    logging.info(mat_row_string)



