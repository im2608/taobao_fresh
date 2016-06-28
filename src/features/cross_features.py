from common import *
import numpy as np
from feature_selection import *

########################################################################################################
########################################################################################################
########################################################################################################
# item 的销量占 category 的销量的比例
def feature_sales_ratio_itme_category(item_sales_vol, category_sales_vol, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    if (item_sales_vol is None or category_sales_vol is None):
        return None, 0

    feature_name = "feature_sales_ratio_itme_category"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    ratio_list = np.zeros((len(item_sales_vol), 1))
   
    for index in range(len(item_sales_vol)):
        if (category_sales_vol[index] == 0):
            ratio_list[index] = 0
        else:
            ratio_list[index] = np.round(item_sales_vol[index] / category_sales_vol[index], 4)

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    # logging.info(" ratio_list is %s" % ratio_list)

    logging.info("leaving feature_category_sals_volume")

    return ratio_list, 1



########################################################################################################
########################################################################################################
########################################################################################################
# item  在各个behavior上的次数占 category 上各个behavior次数的比例
def feature_behavior_cnt_itme_category(item_behavior_cnt, category_behavior_cnt, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    if (item_behavior_cnt is None or category_behavior_cnt is None):
        return None, 0

    feature_name = "feature_behavior_cnt_itme_category"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    ratio_list = np.zeros((len(item_behavior_cnt), 1))
   
    for index in range(len(item_behavior_cnt)):
        for behavior in range(4):
            if (category_behavior_cnt[index, behavior] > 0):
                ratio_list[index] = np.round(item_behavior_cnt[index, behavior] / category_behavior_cnt[index, behavior], 4)

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt        

    # logging.info(" ratio_list is %s" % ratio_list)

    logging.info("leaving feature_behavior_cnt_itme_category")

    return ratio_list, 1


########################################################################################################
########################################################################################################
########################################################################################################
# 购买 item 的用户数量占购买 category 的用户数的比例
def feature_buyer_ratio_item_category(user_cnt_buy_item, user_cnt_buy_category, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    if (user_cnt_buy_item is None or  user_cnt_buy_category is None):
        return None, 0

    feature_name = "feature_buyer_ratio_itme_category"
    if (not cal_feature_importance and final_feature_importance[g_feature_info[feature_name]] == 0):
        logging.info("%s has no useful features" % feature_name)
        return None, 0

    ratio_list = np.zeros((len(user_cnt_buy_item), 1))
   
    for index in range(len(user_cnt_buy_item)):
        if (user_cnt_buy_category[index] == 0):
            ratio_list[index] = 0
        else:
            ratio_list[index] = np.round(user_cnt_buy_item[index] / user_cnt_buy_category[index], 4)

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    # logging.info(" ratio_list is %s" % ratio_list)

    logging.info("leaving feature_category_sals_volume")

    return ratio_list, 1

########################################################################################################
########################################################################################################
########################################################################################################
# item 的1st, last behavior 与 category 的1st， last 相差的天数
def feature_1st_last_between_item_category(days_first_last_item, days_first_last_category, cal_feature_importance, final_feature_importance, cur_total_feature_cnt):
    if (days_first_last_item is None or days_first_last_category is None):
        return None, 0

    features_names = ["feature_1st_between_item_category_view",
                      "feature_1st_between_item_category_fav",
                      "feature_1st_between_item_category_cart",
                      "feature_1st_between_item_category_buy",
                      "feature_last_between_item_category_view",
                      "feature_last_between_item_category_fav",
                      "feature_last_between_item_category_cart",
                      "feature_last_between_item_category_buy"]

    if (not cal_feature_importance):
        useful_features = featuresForForecasting(features_names, final_feature_importance)
        if (len(useful_features) == 0):
            logging.info("During forecasting, [feature_1st_last_between_item_category] has no useful features")
            return None, 0
        else:
            logging.info("During forecasting, [feature_1st_last_between_item_category] has %d useful features" % len(useful_features))

    days_1st_last_difference_list = np.zeros((len(days_first_last_item), len(features_names)))
    # 1st 的天数用 category 减去 item， category 的1st天数等于或早于item
    days_1st_last_difference_list[:, 0:4] = days_first_last_category[:, 0:4] - days_first_last_item[:, 0:4]

    # last 的天数用 item 减去 category， category 的last天数等于或晚于item
    days_1st_last_difference_list[:, 4:7] = days_first_last_item[:, 4:7] - days_first_last_category[:, 4:7]
   
    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    logging.info("feature_1st_last_between_item_category %s" % days_1st_last_difference_list)

    logging.info("leaving feature_1st_last_between_item_category")

    return ratio_list, 1

