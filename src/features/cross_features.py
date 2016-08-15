from common import *
import numpy as np
from feature_selection import *

########################################################################################################
########################################################################################################
########################################################################################################
# item 的销量占 category 的销量的比例, 以及item 销量在category销量中的排序
def feature_sales_ratio_itme_category(item_sales_vol, category_sales_vol, user_item_pairs):

    ratio_list = np.zeros((len(item_sales_vol), 2))

    itmes_sale_vol_in_category = dict()

    #记录下 item 在 sample 中的位置，以方便的设置销量的排序
    item_idx_in_samples = dict()    
   
    for index in range(len(item_sales_vol)):
        if (category_sales_vol[index] == 0):
            ratio_list[index, 0] = 0
        else:
            ratio_list[index, 0] = np.round(item_sales_vol[index] / category_sales_vol[index], 4)

        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]
        item_category = global_train_item_category[item_id]
        if (item_category not in itmes_sale_vol_in_category):
            itmes_sale_vol_in_category[item_category] = dict()

        itmes_sale_vol_in_category[item_category][item_id] = item_sales_vol[index]

        if (item_id not in item_idx_in_samples):
            item_idx_in_samples[item_id] = []

        item_idx_in_samples[item_id].append(index)

    for item_category in itmes_sale_vol_in_category:
        sorted_item_sal_vol = sorted(itmes_sale_vol_in_category[item_category].items(), key=lambda item:item[1], reverse=True)
        sorted_rank = getRankFromSortedTuple(sorted_item_sal_vol)
        for i, rank in enumerate(sorted_rank):
            item_id = sorted_item_sal_vol[i][0]
            for index in item_idx_in_samples[item_id]:
                ratio_list[index, 1] = rank

        # logging.info("category %s sorted_item_sal_vol %s" % (item_category, sorted_item_sal_vol))

    # logging.info(" ratio_list is %s" % ratio_list)

    logging.info("oneHotEncodeRank feature_category_sals_volume ratio_list[:, 1] %d" % np.max(ratio_list[:, 1]))

    # rank_onehot = oneHotEncodeRank(ratio_list[:, 1])
    # feature_mat = np.column_stack((ratio_list[:, 0], rank_onehot, preprocessing.scale(item_sales_vol), preprocessing.scale(category_sales_vol)))
    feature_mat = np.column_stack((ratio_list, preprocessing.scale(item_sales_vol), preprocessing.scale(category_sales_vol)))

    logging.info("leaving feature_category_sals_volume returns feature count %d" % feature_mat.shape[1])

    return feature_mat



########################################################################################################
########################################################################################################
########################################################################################################
# item  在各个behavior上的次数占 category 上各个behavior次数的比例
def feature_behavior_cnt_itme_category(item_behavior_cnt, category_behavior_cnt):
    ratio_list = np.zeros((len(item_behavior_cnt), 1))
   
    for index in range(len(item_behavior_cnt)):
        for behavior in range(4):
            if (category_behavior_cnt[index, behavior] > 0):
                ratio_list[index] = np.round(item_behavior_cnt[index, behavior] / category_behavior_cnt[index, behavior], 4)

    if (cal_feature_importance):
        g_feature_info[feature_name] = cur_total_feature_cnt

    # logging.info(" ratio_list is %s" % ratio_list)

    logging.info("leaving feature_behavior_cnt_itme_category")

    return ratio_list


########################################################################################################
########################################################################################################
########################################################################################################
# 购买 item 的用户数量占购买 category 的用户数的比例, 以及购买item的用户数在购买category内其他item的用户数中的排序
def feature_buyer_ratio_item_category(user_cnt_buy_item, user_cnt_buy_category, user_item_pairs):
    ratio_list = np.zeros((len(user_cnt_buy_item), 2))

    item_buyer_in_category = dict()

    #记录下 item 在 sample 中的位置，以方便的设置用户数量的排序
    item_idx_in_samples = dict()    

    for index in range(len(user_cnt_buy_item)):
        if (user_cnt_buy_category[index] == 0):
            ratio_list[index, 0] = 0
        else:
            ratio_list[index, 0] = np.round(user_cnt_buy_item[index] / user_cnt_buy_category[index], 4)

        item_id = user_item_pairs[index][1]

        # 统计在同一个category中购买各个item的用户数
        item_category = global_train_item_category[item_id]
        if (item_category not in item_buyer_in_category):
            item_buyer_in_category[item_category] = dict()
        item_buyer_in_category[item_category][item_id] = user_cnt_buy_item[index]

        if (item_id not in item_idx_in_samples):
            item_idx_in_samples[item_id] = []
        item_idx_in_samples[item_id].append(index)

    # 对于购买人数的item，rank也应该相同 
    for item_category in item_buyer_in_category:
        sorted_item_buyer_cnt = sorted(item_buyer_in_category[item_category].items(), key=lambda item:item[1], reverse=True)
        rank = 1
        cur_buyer_cnt = sorted_item_buyer_cnt[0][1]
        for item_id_buyer_cnt in sorted_item_buyer_cnt:
            item_id = item_id_buyer_cnt[0]
            if (item_id_buyer_cnt[1] < cur_buyer_cnt):
                rank += 1
                cur_buyer_cnt = item_id_buyer_cnt[1]

            for index in item_idx_in_samples[item_id]:
                ratio_list[index, 1] = rank

    ratio_list[:, 0] = preprocessing.scale(ratio_list[:, 0])

    logging.info("oneHotEncodeRank feature_category_sals_volume ratio_list[:, 1] %d" % np.max(ratio_list[:, 1]))

    # rank_onehot = oneHotEncodeRank(ratio_list[:, 1])
    # feature_mat = np.column_stack((ratio_list[:, 0], rank_onehot, preprocessing.scale(user_cnt_buy_item), preprocessing.scale(user_cnt_buy_category)))
    feature_mat = np.column_stack((ratio_list, preprocessing.scale(user_cnt_buy_item), preprocessing.scale(user_cnt_buy_category)))

    # logging.info(" ratio_list is %s" % ratio_list)

    logging.info("leaving feature_category_sals_volume, return feature cnt %d" % feature_mat.shape[1])

    return feature_mat

########################################################################################################
########################################################################################################
########################################################################################################
# item 的1st, last behavior 与 category 的1st， last 相差的天数
def feature_1st_last_between_item_category(days_first_last_item, days_first_last_category):

    days_1st_last_difference_list = np.zeros((len(days_first_last_item), 8))
    # 1st 的天数用 category 减去 item， category 的1st天数等于或早于item
    days_1st_last_difference_list[:, 0:4] = days_first_last_category[:, 0:4] - days_first_last_item[:, 0:4]

    # last 的天数用 item 减去 category， category 的last天数等于或晚于item
    days_1st_last_difference_list[:, 4:7] = days_first_last_item[:, 4:7] - days_first_last_category[:, 4:7]
   
    days_1st_last_difference_list = preprocessing.scale(days_1st_last_difference_list)

    logging.info("leaving feature_1st_last_between_item_category")

    return days_1st_last_difference_list

########################################################################################################
########################################################################################################
########################################################################################################

# 用户在 item 上各个行为的加权值在用户所有操作过的item 上的排序
# 用户在 item 上各个行为的加权值在用户对同 category 上操作过的item 上的排序
# 用户在 category 上行为的加权值在用户对所有操作过的category 上的排序
def feature_item_category_weight_rank(window_start_date, window_end_date, user_item_pairs):
    logging.info("feature_item_category_weight_rank (%s, %s)" % (window_start_date, window_end_date))

    # 用户在 item 上各个行为的加权值在用户所有操作过的item 上的排序
    user_item_act_rank = dict()

    # 用户在 item 上各个行为的加权值在用户对同 category 上操作过的item 上的排序
    user_item_category_act_rank = dict()

    # 用户在 category 上行为的加权值在用户对所有操作过的category 上的排序
    user_category_act_rank = dict()

    # 得到用户对所操作过的item的加权值
    user_activity_score_dict = dict()
    calculateUserActivity(window_start_date, window_end_date, g_user_buy_transection, user_activity_score_dict,  user_item_pairs)
    calculateUserActivity(window_start_date, window_end_date, g_user_behavior_patten, user_activity_score_dict,  user_item_pairs)

    for user_id in user_activity_score_dict:
        if (user_id == "total_activity"):
            continue

        user_item_act_rank[user_id] = dict()
        user_item_category_act_rank[user_id] = dict()

        items_user_opted = user_activity_score_dict[user_id]["activity_on_item"]
        sorted_item_act = getRankFromSortedTuple(items_user_opted)
        for i, rank in enumerate(sorted_item_act):
            item_id = items_user_opted[i][0]

            # 用户在 item 上各个行为的加权值在用户所有操作过的item 上的排序
            user_item_act_rank[user_id][item_id] = rank

            item_category = global_train_item_category[item_id]
            if (item_category not in user_item_category_act_rank[user_id]):
                user_item_category_act_rank[user_id][item_category] = dict()
                user_item_category_act_rank[user_id][item_category]["activity_on_item"] = []
                user_item_category_act_rank[user_id][item_category]["activity"] = 0

            # 用户在category上操作过的item
            user_item_category_act_rank[user_id][item_category]["activity_on_item"].append(items_user_opted[i])
            user_item_category_act_rank[user_id][item_category]["activity"] += items_user_opted[i][1]

    for user_id in user_item_category_act_rank:
        user_category_act_rank[user_id] = dict()

        for item_category in user_item_category_act_rank[user_id]:
            items_user_opted_in_category = user_item_category_act_rank[user_id][item_category]["activity_on_item"]
            sorted_item_act_in_category = sorted(items_user_opted_in_category, key=lambda item:item[1], reverse=True)

            # 用户在category上操作过的item 的排序
            sorted_item_act = getRankFromSortedTuple(sorted_item_act_in_category)

            user_item_category_act_rank[user_id][item_category]["item_rank"] = dict()

            for i, rank in enumerate(sorted_item_act):
                item_id = sorted_item_act_in_category[i][0]
                user_item_category_act_rank[user_id][item_category]["item_rank"][item_id] = rank

            # 用户在 category 上行为的加权值在用户对所有操作过的category 上的排序
            sorted_category_act_rank = sorted(user_item_category_act_rank[user_id].items(), key=lambda item:item[1]["activity"], reverse=True)
            rank = 1
            cur_actvity_val = sorted_category_act_rank[0][1]["activity"]
            for category_act in sorted_category_act_rank:
                item_category = category_act[0]
                if (cur_actvity_val > category_act[1]["activity"]):
                    cur_actvity_val = category_act[1]["activity"]
                    rank += 1
                user_category_act_rank[user_id][item_category] = rank

    item_category_weight_rank = np.zeros((len(user_item_pairs), 3))

    for index in range(len(user_item_pairs)):
        user_id = user_item_pairs[index][0]
        item_id = user_item_pairs[index][1]

        # 用户在 item 上各个行为的加权值在用户所有操作过的item 上的排序
        item_category_weight_rank[index, 0] = user_item_act_rank[user_id][item_id]

        # 用户在 item 上各个行为的加权值在用户对同 category 上操作过的item 上的排序    
        item_category = global_train_item_category[item_id]
        item_category_weight_rank[index, 1] = user_item_category_act_rank[user_id][item_category]["item_rank"][item_id]

        # 用户在 category 上行为的加权值在用户对所有操作过的category 上的排序
        item_category_weight_rank[index, 2] = user_category_act_rank[user_id][item_category]

    rank_onehot_enc = OneHotEncoder()

    rank_onehot = rank_onehot_enc.fit_transform(item_category_weight_rank)

    logging.info("feature_item_category_weight_rank returns feature count %d" % rank_onehot.shape[1])

    return rank_onehot