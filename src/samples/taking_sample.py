from common import *
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品

def takingPositiveSamplesInPeriord(window_start_date, window_end_date):
    print("        %s taking positive samples(%s, %s)" % (getCurrentTime(), window_start_date, window_end_date))
    logging.info("taking positive samples(%s, %s)" % (window_start_date, window_end_date))
    positive_samples = set()
    buy_cnt = len(g_user_buy_transection)
    index = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                if (each_record[-1][1].date() < window_end_date and 
                    each_record[0][1].date() >= window_start_date and
                    item_id in global_test_item_category):
                    positive_samples.add((user_id, item_id))
        index += 1

    positive_samples_in_validate = list(positive_samples)
    print("        %s positive smaple %d" % (getCurrentTime(), len(positive_samples)))
    return positive_samples_in_validate

def takingPositiveSamplesOnDate(buy_date, during_verifying):
    print("        %s taking positive samples on (%s)" % (getCurrentTime(), buy_date))
    positive_samples = set()
    buy_cnt = len(g_user_buy_transection)
    index = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:                
                if (each_record[-1][1].date() == buy_date):
                    # 若是为了验证而采集正样本， 则当天的购买记录也可作为正样本
                    if (during_verifying):
                        positive_samples.add((user_id, item_id))
                    else:
                        # 若是为了训练而采集正样本,则当天浏览，当天购买的记录不作为正样本
                        if (each_record[0][1].date() != buy_date):
                             #and item_id in global_test_item_category):
                            positive_samples.add((user_id, item_id))
                        else:
                            logging.info("user %s first(%s) operated %s and buy(%s) on same day, skip!" % (user_id,
                                         each_record[0][1].date(), item_id, each_record[-1][1]))
        index += 1

    # positive_samples_in_validate = list(positive_samples)
    print("        %s positive smaple %d" % (getCurrentTime(), len(positive_samples)))
    return positive_samples



#user 在 item 上的某条记录是否能作为样本
def shouldTakeAsSample(window_start_date, window_end_date, user_id, item_id, during_forecasting, user_records):
    # 用户从未购买过任何物品则不能作为样本    
    if (user_id not in g_user_buy_transection):
        return False

    # item 不在测试集中不能作为样本
    if (during_forecasting and item_id not in global_test_item_category):
        return False

    # item 是否被 收藏 or 购物车, 若item只被浏览过则不能作为样本
    for each_record in user_records[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() < window_end_date and
                # behavior_consecutive[1].date() >= window_start_date and
                behavior_consecutive[0] != BEHAVIOR_TYPE_VIEW):
                return True

    return False

def takeNagetiveSampleByPopularity(popularity_range_list, rand_popularity):
    left = 0
    right = len(popularity_range_list) - 1
    try:           
        while (left <= right):
            mid = int((left + right) / 2)

            if (rand_popularity >= popularity_range_list[mid][0] and 
                rand_popularity < popularity_range_list[mid][1]):
                return mid
            if (rand_popularity < popularity_range_list[mid][0]):
                right = mid - 1
            else:
                left = mid + 1
    except IndexError as ex:
        logging.info("popularity_range_list %d" % mid)
        for index in range(len(popularity_range_list)):
            logging.info("%d ==> %s" % (index, popularity_range_list[index]))
        raise ex

    return -1




def calculate_item_popularity_by_records(user_records, item_popularity_dict, item_category_opt_cnt_dict):
    for user_id, item_opt_records in user_records.items():
        for item_id, records in item_opt_records.items():

            item_category = global_train_item_category[item_id]

            if (item_category not in item_category_opt_cnt_dict):
                item_category_opt_cnt_dict[item_category] = dict()

            if (item_id not in item_popularity_dict):
                item_popularity_dict[item_id] = dict()

            for each_record in records:
                for each_behavior in each_record:
                    behavior_type = each_behavior[0]
                    if (behavior_type not in item_category_opt_cnt_dict[item_category]):
                        item_category_opt_cnt_dict[item_category][behavior_type] = 0
                    item_category_opt_cnt_dict[item_category][behavior_type] += each_behavior[2]

                    if (behavior_type not in item_popularity_dict[item_id]):
                        item_popularity_dict[item_id][behavior_type] = 0
                    item_popularity_dict[item_id][behavior_type] += each_behavior[2]
    return 0

# 每个 item 在各个 behavior type 上的热度
# item 在各个 behavior type 上的次数/category 在各个 behavior type 上的次数
def calculate_item_popularity():
    #每个 item 在各个 behavior type 上的热度
    item_popularity_dict = dict()

    #在 category 上进行过各个 behavior type 操作的次数
    item_category_opt_cnt_dict = dict()

    calculate_item_popularity_by_records(g_user_buy_transection, item_popularity_dict, item_category_opt_cnt_dict)
    calculate_item_popularity_by_records(g_user_behavior_patten, item_popularity_dict, item_category_opt_cnt_dict)

    for item_id in item_popularity_dict:
        item_category = global_train_item_category[item_id]
        for behavior_type in [BEHAVIOR_TYPE_VIEW, BEHAVIOR_TYPE_FAV, BEHAVIOR_TYPE_CART, BEHAVIOR_TYPE_BUY]:
            if (behavior_type in item_popularity_dict[item_id]):
                item_popularity_dict[item_id][behavior_type] = \
                    round(item_popularity_dict[item_id][behavior_type]/item_category_opt_cnt_dict[item_category][behavior_type], 6)
            else:
                item_popularity_dict[item_id][behavior_type] = 0

    if (len(item_popularity_dict) <= 500 ):
        logging.info("item popularity %s" % item_popularity_dict)
    return item_popularity_dict


def getBehaviorCnt(user_records, window_start_date, window_end_date, item_behavior_cnt_dict):
    for user_id, item_opt_records in user_records.items():
        for item_id, records in item_opt_records.items():
            if (item_id not in item_behavior_cnt_dict):
                item_behavior_cnt_dict[item_id] = [0, 0, 0, 0]

            for each_record in records:
                for behavior_consecutive in each_record:
                    if (behavior_consecutive[1].date() >= window_start_date and
                        behavior_consecutive[1].date() < window_end_date):
                        item_behavior_cnt_dict[item_id][behavior_consecutive[0] - 1] += behavior_consecutive[2]

#热度： [window_start_date, window_end_date) 窗口内： （点击数*0.01+购买数*0.94+购物车数*0.47+收藏数*0.33）
def calculateItemPopularity(window_start_date, window_end_date):
    logging.info("calculateItemPopularity.. ")
    item_popularity_dict = dict()
    item_behavior_cnt_dict = dict()
    item_popularity_in_category_dict = dict()

    getBehaviorCnt(g_user_buy_transection, window_start_date, window_end_date, item_behavior_cnt_dict)
    getBehaviorCnt(g_user_behavior_patten, window_start_date, window_end_date, item_behavior_cnt_dict)
    index = 0
    total_items = len(item_behavior_cnt_dict)
    for item_id, behavior_cnt in item_behavior_cnt_dict.items():
        popularity = behavior_cnt[0]*0.01 + behavior_cnt[1]*0.33 + behavior_cnt[2]*0.47 + behavior_cnt[3]*0.94
        item_popularity_dict[item_id] = popularity
        
        #计算每个item在category中的权值
        item_category = global_train_item_category[item_id]        
        if (item_category not in item_popularity_in_category_dict):
            item_popularity_in_category_dict[item_category] = []            
        item_popularity_in_category_dict[item_category].append((item_id, popularity))


        # logging.info("%s to %s, %s popularity %s ==> %.1f" % 
        #             (window_start_date, window_end_date, item_id, behavior_cnt, popularity))

        index += 1
        if (index % 1000 == 0):
            print("                %d / %d popularity calculated\r" % (index, total_items), end="")

    for category, item_pop in item_popularity_in_category_dict.items():
        item_popularity_in_category_dict[category] = sorted(item_pop, key=lambda item:item[1], reverse=True)

    logging.info("leaving calculateItemPopularity")

    return item_popularity_dict#, item_popularity_in_category_dict

################################################################################################################
################################################################################################################
################################################################################################################
def loadSavedSamples(window_start_date, window_end_date, start_from, user_cnt):
    Ymat = []
    samples = []
    
    samples_file = "%s\\..\log\\sample_(%s,%s)(%d,%d).txt" % (runningPath, window_start_date, window_end_date, start_from, user_cnt)
    Ymat_file = "%s\\..\log\\Ymat_(%s,%s)(%d,%d).txt" % (runningPath, window_start_date, window_end_date, start_from, user_cnt)

    filehandle = open(samples_file, encoding="utf-8", mode='r')

    for line in filehandle:
        user_item = line.strip("\n").split(" ")          
        samples.append((user_item[0], user_item[1]))

    filehandle.close()

    Ymat_file = "%s\\..\log\\11.27 Ymat_%s_%s.txt" % (runningPath, window_start_date, window_end_date)

    filehandle = open(Ymat_file, encoding="utf-8", mode='r')
    for line in filehandle:
        Ymat.append(int(line))

    print("        %s loadSavedSamples, total %d / %d" % (getCurrentTime(), len(samples), len(Ymat)))

    return samples, Ymat

def takeSamples(window_start_date, window_end_date, nag_per_pos, during_forecasting, start_from, user_cnt):

    # return loadSavedSamples(window_start_date, window_end_date)

    Ymat = []
    samples = []

    positive_samples = takingPositiveSamplesOnDate(window_end_date, False)
    # np.savetxt("%s\\..\log\\positive_sample_%s_%s.txt" % (runningPath, window_start_date, window_end_date), list(positive_samples), fmt="%s", newline="\n")

    Ymat.extend([1 for x in range(len(positive_samples))])

    params = {'window_start_date' : window_start_date,
              'window_end_date' : window_end_date,
              'user_records' : g_user_behavior_patten,
              'during_forecasting' : during_forecasting
    }
    nagetive_samples_pattern = takeSamplesByUserBehavior(**params)

    params = {'window_start_date' : window_start_date,
              'window_end_date' : window_end_date,
              'user_records' : g_user_buy_transection,
              'during_forecasting' : during_forecasting
    }
    nagetive_samples_buy = takeSamplesByUserBehavior(**params)

    nagetive_samples = nagetive_samples_pattern.union(nagetive_samples_buy)
    print("        %s take samples by user behavior %d" % (getCurrentTime(), len(nagetive_samples)))

    # 从负样本集中去掉 在正样本集中出现的样本
    tmp = nagetive_samples.union(positive_samples) ^ (positive_samples ^ nagetive_samples)
    nagetive_samples = nagetive_samples ^ tmp
    print("        %s after removing samples that in positive and nagetive %d" % (getCurrentTime(), len(nagetive_samples)))

    # 在训练过程中，在负样本集中根据用户的活跃度进行采样
    if (not during_forecasting):
        nagetive_samples = takeSamplesByUserActivity(window_start_date, window_end_date, nagetive_samples, nag_per_pos * len(positive_samples))
        print("        %s taking samples by user activity %d / %d" % (getCurrentTime(), len(nagetive_samples), nag_per_pos * len(positive_samples)))

    Ymat.extend([0 for x in range(len(nagetive_samples))])

    # samples = list(positive_samples.union(nagetive_samples))
    samples.extend(list(positive_samples))
    samples.extend(list(nagetive_samples))

    print("        %s takeSamples (%s, %s), %d / %d, total %d / %d" % (getCurrentTime(), window_start_date, window_end_date, 
          len(positive_samples), len(nagetive_samples), len(samples), len(Ymat)))

    # samples_file = "%s\\..\log\\sample_(%s,%s)(%d,%d).txt" % (runningPath, window_start_date, window_end_date, start_from, user_cnt)
    # Ymat_file = "%s\\..\log\\Ymat_(%s,%s)(%d,%d).txt" % (runningPath, window_start_date, window_end_date, start_from, user_cnt)
    # np.savetxt(samples_file, samples, fmt="%s", newline="\n")
    # np.savetxt(Ymat_file, Ymat, fmt="%d", newline="\n")

    return samples, Ymat



# 选择样本规则：
# 1. user 在 end date 前一天有过操作的item
# 2. user 在 [begin date, end date 前两天] 有过非浏览操作的item
# 3. during_forecasting 表示是否只是从测试集中选择样本
def takeSamplesByUserBehavior(window_start_date, window_end_date, user_records, during_forecasting):
    samples = set()
    day_before_end_date = window_end_date - datetime.timedelta(1)
    for user_id, user_operation_records in user_records.items():
        for item_id, item_pattern_record in user_operation_records.items():

            if (not shouldTakeAsSample(window_start_date, window_end_date, user_id, item_id, during_forecasting, user_records)):
                continue

            added_in_sample = False
            for each_record in item_pattern_record:
                if (added_in_sample):
                    break
                for behavior in each_record:
                    if (behavior[1].date() == day_before_end_date
                        or
                        (behavior[1].date() >= window_start_date and behavior[1].date() < day_before_end_date and 
                         behavior[0] != BEHAVIOR_TYPE_VIEW)):
                        if (during_forecasting):
                            if (item_id in global_test_item_category):
                                samples.add((user_id, item_id))
                                added_in_sample = True
                                break
                        else:
                            samples.add((user_id, item_id))
                            added_in_sample = True
                            break
    return samples

################################################################################################################
################################################################################################################
################################################################################################################
# 根据用户的活跃度进行采样
# 用户的活跃度为用户在item上的活跃度之和
# 用户活跃度/Sigma(所有用户活跃度的和) 为每个用户的负样本数 N
# 按照用户在item上的活跃度降序排序， 取前N个作为用户的负样本
def takeSamplesByUserActivity(window_start_date, window_end_date, nagetive_samples, total_nagetive_cnt):
    samples = set()

    user_activity_score_dict = dict()
    calculateUserActivity(window_start_date, window_end_date, g_user_behavior_patten, user_activity_score_dict, nagetive_samples)
    for user_id, user_act_dict in user_activity_score_dict.items():
        if (user_id == "total_activity"):
            continue

        nagetive_cnt_of_user = round(user_act_dict["activity"] / user_activity_score_dict["total_activity"] * total_nagetive_cnt)
        # logging.info("user %s has %d nagetive_samples" % (user_id, nagetive_cnt_of_user))
        if (nagetive_cnt_of_user > len(user_act_dict["activity_on_item"])):
            nagetive_cnt_of_user = len(user_act_dict["activity_on_item"])

        for i, item_score in enumerate(user_act_dict["activity_on_item"]):
            if (i == nagetive_cnt_of_user):
                break
            samples.add((user_id, item_score[0]))

    logging.info("takeNagetiveSamplesByUserActivity nagetive samples %d" % len(samples))

    return samples
