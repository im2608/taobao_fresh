from common import *
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品
def takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos):
    # samples = set()
    samples = []
    Ymat = []
    index = 0
    total_positive = 0
    totoal_nagetive = 0
    positive_users = set()

    positive_samples = takingPositiveSamplesOnDate(window_start_date, window_end_date)
    for user_item in positive_samples:
        samples.append(user_item)
        positive_users.add(user_item[0])
        Ymat.append(1)
        total_positive += 1

    nagetive_samples = takeNagetiveSamplesByUserActivity(window_start_date, window_end_date, positive_samples, total_positive* nag_per_pos)
    for user_item in nagetive_samples:
        samples.append(user_item)
        Ymat.append(0)
        totoal_nagetive += 1

    # nagetive_samples = takingNagetiveSamples(window_start_date, window_end_date, positive_samples, nag_per_pos, item_popularity_dict)
    # for user_item in nagetive_samples:
    #     samples.append(user_item)
    #     Ymat.append(0)
    #     totoal_nagetive += 1

    print("        %s %s - %s, postive samples %d, nagetive samples %d" % 
         (getCurrentTime(), window_start_date, window_end_date, total_positive, totoal_nagetive))

    return list(samples), Ymat

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

def takingPositiveSamplesOnDate(window_start_date, buy_date):
    print("        %s taking positive samples on (%s)" % (getCurrentTime(), buy_date))
    positive_samples = set()
    buy_cnt = len(g_user_buy_transection)
    index = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                # 当天浏览， 当天购买的记录不作为正样本
                if (each_record[-1][1].date() == buy_date and each_record[0][1].date() != buy_date):
                     #and item_id in global_test_item_category):
                    positive_samples.add((user_id, item_id))
        index += 1

    positive_samples_in_validate = list(positive_samples)
    print("        %s positive smaple %d" % (getCurrentTime(), len(positive_samples)))
    return positive_samples_in_validate



#user 在 item 上的某条记录是否能作为负样本
def shouldTakeNagetiveSample(window_start_date, window_end_date, user_id, item_id, from_test_set):
    # 用户从未购买过任何物品则不能作为负样本
    # item 不在测试集中不能作为负样本
    if (user_id not in g_user_buy_transection):
        return False

    if (from_test_set and item_id not in global_test_item_category):
        return False

    # item 是否被 收藏 or 购物车, 若item只被浏览过则不能作为负样本
    for each_record in g_user_behavior_patten[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() < window_end_date and
                #behavior_consecutive[1].date() >= window_start_date and
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


def takingNagetiveSamples(window_start_date, window_end_date, positive_samples, nag_per_pos, item_popularity_dict):
    nagetive_sample_candidates = dict()

    buy_cnt = len(g_user_buy_transection)
    index = 0

    for user_id, item_pattern_records in g_user_behavior_patten.items():
        for item_id in item_pattern_records:
            if (shouldTakeNagetiveSample(window_start_date, window_end_date, user_id, item_id, False)):
                nagetive_sample_candidates[(user_id, item_id)] = 1

    popularity_range_dict = dict()
    total_popularity = 0
    pop_range_start = 0
    pop_range_end = 0
    print("                %s calculating popularities of items in nagetive samples %d " % 
         (getCurrentTime(), len(nagetive_sample_candidates)))
    for user_item in nagetive_sample_candidates:
        #去掉 range 中的小数, 并计算每个 user_item 的 popularity range
        pop_range_end = round(pop_range_start + item_popularity_dict[user_item[1]] * 100)
        popularity_range_dict[(pop_range_start, pop_range_end)] = user_item
        pop_range_start = pop_range_end
        total_popularity += item_popularity_dict[user_item[1]]

    total_popularity = round(total_popularity *100)

    nagetive_samples = set()

    sorted_pop_ranges = list(popularity_range_dict.keys())
    sorted_pop_ranges.sort()

    print("                %s total popularity %d" % (getCurrentTime(), total_popularity))
    print("                %s taking nagetive samples..." % getCurrentTime())
    nagetive_sample_cnt = len(positive_samples) * nag_per_pos
    for index in range(nagetive_sample_cnt):
        # 在 total popularity 范围内取随机值
        rand_pop = random.randint(0, total_popularity - 1)
        pop_range_index = takeNagetiveSampleByPopularity(sorted_pop_ranges, rand_pop)
        pop_range = sorted_pop_ranges[pop_range_index]
        nagetive_samples.add(popularity_range_dict[pop_range])
        print("                %d / %d nagetive samples taken\r" % (index, nagetive_sample_cnt), end="")

    return nagetive_samples


# from_test_set 表示是否只是从test set 中采样
def takingSamplesForForecasting(window_start_date, forecasting_date, from_test_set):
    samples_for_forecasting = dict()

    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            if (from_test_set and item_id not in global_test_item_category):
                continue
            for each_record in buy_records:
                if (each_record[-1][1].date() == forecasting_date):
                    samples_for_forecasting[(user_id, item_id)] = 1
    print("        %s takingSamplesForForecasting postive samples %d " % (getCurrentTime(), len(samples_for_forecasting)))

    #从测试集中取样与在训练集中采样负样本的逻辑相同
    for user_id, item_pattern_records in g_user_behavior_patten.items():
        for item_id in item_pattern_records:
            if (shouldTakeNagetiveSample(window_start_date, forecasting_date, user_id, item_id, from_test_set)):
                samples_for_forecasting[(user_id, item_id)] = 0

    print("        %s taking %d sample from testing set" % (getCurrentTime(), len(samples_for_forecasting)))
    samples = list(samples_for_forecasting.keys())
    Ymat = []
    for each_sample in samples:
        Ymat.append(samples_for_forecasting[each_sample])

    return samples, Ymat




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

def takeSamples(window_start_date, window_end_date, only_from_test_set):
    Ymat = []
    samples = set()
    params = {'window_start_date' : window_start_date,
              'window_end_date' : window_end_date,
              'samples' : samples, 
              'user_records' : g_user_buy_transection,
              'only_from_test_set' : only_from_test_set
    }
    takeSamplesByUserBehavior(**params)

    params = {'window_start_date' : window_start_date,
              'window_end_date' : window_end_date,
              'samples' : samples, 
              'user_records' : g_user_behavior_patten,
              'only_from_test_set' : only_from_test_set
    }
    takeSamplesByUserBehavior(**params)

    samples = list(samples)
    positive_samples = 0
    nagetive_samples = 0
    for user_item in samples:
        user_id = user_item[0]
        item_id = user_item[1]

        if (user_id not in g_user_buy_transection or 
            item_id not in g_user_buy_transection[user_id]):
            Ymat.append(0)
            nagetive_samples += 1
            continue

        added_in_positive = False
        for each_record in g_user_buy_transection[user_id][item_id]:
            if (each_record[-1][1].date() == window_end_date):
                Ymat.append(1)
                positive_samples += 1
                added_in_positive = True
                break
        if (not added_in_positive):
            Ymat.append(0)
            nagetive_samples += 1

    print("        %s taking sample (%s, %s), total %d samles, (%d, 1), (%d, 0)" %        
          (getCurrentTime(), window_start_date, window_end_date, len(samples), positive_samples, nagetive_samples))

    return samples, Ymat



# 选择样本规则：
# 1. user 在 end date 前一天有过操作的item
# 2. user 在 [begin date, end date 前两天] 有过非浏览操作的item
# 3. only_from_test_set 表示是否只是从测试集中选择样本
def takeSamplesByUserBehavior(window_start_date, window_end_date, samples, user_records, only_from_test_set):
    day_before_end_date = window_end_date - datetime.timedelta(1)
    for user_id, user_operation_records in user_records.items():
        for item_id, item_pattern_record in user_operation_records.items():
            added_in_sample = False
            for each_record in item_pattern_record:
                if (added_in_sample):
                    break
                for behavior in each_record:
                    if (behavior[1].date() == day_before_end_date
                        or
                        (behavior[1].date() >= window_start_date and behavior[1].date() < day_before_end_date and 
                        behavior[0] != BEHAVIOR_TYPE_VIEW)):
                        if (only_from_test_set):
                            if (item_id in global_test_item_category):
                                samples.add((user_id, item_id))
                                added_in_sample = True
                                break
                        else:
                            samples.add((user_id, item_id))
                            added_in_sample = True
                            break

################################################################################################################
################################################################################################################
################################################################################################################
# 根据用户的活跃度进行负采样
# 用户的活跃度为用户在item上的活跃度之和
# 用户活跃度/Sigma(所有用户活跃度的和) 为每个用户的负样本数 N
# 按照用户在item上的活跃度降序排序， 取前N个作为用户的负样本
def takeNagetiveSamplesByUserActivity(window_start_date, window_end_date, positive_samples, total_nagetive_cnt):
    nagetive_samples = []

    user_item_pairs = getUserItemPairsByUserBehavior(window_start_date, window_end_date, positive_samples)

    user_activity_score_dict, total_activity = calculateUserActivity(window_start_date, window_end_date, user_item_pairs)
    for user_id, user_act_dict in user_activity_score_dict.items():
        nagetive_cnt_of_user = round(user_act_dict["activity"] / total_activity * total_nagetive_cnt + 1)
        # logging.info("user %s's nagetive samples: %d " % (user_id, nagetive_cnt_of_user))
        if (nagetive_cnt_of_user > len(user_act_dict["activity_on_item"])):
            nagetive_cnt_of_user = len(user_act_dict["activity_on_item"])

        for i, item_score in enumerate(user_act_dict["activity_on_item"]):
            if (i == nagetive_cnt_of_user):
                break
            nagetive_samples.append((user_id, item_score[0]))

    logging.info("takeNagetiveSamplesByUserActivity nagetive samples %d" % len(nagetive_samples))

    return nagetive_samples

def calculateUserActivity(window_start_date, window_end_date, user_item_pairs):
    user_activity_score_dict = dict()
    total_activity = 0.0


    for user_item in user_item_pairs:
        user_id = user_item[0]
        item_id = user_item[1]

        if (user_id not in user_activity_score_dict):
            user_activity_score_dict[user_id] = dict()
            user_activity_score_dict[user_id]["activity_on_item"] = []
            user_activity_score_dict[user_id]["activity"] = 0            

        user_activiey_on_item = 0
        for each_record in g_user_behavior_patten[user_id][item_id]:
            for behavior in each_record:
                if (behavior[1].date() >= window_start_date and 
                    behavior[1].date() < window_end_date):
                    days = (window_end_date - behavior[1].date()).days
                    buy_prob = g_prob_bwteen_1st_days_and_buy[days]
                    # 用户在item上的行为次数 * 行为的权值 * 行为日至购买日之前的天数导致购买的可能性 = 用户在item上的分数， 分数越高，用户对item越感兴趣
                    user_activiey_on_item += behavior[2] * g_behavior_weight[behavior[0]] * buy_prob

        user_activity_score_dict[user_id]["activity_on_item"].append((item_id, user_activiey_on_item))
        user_activity_score_dict[user_id]["activity"] += user_activiey_on_item
        total_activity += user_activiey_on_item

    # 按照用户对item的分数从高到低排序
    for user_id, item_score in user_activity_score_dict.items():
        user_activity_score_dict[user_id]["activity_on_item"] = sorted(item_score["activity_on_item"], key=lambda item:item[1], reverse=True)
        # logging.info("user activity %s %.2f activity %s" % 
        #              (user_id, 
        #               user_activity_score_dict[user_id]["activity"], 
        #               user_activity_score_dict[user_id]["activity_on_item"]))

    return user_activity_score_dict, total_activity