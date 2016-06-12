from common import *
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品
def takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos, item_popularity_dict):
    samples = []
    Ymat = []
    index = 0
    total_positive = 0
    totoal_nagetive = 0
    positive_samples = takingPositiveSamples(window_end_date)
    for user_item in positive_samples:
        samples.append(user_item)
        Ymat.append(1)

    nagetive_samples = takingNagetiveSamples2(window_start_date, window_end_date, positive_samples, nag_per_pos, item_popularity_dict)
    for user_item in nagetive_samples:
        samples.append(user_item)
        Ymat.append(0)

    return samples, Ymat

def takingPositiveSamples(buying_date):
    print("        %s taking positive samples(%s)" % (getCurrentTime(), buying_date))
    positive_samples = set()
    buy_cnt = len(g_user_buy_transection)
    index = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                if (each_record[-1][1].date() == buying_date and 
                    #each_record[0][1].date() >= window_start_date and
                    item_id in global_test_item_category):
                    positive_samples.add((user_id, item_id))
        index += 1

    positive_samples_in_validate = list(positive_samples)
    print("        %s positive smaple %d" % (getCurrentTime(), len(positive_samples)))
    return positive_samples_in_validate


#user 在 item 上的某条记录是否能作为负样本
def shouldTakeNagetiveSample(window_start_date, window_end_date, user_id, item_id):
    # 用户从未购买过任何物品则不能作为负样本
    # item 不在测试集中不能作为负样本
    if (user_id not in g_user_buy_transection or 
        item_id not in global_test_item_category):
        return False

    # item 是否被 收藏 or 购物车, 若item只被浏览过则不能作为负样本
    for each_record in g_user_behavior_patten[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() < window_end_date and
                #behavior_consecutive[1].date() >= window_start_date and
                behavior_consecutive[0] != BEHAVIOR_TYPE_VIEW):
                return True

    return False


#根据商品热度进行有放回采样， 商品热度作为采样的概率， 所以 item id 相同的 samples 会有相同的采样概率
def takingNagetiveSamples(checking_date, positive_samples, nag_per_pos, item_popularity_dict):
    nagetive_sample_candidates = dict()
    #在购物记录中采样在checking date这一天没有购买的商品
    buy_cnt = len(g_user_buy_transection)
    index = 0

    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                buy_behavior = each_record[-1]
                if (buy_behavior[0] != BEHAVIOR_TYPE_BUY):
                    logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                    continue

                if (buy_behavior[1].date() == checking_date):
                    continue

                nagetive_sample_candidates[(user_id, item_id)] = 1
        index += 1
        print("        nagetive samples not bought on checking date %d / %d\r " % (index, buy_cnt), end="")


    #在pattern 中采样用户浏览过的item
    index = 0
    print("        %s getting all nagetive records from pattern..." % getCurrentTime())
    pattern_count = len(g_user_behavior_patten)
    index = 0
    for user_id, item_pattern_record in g_user_behavior_patten.items():
        for item_id in item_pattern_record:
            if (shouldTakeNagetiveSample(checking_date, user_id, item_id)):
                nagetive_sample_candidates[(user_id, item_id)] = 1
        index += 1
        if (index % 100000 == 0):
            print("         %d / %d patterns checked\r" % (index, pattern_count), end="")

    candidates_in_test_set = set()

    # 如果用户购买了某产品， 然后又浏览该商品，则该商品会出现在 pattern 中，
    # 为了防止正/负样本同时采样到该商品，这里去掉负样本
    index = 0
    print("         %s filtering pos/nag samples..." % getCurrentTime())
    nagetive_cnt = len(nagetive_sample_candidates)
    for user_item in nagetive_sample_candidates:
        if (user_item in positive_samples):
            logging.warn("%s: pos/nag take %s as sample at same time, skip!" % (user_item[0], user_item[1]))
        else:
            candidates_in_test_set.add(user_item)
        index += 1
        if (index % 100000 == 0):
            print("                 %s %d / %d filtered\r" % (getCurrentTime(), index, nagetive_cnt), end="")

    #商品热度精确到小数点后5 位
    # range 取值为 [pop_range_start, pop_range_end), 半开区间
    total_popularity_dict = dict()
    total_popularity_dict[BEHAVIOR_TYPE_VIEW] = 0.0
    total_popularity_dict[BEHAVIOR_TYPE_FAV] = 0.0
    total_popularity_dict[BEHAVIOR_TYPE_CART] = 0.0
    total_popularity_dict[BEHAVIOR_TYPE_BUY] = 0.0

    pop_range_start_dict = dict()
    pop_range_start_dict[BEHAVIOR_TYPE_VIEW] = 0
    pop_range_start_dict[BEHAVIOR_TYPE_FAV] = 0
    pop_range_start_dict[BEHAVIOR_TYPE_CART] = 0
    pop_range_start_dict[BEHAVIOR_TYPE_BUY] = 0

    pop_range_end_dict = dict()
    pop_range_end_dict[BEHAVIOR_TYPE_VIEW] = 0
    pop_range_end_dict[BEHAVIOR_TYPE_FAV] = 0
    pop_range_end_dict[BEHAVIOR_TYPE_CART] = 0
    pop_range_end_dict[BEHAVIOR_TYPE_BUY] = 0

    # 记录各个 behavior type 在不同的 range 上对应的 sample
    # nagetive_samples_pop_range_dict[behavior_type][(range start, range end)] 得到的是 (user id, item id)
    nagetive_samples_pop_range_dict = dict()
    nagetive_samples_pop_range_dict[BEHAVIOR_TYPE_VIEW] = dict()
    nagetive_samples_pop_range_dict[BEHAVIOR_TYPE_FAV] = dict()
    nagetive_samples_pop_range_dict[BEHAVIOR_TYPE_CART] = dict()
    nagetive_samples_pop_range_dict[BEHAVIOR_TYPE_BUY] = dict()

    no_user_opted_item = 0
    print("                 %s calculating posbility range of nagetive records according to item popularity..." % getCurrentTime())
    nagetive_records_cnt = len(candidates_in_test_set)
    index = 0
    for user_item in candidates_in_test_set:
        item_id = user_item[1]
        #没有用户操作过该产品，则 item_popularity_dict 中就没有该 item 
        if (item_id not in item_popularity_dict):
            logging.info("no user operated item %s " % item_id)
            no_user_opted_item += 1
            continue

        for behavior_index in range(1, 5):
            if (behavior_index not in item_popularity_dict[item_id]):
                continue
            # popularity range 从小数变成整数
            total_popularity_dict[behavior_index] += item_popularity_dict[item_id][behavior_index]
            pop_range_end_dict[behavior_index] = pop_range_start_dict[behavior_index] + \
                                                 int(item_popularity_dict[item_id][behavior_index] * (10**5))
            nagetive_samples_pop_range_dict[behavior_index]\
                                           [(pop_range_start_dict[behavior_index], pop_range_end_dict[behavior_index])] = user_item

            logging.info("poplarity (%s, %s) on %d ==> [%d, %d)" % \
                         (user_item[0], user_item[1], behavior_index, \
                          pop_range_start_dict[behavior_index], pop_range_end_dict[behavior_index]))

            pop_range_start_dict[behavior_index] = pop_range_end_dict[behavior_index]
        index += 1
        if (index % 10000 == 0):
            print("                %d / %d nagetive records calculated\r" % (index, nagetive_records_cnt), end="")
    
    for behavior_type in total_popularity_dict:
        total_popularity_dict[behavior_type] =  int(total_popularity_dict[behavior_type] * 10**5)

    print("                 %s no user operated item count %d" % (getCurrentTime(), no_user_opted_item))
    logging.info("total popularity %s" % (total_popularity_dict))
    logging.info("nagetive_samples_pop_range[BEHAVIOR_TYPE_BUY] %s" % nagetive_samples_pop_range_dict[4])

    sorted_pop_range = dict()
    for behavior_type, popularity_range_dict in nagetive_samples_pop_range_dict.items():
        pop_range_list = list(popularity_range_dict.keys())
        pop_range_list.sort()
        sorted_pop_range[behavior_type] = pop_range_list

    # 按照计算好的 popularity 进行采样
    # 1. 随机取一个 behavior type
    # 2. 在 behavior type 的 total popularity 范围内取随机值
    # 3. 随机值落在哪个 sample 的 popularity range 范围内
    nagetive_samples = set()

    nagetive_records_cnt = len(positive_samples) * nag_per_pos
    print("                %s taking nagetive samples according to item popularity, %d are going to take..." %\
          (getCurrentTime(), nagetive_records_cnt))

    while (len(nagetive_samples) <= nagetive_records_cnt):
        # 1. 随机取一个 behavior type
        rand_behavior = random.randint(1, 4)

        # 2. 在 behavior type 的 total popularity 范围内取随机值
        rand_pop = random.randint(0, total_popularity_dict[rand_behavior] - 1)
        # for pop_range, user_item in nagetive_samples_pop_range_dict[rand_behavior].items():
        #     if (pop_range[0] <= rand_pop and rand_pop < pop_range[1]):
        #         nagetive_samples.add(user_item)
        #         break
        pop_range_index = takeNagetiveSampleByPopularity(sorted_pop_range[rand_behavior], rand_pop)
        pop_range = sorted_pop_range[rand_behavior][pop_range_index]
        nagetive_samples.add(nagetive_samples_pop_range_dict[rand_behavior][pop_range])

        print("                %d / %d taken\r" % (index, nagetive_records_cnt), end="")

    return nagetive_samples


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


def takingNagetiveSamples2(window_start_date, window_end_date, positive_samples, nag_per_pos, item_popularity_dict):
    nagetive_sample_candidates = dict()

    buy_cnt = len(g_user_buy_transection)
    index = 0
    # 在购物记录中得到在 window_end_date 这一天没有购买的商品作为负样本
    # for user_id, item_buy_records in g_user_buy_transection.items():
    #     for item_id, buy_records in item_buy_records.items():
    #         if (item_id not in global_test_item_category):
    #             continue

    #         for each_record in buy_records:
    #             buy_behavior = each_record[-1]
    #             if (buy_behavior[1].date() >= window_end_date):
    #                 continue

    #             nagetive_sample_candidates[(user_id, item_id)] = 1                
    #     index += 1
    #     print("        nagetive samples not bought on checking date %d / %d\r " % (index, buy_cnt), end="")

    for user_id, item_pattern_records in g_user_behavior_patten.items():
        for item_id in item_pattern_records:
            if (shouldTakeNagetiveSample(window_start_date, window_end_date, user_id, item_id)):
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

def takingSamplesForForecasting(window_start_date, forecasting_date):
    samples_for_testing = dict()
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            if (item_id not in global_test_item_category):
                continue
            for each_record in buy_records:
                if (each_record[-1][1].date() == forecasting_date):
                    samples_for_testing[(user_id, item_id)] = 1

    #从测试集中取样与在训练集中采样负样本的逻辑相同
    for user_id, item_pattern_records in g_user_behavior_patten.items():
        for item_id in item_pattern_records:
            if (shouldTakeNagetiveSample(window_start_date, forecasting_date, user_id, item_id)):
                samples_for_testing[(user_id, item_id)] = 1

    print("        %s taking %d sample from testing set" % (getCurrentTime(), len(samples_for_testing)))
    return list(samples_for_testing.keys())





def plotDeviance(window_start_date, window_end_date, nag_per_pos, params, clf):
    window_start_date = window_start_date + datetime.timedelta(1)
    window_end_date = window_end_date + datetime.timedelta(1)
    item_popularity_dict = calculateItemPopularity(window_start_date, window_end_date)
    X_test, Y_test = takingSamplesForTraining(window_start_date, window_end_date, nag_per_pos, item_popularity_dict)
    mse = mean_squared_error(Y_test, clf.predict(X_test))
    print("        %s MSE: %.4f" % (getCurrentTime(), mse))
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()