from common import *
from LR_common import *
import random

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品
def takingSamples(checking_date, nag_per_pos, item_popularity_dict):
    samples = []
    Ymat = []
    index = 0

    logging.info("taking samples for date %s, nagetive samples per positive samples %d" % (checking_date, nag_per_pos))

    total_positive = 0
    totoal_nagetive = 0
    positive_samples = takingPositiveSamples(checking_date)
    for user_item in positive_samples:
        samples.append(user_item)
        Ymat.append(1)

    nagetive_samples = takingNagetiveSamples(checking_date, positive_samples, nag_per_pos, item_popularity_dict)
    for user_item in nagetive_samples:
        samples.append(user_item)
        Ymat.append(0)

    return samples, Ymat

def takingPositiveSamples(checking_date):
    positive_samples = set()
    buy_cnt = len(g_user_buy_transection)
    index = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                if (each_record[-1][0] != BEHAVIOR_TYPE_BUY):
                    logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                    continue

                if (each_record[-1][1].date() != checking_date):
                    continue

                positive_samples.add((user_id, item_id))
        index += 1
        if (index % 10000 == 0):
            print("                postive samples %d / %d\r " % (index, buy_cnt), end="")

    #只采样出现在 test set 中的 item
    items_in_test_set = filterItemByTestset(positive_samples)
    positive_samples_in_validate = list(items_in_test_set)
    logging.info("positive smaple %d, in validate %d" % (len(positive_samples), len(positive_samples_in_validate)))
    print("%s positive smaple %d, %d are in validate" % (getCurrentTime(), len(positive_samples), len(positive_samples_in_validate)))
    return positive_samples_in_validate

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
            nagetive_sample_candidates[(user_id, item_id)] = 1
        index += 1
        if (index % 100000 == 0):
            print("                 %d / %d patterns checked\r" % (index, pattern_count), end="")

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
            print("                 %d / %d filtered\r" % (index, nagetive_cnt), end="")

    print("         %s filtering nagetive samples by test set..." % getCurrentTime())
    nagetive_samples_in_test = filterItemByTestset(candidates_in_test_set)
    print("         %s total %d nagetive samples in test set..." % (getCurrentTime(), len(nagetive_samples_in_test)))

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
    print("         %s calculating posbility range of nagetive records according to item popularity..." % getCurrentTime())
    nagetive_records_cnt = len(nagetive_samples_in_test)
    index = 0
    # 计算在各个 beahvior 上的 popularity 的总和以及各个 range 所对应的 user_item
    for user_item in nagetive_samples_in_test:
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

    logging.info("no user operated item count %d" % no_user_opted_item)
    logging.info("total popularity %s" % (total_popularity_dict))
    #if (len(nagetive_samples_pop_range_dict[1]) <= 500):
    logging.info("nagetive_samples_pop_range[BEHAVIOR_TYPE_BUY] %s" % nagetive_samples_pop_range_dict[4])

    # 按照计算好的 popularity 进行采样
    # 1. 随机取一个 behavior type
    # 2. 在 behavior type 的 total popularity 范围内取随机值
    # 3. 随机值落在哪个 sample 的 pop range 范围内
    nagetive_samples = set()
    print("        %s taking nagetive samples according to item popularity..." % getCurrentTime())
    nagetive_records_cnt = len(positive_samples) * nag_per_pos
    for index in range(nagetive_records_cnt):
        # 1. 随机取一个 behavior type
        rand_behavior = random.randint(1, 4)

        # 2. 在 behavior type 的 total popularity 范围内取随机值
        rand_pop = random.randint(0, total_popularity_dict[rand_behavior] - 1)
        for pop_range, user_item in nagetive_samples_pop_range_dict[rand_behavior].items():
            if (pop_range[0] <= rand_pop and rand_pop < pop_range[1]):
                nagetive_samples.add(user_item)
                break
        print("                %d / %d taken\r" % (index, nagetive_records_cnt), end="")

    print("")
    print("%s nagetive samples %d " % (getCurrentTime(), len(nagetive_samples)))
    logging.info("takingNagetiveSamples %d " % len(nagetive_samples))
    return nagetive_samples

def filterItemByTestset(user_item_id_set):
    items_in_test_set = set()
    for user_item in user_item_id_set:
        if (user_item[1] in global_train_item_category):
            items_in_test_set.add(user_item)

    return items_in_test_set
