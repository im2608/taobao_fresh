from common import *
from LR_common import *
import random

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品
def takingSamples(checking_date, nag_per_pos):
    samples = []
    Ymat = []
    index = 0

    logging.info("taking samples for date %s, nagetive samples per positive samples %d" % (checking_date, nag_per_pos))

    total_positive = 0
    totoal_nagetive = 0
    positive_samples, items_in_samples = takingPositiveSamples(checking_date)
    for user_item in positive_samples:
        samples.append(user_item)
        Ymat.append(1)

    nagetive_samples, item_popularity_dict = takingNagetiveSamples(checking_date, positive_samples, nag_per_pos, items_in_samples)
    for user_item in nagetive_samples:
        samples.append(user_item)
        Ymat.append(0)

    return samples, Ymat, item_popularity_dict

def takingPositiveSamples(checking_date):
    items_in_samples = set()
    positive_samples = set()
    buy_cnt = len(g_user_buy_transection)
    index = 0
    total_positive_cnt = 0
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            for each_record in buy_records:
                if (each_record[-1][0] != BEHAVIOR_TYPE_BUY):
                    logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                    continue

                if (each_record[-1][1].date() == checking_date and \
                    item_id in global_test_item_category):
                    positive_samples.add((user_id, item_id))
                    items_in_samples.add(item_id)

                total_positive_cnt += 1
        index += 1
        if (index % 10000 == 0):
            print("                postive samples %d / %d\r " % (index, buy_cnt), end="")

    positive_samples_in_validate = list(positive_samples)

    print("%s took positive smaple %d / %d" % (getCurrentTime(), len(positive_samples_in_validate), total_positive_cnt))
    return positive_samples_in_validate, items_in_samples

#user 在 item 上的某条记录是否能作为负样本
def shouldTakeNagetiveSample(checking_date, user_id, item_id):
    # item 不在测试集中则不采样
    # 用户从未购买过任何物品则不采样
    if (item_id not in global_test_item_category or 
        user_id not in g_user_buy_transection):
        return False

    # item 是否被 收藏 or 购物车
    for each_record in g_user_behavior_patten[user_id][item_id]:
        for behavior_consecutive in each_record:
            if (behavior_consecutive[1].date() < checking_date and\
                behavior_consecutive[0] != BEHAVIOR_TYPE_VIEW):
                return True

    #logging.info("user %s item %s has only viewed." % (user_id, item_id))

    return False


def takingNagetiveSamples(checking_date, positive_samples, nag_per_pos, items_in_samples):
    nagetive_sample_candidates = dict()

    buy_cnt = len(g_user_buy_transection)
    index = 0    
    #在购物记录中得到在checking date这一天没有购买的商品作为负样本
    for user_id, item_buy_records in g_user_buy_transection.items():
        for item_id, buy_records in item_buy_records.items():
            if (item_id not in global_test_item_category):
                continue

            for each_record in buy_records:
                buy_behavior = each_record[-1]
                if (buy_behavior[0] != BEHAVIOR_TYPE_BUY):
                    logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                    continue

                if (buy_behavior[1].date() >= checking_date):
                    continue

                nagetive_sample_candidates[(user_id, item_id)] = 1
                items_in_samples.add(item_id)
        index += 1
        print("        nagetive samples not bought on checking date %d / %d\r " % (index, buy_cnt), end="")

    for user_id, item_pattern_records in g_user_behavior_patten.items():
        for item_id in item_pattern_records:
            if (shouldTakeNagetiveSample(checking_date, user_id, item_id)):
                nagetive_sample_candidates[(user_id, item_id)] = 1
                items_in_samples.add(item_id)

    print("        %s calculating popularities of items in nagetive samples %d " % (getCurrentTime(), len(nagetive_sample_candidates)))
    item_popularity_dict = calculateItemPopularity(checking_date, items_in_samples)

    popularity_range_dict = dict()
    total_popularity = 0
    pop_range_start = 0
    pop_range_end = 0
    for user_item in nagetive_sample_candidates:
        #去掉 range 中的小数, 并计算每个 user_item 的 pop range
        pop_range_end = round(pop_range_start + item_popularity_dict[user_item[1]] * 100)
        popularity_range_dict[user_item] = (pop_range_start, pop_range_end)
        pop_range_start = pop_range_end
        total_popularity += item_popularity_dict[user_item[1]]

    #logging.info("popularity_range_dict %s" % popularity_range_dict)

    total_popularity = round(total_popularity *100)

    nagetive_samples = set()

    print("        %s total popularity %d" % (getCurrentTime(), total_popularity))
    print("        %s taking nagetive samples..." % getCurrentTime())

    nagetive_sample_cnt = len(positive_samples) * nag_per_pos
    for index in range(nagetive_sample_cnt):
        # 在 total popularity 范围内取随机值
        rand_pop = random.randint(0, total_popularity - 1)
        for user_item, pop_range in popularity_range_dict.items():
            if (pop_range[0] <= rand_pop and rand_pop < pop_range[1]):
                nagetive_samples.add(user_item)
                break

        print("        %d / %d nagetive samples taken\r" % (index, nagetive_sample_cnt), end="")

    #nagetive_samples = nagetive_sample_candidates

    return nagetive_samples, item_popularity_dict