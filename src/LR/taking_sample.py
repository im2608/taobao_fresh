from common import *
from LR_common import *

# positive_samples_cnt_per_user 每个用户取得正样本的数量
# nag_per_pos 正负样本比例，一个正样本对应 nag_per_pos 个负样本
# 根据 checking_date 进行采样
# 正样本为用户在 checking_date 这一天购买过的商品，
# 负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品

def takingSamples(positive_samples_cnt_per_user, checking_date, nag_per_pos, item_popularity):
    samples = []
    Ymat = []
    index = 0

    logging.info("taking samples, positive could be %d, nagetive could be %d" % \
                (positive_samples_cnt_per_user, positive_samples_cnt_per_user * nag_per_pos))

    #在购物记录中根据 checking date 采集正样本
    for user_id in g_user_buy_transection:        
        item_list_user_bought = getPositiveSamplesListByUser(checking_date, user_id, positive_samples_cnt_per_user)

        actual_pos = len(item_list_user_bought)
        for i in range(actual_pos):
            samples.append((user_id, item_list_user_bought[i]))
            Ymat.append(1)

        #负样本为用户操作过但是没有购买的商品，或是购买过但不是在 checking_date 这一天购买的商品
        nagetive_cnt = actual_pos * nag_per_pos
        item_list_user_opted = getNagetiveSampleListByUser(checking_date, user_id, nagetive_cnt, item_list_user_bought)
        actual_nag = len(item_list_user_opted)
        for i in range(actual_nag):
            samples.append((user_id, item_list_user_opted[i]))
            Ymat.append(0)

        logging.info("%s acutal positive %d, acutal nagetive %d" % (user_id, actual_pos, actual_nag))

    return samples, Ymat

# 对用户操作过的item 采样, sample_cnt 采集多少个样本
# TODO: 这里需要增强，按照item的热度来采样
def getPositiveSamplesListByUser(checking_date, user_id, sample_cnt):
    item_list_user_bought = set()
    #得到符合条件的item list
    for item_id, item_buy_records in g_user_buy_transection[user_id].items():
        for each_record in item_buy_records:
            buy_behavior = each_record[-1]
            if (buy_behavior[0] != BEHAVIOR_TYPE_BUY):
                logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                continue

            if (buy_behavior[1].date() != checking_date):
                continue

            item_list_user_bought.add(item_id)

    #只采样出现在 test set 中的 item
    items_in_test_set = filterItemByTestset(item_list_user_bought)
    item_list_user_bought = list(items_in_test_set)

    return item_list_user_bought[0 : sample_cnt]
    
def getNagetiveSampleListByUser(checking_date, user_id, sample_cnt, positive_samples):
    nagetive_sample_candidates = []
    #在购物记录中采样在checking date这一天没有购买的商品
    for item_id, item_buy_records in g_user_buy_transection[user_id].items():
        for each_record in item_buy_records:
            buy_behavior = each_record[-1]
            if (buy_behavior[0] != BEHAVIOR_TYPE_BUY):
                logging.error("Buy record of %s %s is not Buy" % (user_id, item_id))
                continue

            if (buy_behavior[1].date() == checking_date):
                continue

            nagetive_sample_candidates.append(item_id)

    #在pattern 中采样用户浏览过的item
    if (user_id in g_user_behavior_patten):
        nagetive_sample_candidates.extend(list(g_user_behavior_patten[user_id].keys()))

    candidates_in_test_set = set()

    # 如果用户购买了某产品， 然后又浏览该商品，则该商品会出现在 pattern 中，
    # 为了防止正/负样本同时采样到该商品，这里做过滤
    for item_id in nagetive_sample_candidates:
        if (item_id in positive_samples):
            logging.warn("%s: pos/nag take %s as sample at same time, skip!" % (user_id, item_id))
        else:
            candidates_in_test_set.add(item_id)

    nagetive_samples = filterItemByTestset(candidates_in_test_set)

    return list(nagetive_samples)[: sample_cnt]


def filterItemByTestset(item_id_set):
    items_in_test_set = set()
    for item_id in item_id_set:
        item_category = getCatalogByItemId(item_id)
        if (item_category is not None):
            items_in_test_set.add(item_id)

    return items_in_test_set
