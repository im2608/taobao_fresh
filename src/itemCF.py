
from common import *
import math

global_itemCatalogSimilarities = dict()

#不考虑权值, Inverse User Frequence
#这里计算的是两个 item category 的相似度而不是两个item的相似度，因为 test 中的 item 不是都出现在train中
def itemSimilarity_IUF():
	logging.info("%s itemSimilarity_IUF..." % getCurrentTime())

	user_idx = 0
	user_cnt = len(global_user_item_dict)

	for (user_id, user_operations) in global_user_item_dict.items():
		user_idx += 1

		#用户操作了多少个 item categories
		user_operations_cnt = len(user_operations)
		print("%s itemCF: user %s [%d/%d] operated %d item categories" % \
			  (getCurrentTime(), user_id, user_idx, user_cnt, user_operations_cnt))

		categories_user_opted = list(user_operations.keys())

		user_opt_idx1 = 0
		while (user_opt_idx1 < user_operations_cnt):

			user_opt_idx2 = user_opt_idx1 + 1
			while (user_opt_idx2 < user_operations_cnt):
				item1_cate = categories_user_opted[user_opt_idx1]
				item2_cate = categories_user_opted[user_opt_idx2]

				user_opt_idx2 += 1

				# 查看是否已经计算过 category1, category2 的相似度，若没有，则将sim 插入到
				# global_itemCatalogSimilarities[key1][key2] 的位置
				# key1, key2 分别为 < cat1, cat2> or <cat2, cat1>
				key1, key2 = getPosOfDoubleHash(item1_cate, item2_cate, global_itemCatalogSimilarities)
				if (key1 == None):
					continue

				#在 item1 category and item2 category 上有过操作的用户集合
				users_opted_item1 = global_item_user_dict[item1_cate][USER_ID]
				users_opted_item2 = global_item_user_dict[item2_cate][USER_ID]

				#这两个集合的交集
				users_opted_item1and2 = set(users_opted_item1).union(users_opted_item2) ^ (set(users_opted_item1) ^ set(users_opted_item2))

				sim = 0.0
				for user_id in users_opted_item1and2:
					sim += 1 / math.log( 1 + len(global_user_item_dict[user_id]) )

				#两个 item categories 的相似度
				sim /= math.sqrt( len(users_opted_item1) * len(users_opted_item2) )

				global_itemCatalogSimilarities[key1][key2] = sim
				logging.info("%s itemCF [%s -- %s] = %.3f" % (getCurrentTime(), key1, key2, sim))

			user_opt_idx1 += 1

	return 0

def recommendationItemCF():
	logging.info("recommendationItemCF...")
	for user_id, user_opt in global_user_item_dict.items():
		most_fav_catgory, max_weight = getMostFavCategoryOfUser(user_id)
		bought_items = user_opt[most_fav_catgory][BEHAVIOR_TYPE_BUY]
		if (len(bought_items) == 0):
			logging.info("user [%s] has bought nothing, skip!")
			continue

		for test_item in global_train_item[most_fav_catgory]:
			test_item_id = test_item[0]
			if (not userHasOperatedItem(most_fav_catgory, test_item_id)):
				continue

	return 0


def ItemCollaborativeFiltering():
	itemSimilarity_IUF()
	return 0;



def verifyPrediction(window_start_date, forecast_date, min_proba, nag_per_pos, verify_user_start, verify_user_cnt, clf, grd_enc, logisticReg):
    print("%s verifying..." % (getCurrentTime()))
    g_user_buy_transection.clear()

    print("%s reloading verifying users..." % (getCurrentTime()))
    loadRecordsFromRedis(verify_user_start, verify_user_cnt)

    item_popularity_dict = calculateItemPopularity(window_start_date, forecast_date)

    verify_samples, _ = takingSamplesForForecasting(window_start_date, forecast_date)
    
    print("%s creating verifying feature matrix..." % (getCurrentTime()))
    Xmat_verify = GBDT.createTrainingSet(window_start_date, forecast_date, nag_per_pos, verify_samples, item_popularity_dict, False)
    Xmat_verify = preprocessing.scale(Xmat_verify)

    X_leaves_verify = grd_enc.transform(clf.apply(Xmat_verify))

    predicted_prob = logisticReg.predict_proba(X_leaves_verify)

    predicted_user_item = []
    for index in range(len(predicted_prob)):
        if (predicted_prob[index][1] >= min_proba):
            predicted_user_item.append(verify_samples[index])

    actual_user_item = takingPositiveSamples(forecast_date)

    actual_count = len(actual_user_item)
    predicted_count = len(predicted_user_item)

    hit_count = 0
    user_hit_list = set()

    for user_item in predicted_user_item:
        logging.info("predicted %s , %s" % (user_item[0], user_item[1]))
        if (user_item in actual_user_item):
            hit_count += 1

    for user_item in predicted_user_item:
        for user_item2 in actual_user_item:
            if (user_item[0] == user_item2[0]):
                user_hit_list.add(user_item[0])

    print("hit user: %s" % user_hit_list)

    for user_item in actual_user_item:
        logging.info("acutal buy %s , %s" % (user_item[0], user_item[1]))

    print("forecasting date %s, positive count %d, predicted count %d, hit count %d" %\
          (forecast_date, actual_count, predicted_count, hit_count))

    if (predicted_count != 0):
        p = hit_count / predicted_count
    else:
        p = 0

    if (actual_count != 0):
        r = hit_count / actual_count
    else:
        r = 0

    if (p != 0 or r != 0):        
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0

    print("precission: %.4f, recall %.4f, F1 %.4f" % (p, r, f1))

    return

	