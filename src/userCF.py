from common import *
import math


global_userSimilarities = dict()

#得到两个用户共同操作过的 item categories
def getItemIntersectionByUser(user1, user2):
	itemsUser1_Opted = set(global_user_item_dict[user1].keys())
	itemsUser2_Opted = set(global_user_item_dict[user2].keys())

	return itemsUser1_Opted.union(itemsUser2_Opted) ^ (itemsUser1_Opted ^ itemsUser2_Opted)

#这里通过两个用户共同操作了多少个相同的item category来计算用户相似度，而不是通过共同操作了多少个相同的item，
#因为 test 中的item 不是都出现在train中
def userSimilarity_IIF():
	print(getCurrentTime(), " userSimilarity_IIF")
	item_category_cnt = len(global_item_user_dict)
	item_category_idx = 0


	for (item_id, user_ids) in global_item_user_dict.items():		
		item_category_idx += 1
		userCnt = len(user_ids)
		user_id_list = list(user_ids)
		userIdx1 = 0
		# print("%s userCF: item category %s [%d/%d] is operated by %d users." % \
		# 	  (getCurrentTime(), item_id, item_category_idx, item_category_cnt, userCnt))

		while (userIdx1 < userCnt):
			userIdx2 = userIdx1 + 1
			while (userIdx2 < userCnt):
				user1_id = user_id_list[userIdx1]
				user2_id = user_id_list[userIdx2]

				userIdx2 += 1

				# 查看是否已经计算过 user1, user2的相似度，若没有，则将sim 插入到
				# global_userSimilarities[key1][key2] 的位置
				# key1, key2 分别为 < user1, user2> or <user2, user1>
				key1, key2 = getPosOfDoubleHash(user1_id, user2_id, global_userSimilarities)
				if (key1 == None):
					continue

				itemsUser1and2_Opted = getItemIntersectionByUser(user1_id, user2_id)
				sim = 0.0
				for item_id in itemsUser1and2_Opted:
					sim += 1 / math.log(1 + len(global_item_user_dict[item_id]))

				sim /= math.sqrt( len(global_user_item_dict[user1_id]) * \
	            	              len(global_user_item_dict[user2_id]) )

				global_userSimilarities[key1][key2] = sim

				logging.info("%s userCF [%s -- %s] = %.3f" % (getCurrentTime(), key1, key2, sim))

			userIdx1 += 1

	return 0

def recommendationUserCF(topK):
	logging.info("recommendationUserCF topK %d", topK)

	user_sim_topK = dict()

	for user1_id in global_userSimilarities.keys():
		
		user_sim_topK[user1_id] = []
		users_no_sim = ""

		# 计算每个用户相似度的 topK
		for user2_id in global_userSimilarities.keys():
			if (user1_id == user2_id):
				continue

			similiarity = 0.0
			if (user2_id in global_userSimilarities[user1_id]):
				similiarity = global_userSimilarities[user1_id][user2_id]
			elif (user1_id in global_userSimilarities[user2_id]):
				similiarity = global_userSimilarities[user2_id][user1_id]
			else:
				users_no_sim += "%s " % user2_id
				continue

			# topK 是一个数组， 数组中的每个元素是一个两个元素的数组，第一个是user id， 第二个是similiarity
			if (len(user_sim_topK[user1_id]) < topK):
				user_sim_topK[user1_id].append([user2_id, similiarity])
			else:
				max_sim = 0
				min_sim = 0

				for idx in range(topK):
					if (user_sim_topK[user1_id][min_sim][1] > user_sim_topK[user1_id][idx][1]):
						min_sim = idx

				if (similiarity > user_sim_topK[user1_id][min_sim][1]):
					user_sim_topK[user1_id][min_sim][0] = user2_id
					user_sim_topK[user1_id][min_sim][1] = similiarity

				# 	if (user_sim_topK[user1_id][max_sim][1] < user_sim_topK[user1_id][idx][1]):
				# 		max_sim = idx

				# if (similiarity > user_sim_topK[user1_id][min_sim][1] and \
				# 	similiarity < user_sim_topK[user1_id][max_sim][1]):
				# 	user_sim_topK[user1_id][min_sim][0] = user2_id
				# 	user_sim_topK[user1_id][min_sim][1] = similiarity

		if (len(users_no_sim) > 0):
			logging.info("%s has no similiarity with following users: %s" % (user1_id, users_no_sim))

		logging.info("top%d of %s is %s" % (topK, user1_id, user_sim_topK[user1_id]))

		#根据相似度 topK 来计算用户对相应的item categories的权值
		item_category_weight = calcuteItemCategoryWeight(user1_id, user_sim_topK[user1_id])

		# final weight = 根据相似度 topK 得到的 item categories的权值 * 根据behavior 得到的 item category weight 
		# 推荐 final weight 最大的 category
		for category, weight in item_category_weight.items():
			tmp = global_user_item_dict[user1_id][category]["w"] * weight
			logging.info("user [%s] final category weight = topK_weight[%.3f] * [%.3f]opt_weight = %.3f" % 
				         (user1_id, global_user_item_dict[user1_id][category]["w"], weight, tmp))
			item_category_weight[category] = tmp

		sorted_category_weight = sorted(item_category_weight.items(), key=lambda d:d[1], reverse=True)
		logging.info("user [%s] sorted final category weight %s" % (user1_id, sorted_category_weight))

		# final weight 最大的 category
		category = sorted_category_weight[0][0]
		weight = sorted_category_weight[0][1]
		finalRecommendation(user1_id, category)
	return 0

def finalRecommendation(user_id, category):
	if (category not in global_train_item):
		return

	for item_id in global_train_item[category]:
		outputFile.write("%s,%s\n" % (user_id, item_id[0]))

	return 0

#根据相似度 topK 来计算用户对相应的item categories的权值
def calcuteItemCategoryWeight(user_id, user_sim_topK):
	item_category_weight = dict()

	# 根据 topK 中的 similiarity， 来计算所有用户操作过的 item category 的权值		
	for item_category in global_user_item_dict[user_id].keys():
		item_category_weight[item_category] = 0.0

		for user_idx in range(len(user_sim_topK)):

		    user_in_topK = user_sim_topK[user_idx][0]
		    sim_in_topK = user_sim_topK[user_idx][1]

		    if (item_category in global_user_item_dict[user_in_topK]):
		    	item_category_weight[item_category] += sim_in_topK

		logging.info("user [%s] topK category weight [%s -- %.3f]" % (user_id, item_category, item_category_weight[item_category]))
	return item_category_weight

    #按照权值从大到小排序， 返回一个list
	#return sorted(item_category_weight.items(), key=lambda d:d[1], reverse=True)

def UserCollaborativeFiltering():
	userSimilarity_IIF()
	return 0
