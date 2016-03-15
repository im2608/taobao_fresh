from common import *
import math


global_userSimilarities = dict()

#得到两个用户共同操作过的 item categories
def getItemIntersectionByUser(user1, user2):
	itemsUser1_Opted = set(global_user_item_dict[user1].keys())
	itemsUser2_Opted = set(global_user_item_dict[user2].keys())

	return itemsUser1_Opted.union(itemsUser2_Opted) ^ (itemsUser1_Opted ^ itemsUser2_Opted)

#不考虑权值, Inverse Item Frequence
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
		print("%s userCF: item category %s [%d/%d] is operated by %d users." % \
			  (getCurrentTime(), item_id, item_category_idx, item_category_cnt, userCnt))

		while (userIdx1 < userCnt):
			userIdx2 = userIdx1 + 1
			while (userIdx2 < userCnt):
				user1Id = user_id_list[userIdx1]
				user2Id = user_id_list[userIdx2]

				userIdx2 += 1

				# 查看是否已经计算过 user1, user2的相似度，若没有，则将sim 插入到
				# global_userSimilarities[key1][key2] 的位置
				# key1, key2 分别为 < user1, user2> or <user2, user1>
				key1, key2 = getPosOfDoubleHash(user1Id, user2Id, global_userSimilarities)
				if (key1 == None):
					continue

				itemsUser1and2_Opted = getItemIntersectionByUser(user1Id, user2Id)
				sim = 0.0
				for item_id in itemsUser1and2_Opted:
					sim += 1 / math.log(1 + len(global_item_user_dict[item_id]))

				sim /= math.sqrt( len(global_user_item_dict[user1Id]) * \
	            	              len(global_user_item_dict[user2Id]) )

				global_userSimilarities[key1][key2] = sim

				print("%s userCF [%s -- %s] = %.3f" % (getCurrentTime(), key1, key2, sim))

			userIdx1 += 1

	return 0


def UserCollaborativeFiltering():
	userSimilarity_IIF()
	return 0
