
from common import *
import math

global_itemCatalogSimilarities = dict()

#不考虑权值, Inverse User Frequence
#这里计算的是两个 item category 的相似度而不是两个item的相似度，因为 test 中的 item 不是都出现在train中
def itemSimilarity_IUF():
	print("%s itemSimilarity_IUF..." % getCurrentTime())

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
				users_opted_item1 = global_item_user_dict[item1_cate]
				users_opted_item2 = global_item_user_dict[item2_cate]

				#这两个集合的交集
				users_opted_item1and2 = set(users_opted_item1).union(users_opted_item2) ^ (set(users_opted_item1) ^ set(users_opted_item2))

				sim = 0.0
				for user_id in users_opted_item1and2:
					sim += 1 / math.log( 1 + len(global_user_item_dict[user_id]) )

				sim /= math.sqrt( len(users_opted_item1) * len(users_opted_item2) )

				global_itemCatalogSimilarities[key1][key2] = sim
				print("%s itemCF [%s -- %s] = %.3f" % (getCurrentTime(), key1, key2, sim))

			user_opt_idx1 += 1

	return 0


def ItemCollaborativeFiltering():
	itemSimilarity_IUF()
	return 0;