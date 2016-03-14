from common import *
import math

global_userSimilarities = dict()

def getItemIntersectionByUser(user1, user2):
	itemsUser1_Opted = getItemsUserOpted(user1)
	itemsUser2_Opted = getItemsUserOpted(user1)

	return set(itemsUser1_Opted).union(itemsUser2_Opted) ^ (set(itemsUser1_Opted) ^ set(itemsUser2_Opted))

#不考虑权值
def userSimilarity_IIF():
	print(getCurrentTime(), " userSimilarity_IIF")
	for (item_id, user_ids) in global_item_user_dict.items():		
		userCnt = len(user_ids)
		user_id_list = list(user_ids)


		userIdx1 = 0
		while (userIdx1 < userCnt):
			userIdx2 = userIdx1 + 1
			while (userIdx2 < userCnt):
				user1Id = user_id_list[userIdx1]
				user2Id = user_id_list[userIdx2]

				user_key1 = "%s,%s" % (user1Id, user2Id)
				user_key2 = "%s,%s" % (user2Id, user1Id)

				if ( (user_key1 in global_userSimilarities) or \
					 (user_key2 in global_userSimilarities) ):				    
				    userIdx2 += 1
				    continue

				itemsUser1and2_Opted = getItemIntersectionByUser(user1Id, user2Id)
				sim = 0.0
				for item_id in itemsUser1and2_Opted:
					sim += 1 / math.log(1 + len(global_item_user_dict[item_id]))

				sim /= math.sqrt( len(global_user_item_dict[user1Id]) * \
	            	              len(global_user_item_dict[user2Id]) )
				global_userSimilarities[user_key1] = sim

				print("%s [%s] = %.3f" % (getCurrentTime(), user_key1, sim))
				userIdx2 += 1

			userIdx1 += 1

	print("global_userSimilarities is ", global_userSimilarities)
	return 0


def UserCollaborativeFiltering():
	userSimilarity_IIF()
	return 0
