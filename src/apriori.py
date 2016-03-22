from common import *

#记录的用户在item上的购买行为
g_user_buy_transection = dict()

#在用户的操作记录中，从最后一条购买记录到train时间结束之间的操作行为记录，
#以它们作为patten
g_user_behavior_patten = dict()

#总共的购买记录数
g_buy_record_cnt = 0
g_min_support = 0.0

def loadData(user_opt_file_name):
    global g_buy_record_cnt
    global g_min_support

    filehandle1 = open(user_opt_file_name, encoding="utf-8", mode='r')
    user_behavior_csv = csv.reader(filehandle1)
    index = 0

    user_behavior_record = dict()
    logging.info("loading file %s" % user_opt_file_name)
    for aline in user_behavior_csv:
        if (index == 0):
            index += 1
            continue

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        user_geohash  = aline[3]
        item_category = int(aline[4])
        behavior_time = datetime.datetime.strptime(aline[5], "%Y-%m-%d %H")

        if (user_id not in user_behavior_record):
            user_behavior_record[user_id] = dict()

        if (item_category not in user_behavior_record[user_id]):
        	user_behavior_record[user_id][item_category] = []

        user_behavior_seq = user_behavior_record[user_id][item_category]
        seq_len = len(user_behavior_seq)
        #按照操作时间生成操作序列
        if (seq_len == 0):
        	user_behavior_seq.append((behavior_type, behavior_time))
        else:
        	if (behavior_time > user_behavior_seq[seq_len - 1][1]):
        		user_behavior_seq.insert(seq_len, (behavior_type, behavior_time))
        		continue

        	for idx in range(seq_len):
        		if (behavior_time <= user_behavior_seq[idx][1]):
        			user_behavior_seq.insert(idx, (behavior_type, behavior_time))
        			break

    #根据操作序列得到用户的购买记录，以及pattern
    for user_id, item_catgory_list in user_behavior_record.items():
    	if (user_id not in g_user_buy_transection):
    		g_user_buy_transection[user_id] = dict()

    	if (user_id not in g_user_behavior_patten):
    		g_user_behavior_patten[user_id] = dict()

    	for item_category, behavior_seq in item_catgory_list.items():
    		# if (len(behavior_seq) == 1):
    		# 	logging.info("user %s only operated %s once %s, skip!" % (user_id, item_category, behavior_seq))
    		# 	continue

    		#用户购买记录，按照时间排序，相同时间的话，1，2，3在前，4在后    		
    		sorted_seq = sortBuyRecord(behavior_seq)

    		user_buy_record = []
    		#logging.info("sortBuyRecord user: %s, item category: %s, seq: %s" % (user_id, item_category, sorted_seq))
    		for behavior_type in sorted_seq:
    			if (behavior_type[0] != BEHAVIOR_TYPE_BUY):
    				user_buy_record.append(behavior_type)
    			else:
    				if (item_category not in g_user_buy_transection[user_id]):
    					g_user_buy_transection[user_id][item_category] = []

    				#logging.info("appending user %s buy %s, %s" % (user_id, item_category, user_buy_record))
    				#一条购买记录, 1表示该购买记录是频繁项， 在算法中若发现该记录不是频繁项，则将其置0
    				g_user_buy_transection[user_id][item_category].append([user_buy_record.copy(), 1])
    				#g_buy_record_cnt = g_buy_record_cnt + 1
    				g_buy_record_cnt += 1
    				user_buy_record.clear()

    		if (len(user_buy_record) > 0):
    		    if (item_id not in g_user_behavior_patten[user_id]):
    		    	g_user_behavior_patten[user_id][item_category] = []

    		    g_user_behavior_patten[user_id][item_category].append(user_buy_record.copy())

#    logging.info("user_behavior_seq is %s" % user_behavior_record)
    for user_id, item_category_buy in g_user_buy_transection.items():
        if (user_id != '33918445'):
            continue

        buy_cnt = 0
        for item_category, buy_record in item_category_buy.items():
            logging.info("item category %s : %s" % (item_category, buy_record))
            buy_cnt += len(buy_record)
        logging.info("user %s -- %d buy records" % (user_id, buy_cnt))

    #logging.info("g_user_buy_transection is (%d) %s" % (len(g_user_buy_transection), g_user_buy_transection.keys()))

    g_min_support = round(g_buy_record_cnt * 0.3)
    logging.info("total buy record count is %d, min support is %d" % (g_buy_record_cnt, g_min_support))

    # logging.info("g_user_behavior_patten is %s" % g_user_behavior_patten)

    return 0

# def transferData(buy_record_file_name, patten_file_name):
# 	output_buy = open("%s..\\input\\%s" % (runningPath, output_file_name), encoding="utf-8", mode='w')
# 	output_patten = open("%s..\\input\\%s" % (runningPath, patten_file_name), encoding="utf-8", mode='w')
# 	for user_id, item_id in g_user_buy_transection.items():


# 	return 0


#用户购买记录，按照时间排序，相同时间的情况下，1，2，3排在前，4在后
def sortBuyRecord(user_buy_record):
	sorted_behavior = []

	#先得到所有的浏览记录
	for user_behavior in user_buy_record:
		if (user_behavior[0] == BEHAVIOR_TYPE_VIEW):
			sorted_behavior.append(user_behavior)

	#在浏览记录中插入收藏，购物车记录
	for user_behavior in user_buy_record:
		if (user_behavior[0] == BEHAVIOR_TYPE_FAV or user_behavior[0] == BEHAVIOR_TYPE_CART):
			#有收藏，购物车记录但却没有浏览记录，补一条浏览记录
			if (len(sorted_behavior) == 0):
				sorted_behavior.append((1, user_behavior[1]))

			if (user_behavior[1] >= sorted_behavior[len(sorted_behavior)-1][1]):
				sorted_behavior.append(user_behavior)
				continue

			for index in range(len(sorted_behavior) - 1):
				if (user_behavior[1] < sorted_behavior[index][1]):
					sorted_behavior.insert(index, user_behavior)
					break

	#在浏览记录中插入购买记录
	for user_behavior in user_buy_record:
		if (user_behavior[0] != BEHAVIOR_TYPE_BUY):
			continue

		#有购买记录但却没有浏览记录，补一条浏览记录
		if (len(sorted_behavior) == 0):
			sorted_behavior.append((1, user_behavior[1]))

		if (user_behavior[1] >= sorted_behavior[len(sorted_behavior)-1][1]):
			sorted_behavior.append(user_behavior)
			continue

		for index in range(len(sorted_behavior) - 1):
			if (user_behavior[1] < sorted_behavior[index][1]):
				sorted_behavior.insert(index, user_behavior)
				break				

	return sorted_behavior
