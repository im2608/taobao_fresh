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

        if (user_id != '135929892'):
            continue


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

            # if (behavior_time >= user_behavior_seq[seq_len - 1][1]):
            #     idx = seq_len - 1
            #     while (behavior_time >= user_behavior_seq[idx][1] and idx >= 0):
            #         idx -= 1
            #     user_behavior_seq.insert(idx + 1, (behavior_type, behavior_time))
            #     continue

            # for idx in range(seq_len):
            #     if (behavior_time <= user_behavior_seq[idx][1]):
            #         user_behavior_seq.insert(idx, (behavior_type, behavior_time))
            #         break

    #根据操作序列得到用户的购买记录，以及pattern
    for user_id, item_catgory_list in user_behavior_record.items():
        if (user_id not in g_user_buy_transection):
            g_user_buy_transection[user_id] = dict()

        if (user_id not in g_user_behavior_patten):
            g_user_behavior_patten[user_id] = dict()

        for item_category, behavior_seq in item_catgory_list.items():
            if (len(behavior_seq) == 1):
                logging.info("user %s only operated %s once %s, skip!" % (user_id, item_category, behavior_seq))
                if (behavior_seq[0][0] == BEHAVIOR_TYPE_BUY):
                    g_buy_record_cnt += 1
                continue
            #用户购买记录，按照时间排序，相同时间的话，1，2，3在前，4在后
            #将连续的行为压缩成 {behavior：count}的形式
            #[1,1,1,2,3,4] => [{1:3}, {2:1},{3:1},{4:1}]
            sorted_seq = sortAndCompressBuyRecord2(behavior_seq)

            user_buy_record = []
            logging.info("user: %s, cate: %s, seq: %s" % (user_id, item_category, sorted_seq))
            for behavior_consecutive in sorted_seq:
                behavior_type = behavior_consecutive[0][0]
                if (behavior_type != BEHAVIOR_TYPE_BUY):
                    user_buy_record.append(behavior_consecutive)
                else:
                    if (item_category not in g_user_buy_transection[user_id]):
                        g_user_buy_transection[user_id][item_category] = []

                    #logging.info("appending user %s buy %s, %s" % (user_id, item_category, user_buy_record))
                    g_user_buy_transection[user_id][item_category].append(user_buy_record.copy())
                    g_buy_record_cnt += 1
                    user_buy_record.clear()

            if (len(user_buy_record) > 0):
                if (item_id not in g_user_behavior_patten[user_id]):
                    g_user_behavior_patten[user_id][item_category] = []

                g_user_behavior_patten[user_id][item_category].append(user_buy_record.copy())

#    logging.info("user_behavior_seq is %s" % user_behavior_record)
    for user_id, item_category_buy in g_user_buy_transection.items():
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

def insertBehaviorIntoSequence(behavior_type, behavior_time, behavior_sequence):
    # behavior time 大于序列中的最大时间, 从后向前查找插入位置，保证相同时间段内的相同的 behavior 排列在一起
    seq_len = len(behavior_sequence)

    if (behavior_time >= behavior_sequence[seq_len - 1][1]):
        idx = seq_len - 1
        while (idx >= 0 and behavior_sequence[idx][1] >= behavior_time and \
               behavior_sequence[idx][0] != behavior_type):
            idx -= 1
        behavior_sequence.insert(idx + 1, (behavior_type, behavior_time))
    else:
        idx = 0
        while (idx < seq_len and behavior_time < behavior_sequence[idx][1] and \
               behavior_sequence[idx][0] != behavior_type):
            idx += 1
        behavior_sequence.insert(idx, (behavior_type, behavior_time))
    

    return 0

# def transferData(buy_record_file_name, patten_file_name):
#     output_buy = open("%s..\\input\\%s" % (runningPath, output_file_name), encoding="utf-8", mode='w')
#     output_patten = open("%s..\\input\\%s" % (runningPath, patten_file_name), encoding="utf-8", mode='w')
#     for user_id, item_id in g_user_buy_transection.items():


#     return 0


#用户购买记录，按照时间排序，相同时间的情况下，1，2，3排在前，4在后
def sortAndCompressBuyRecord2(user_buy_record):
    sorted_compressed_hehavior = []

    for user_behavior in user_buy_record: #user_behavior is [behavior, time]
        sorted_len = len(sorted_compressed_hehavior)
        if (sorted_len == 0):
            sorted_compressed_hehavior.append([user_behavior, 1])
            continue

        if (user_behavior[1] > sorted_compressed_hehavior[sorted_len - 1][0][1]):
            sorted_compressed_hehavior.append([user_behavior, 1])
            continue

        if (user_behavior[1] < sorted_compressed_hehavior[0][0][1]):
            sorted_compressed_hehavior.insert(0, [user_behavior, 1])
            continue

        inserted = False
        idx = 0
        for behavior_consecutive in sorted_compressed_hehavior:
            if (behavior_consecutive[0] == user_behavior):
                behavior_consecutive[1] += 1
                inserted = True
                break
            if (user_behavior[1] < behavior_consecutive[0][1]):
                break

            idx += 1
        if (not inserted):
            sorted_compressed_hehavior.insert(idx, [user_behavior, 1])

    return sorted_compressed_hehavior

def sortAndCompressBuyRecord(user_buy_record):
    sorted_behavior = []

    #先得到所有的浏览记录
    for user_behavior in user_buy_record:
        if (user_behavior[0] == BEHAVIOR_TYPE_VIEW):
            sorted_behavior.append(user_behavior)

    #在浏览记录中插入收藏，购物车记录
    for user_behavior in user_buy_record:
        if (user_behavior[0] == BEHAVIOR_TYPE_FAV or user_behavior[0] == BEHAVIOR_TYPE_CART):
            sorted_len = len(sorted_behavior)
            #有收藏，购物车记录但却没有浏览记录，补一条浏览记录
            if (sorted_len  == 0):
                sorted_behavior.append((1, user_behavior[1]))

            if (user_behavior[1] >= sorted_behavior[sorted_len -1][1]):
                sorted_behavior.append(user_behavior)
                continue

            for index in range(sorted_len - 1):
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

    sorted_compressed_hehavior = []
    cur_behavior_dict = [sorted_behavior[0][0], 0]

    #将 [behavior, time] 压缩成 [behavior, time, count] 的形式
    for behavior in sorted_behavior:
        if (behavior[0] == cur_behavior_dict[0]):
            cur_behavior_dict[1] += 1 # count + 1
        else:
            # [[1:3], [2:1]] 需要再次添加 [1： 3]，此时有两种情况
            # 1. 两个 [1:3] 时间相同， 将第一个[1:3] +1 得到 [1:4]
            # 2. 两个 [1:3] 时间不同， 将第二个[1：3][0] 加上 4 得到[5：3]与第一个[1：3]区分开，以后直接 mod 4 即可
            the_same = 0
            for behavior_consecutive in sorted_compressed_hehavior:
                #第一种情况, behavior, time 都相同
                if (behavior_consecutive[0] % 4 == cur_behavior_dict[0] and \
                    behavior_consecutive[1] == cur_behavior_dict[1]):
                    behavior_consecutive[1] += 1
                    break
                #第二种情况， behavior 相同， time 不同
                elif (cur_behavior_dict[0] == behavior_consecutive[0]):
                    the_same += 1

            cur_behavior_dict[0] += the_same * 4

            sorted_compressed_hehavior.append(cur_behavior_dict.copy())
            cur_behavior_dict = [behavior[0], 1]

    sorted_compressed_hehavior.append(cur_behavior_dict)

    logging.info("sorted_behavior %s" % sorted_behavior)
    logging.info("sorted_compressed_hehavior %s" % sorted_compressed_hehavior)

    return sorted_compressed_hehavior

#以单项形式得到所有的 behavior_consecutive
def createC1():
    C1 = []
    for user_id, item_category_buy in g_user_buy_transection.items():
        for item_category, buy_record in item_category_buy.items():
            for behavior_consecutive in buy_record:
                if (behavior_consecutive not in C1):
                    C1.append([behavior_consecutive])

    return map(frozenset, C1)

#从Ck中查找满足minsupoprt的 behavior_consecutive
def scanMinSupport(Ck):
    ssCnt = {}
    for user_id, item_category_buy in g_user_buy_transection.items():
        for item_category, buy_record in item_category_buy.items():
            for behavior_consecutive in Ck:
                if (behavior_consecutive.issubset(buy_record)):
                    if (can not in ssCnt):
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1

    retList = []
    for behavior_consecutive, support in ssCnt:
        if (support >= g_min_support):
            retList.append(behavior_consecutive)

    return retList

# Lk 内每个元素均为k项， 将它们合并成 k+1 项
def aprioriGen(Lk, K):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            #截取 [0, k-2) 的元素
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            if (L1 == L2):
                retList.append(L1 | L2)
    return retList

def aprioriAlgorithm():
    logging.info("entered aprioriAlgorithm")

    C1 = createC1()
    L1 = scanMinSupport(C1)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2])
        Lk = scanMinSupport(Ck)
        L.append(Lk)
        k += 1

    logging.info("aprioriAlgorithm L is %s" % L)
    return L