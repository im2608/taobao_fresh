from common import *

#记录的用户在item上的购买行为
g_user_buy_transection = dict()

#在用户的操作记录中，从最后一条购买记录到train时间结束之间的操作行为记录，
#以它们作为patten
g_user_behavior_patten = dict()

#总共的购买记录数
g_buy_record_cnt = 0
g_min_support = 0.0

redis_cli = redis.Redis(host='10.57.14.3', port=6379, db=0)


def loadData(save_to_redis=False, user_opt_file_name = tianchi_fresh_comp_train_user):
    global g_buy_record_cnt
    global g_min_support

    filehandle1 = open(user_opt_file_name, encoding="utf-8", mode='r')

    user_behavior_csv = csv.reader(filehandle1)
    index = 0
    users = ['1774834']

    user_behavior_record = dict()
    skiped_buy_cnt = 0
    logging.info("loading file %s" % user_opt_file_name)
    for aline in user_behavior_csv:
        if (index == 0):
            index += 1
            continue

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        user_geohash  = aline[3]
        item_category = aline[4]
        behavior_time = datetime.datetime.strptime(aline[5], "%Y-%m-%d %H")

        if (index % 100000 == 0):
            print("%d lines read\r" % index,  end="")

        # if (user_id not in  users):
        #     index += 1
        #     continue

        if (user_id not in user_behavior_record):
            user_behavior_record[user_id] = dict()

        if (item_category not in user_behavior_record[user_id]):
            user_behavior_record[user_id][item_category] = []

        #用户在某个category上的每个操作以（操作类型，操作时间） 的二元组表示
        #user_behavior_seq 为该二元组序列，按照时间排序，相同时间的按照1，2，3，4排序
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
        index += 1

    print("\r\ntotal %d lines read" % index)

    logging.debug("user_behavior_record is %s" % user_behavior_record)

    #根据操作序列得到用户的购买记录，以及pattern
    print("%s getting user buy records" % getCurrentTime())
    for user_id, item_catgory_list in user_behavior_record.items(): #用户， 该用户在哪些categories 上有操作
        if (user_id not in g_user_buy_transection):
            g_user_buy_transection[user_id] = dict()

        if (user_id not in g_user_behavior_patten):
            g_user_behavior_patten[user_id] = dict()

        for item_category, behavior_seq in item_catgory_list.items(): # category, 在
            #用户在某个category上只有一次非购物操作，略过
            if (len(behavior_seq) == 1 and behavior_seq[0][0] != BEHAVIOR_TYPE_BUY):
                continue

            #用户购买记录，按照时间排序，相同时间的话，1，2，3在前，4在后
            #将连续的行为压缩成 {behavior：count}的形式
            #[1,1,1,2,3,4] => [((1, time), 3), ((2, time), 1), ((3, time), 1), ((4, time), 1)]
            sorted_seq = sortAndCompressBuyRecord(behavior_seq)

            user_buy_record = []
            logging.debug("user: %s, cate: %s, behavior seq: %s, sorted seq: %s" % (user_id, item_category, behavior_seq, sorted_seq))
            for behavior_consecutive in sorted_seq:
                behavior_type = behavior_consecutive[0][0]                
                if (behavior_type != BEHAVIOR_TYPE_BUY):
                    user_buy_record.append(behavior_consecutive)
                else:
                    #有些购物记录没有任何浏览记录，跳过
                    if (len(user_buy_record) == 0):
                        g_buy_record_cnt += 1
                        skiped_buy_cnt += 1
                        continue

                    if (item_category not in g_user_buy_transection[user_id]):
                        g_user_buy_transection[user_id][item_category] = []

                    #logging.info("appending user %s buy %s, %s" % (user_id, item_category, user_buy_record))
                    
                    #将绝对时间转成相对时间， 起始时间为该购物记录的第一个操作的时间, 单位为小时
                    #采用序号来代替时间
                    start_time = user_buy_record[0][0][1]
                    for idx in range(len(user_buy_record)):
                        time_dif = user_buy_record[idx][0][1] - start_time
                        time_dif = time_dif.days * 24 + time_dif.seconds / 3600
                        #重新生成新的三元组
                        user_buy_record[idx] = (user_buy_record[idx][0][0], idx, user_buy_record[idx][1])

                    #如果有连续的购买，则为每个购买行为生成一条购物记录
                    buy_cnt = behavior_consecutive[1]
                    for each_buy in range(buy_cnt):
                        g_user_buy_transection[user_id][item_category].append(user_buy_record.copy())
                        g_buy_record_cnt += 1

                    user_buy_record.clear()


            if (len(user_buy_record) > 0):
                if (item_id not in g_user_behavior_patten[user_id]):
                    g_user_behavior_patten[user_id][item_category] = []

                g_user_behavior_patten[user_id][item_category].append(user_buy_record.copy())

    if (save_to_redis):
        saveRecordstoRedis()

    for user_id, item_category_buy in g_user_buy_transection.items():
        buy_cnt = 0
        for item_category, buy_records in item_category_buy.items():
            for each_record in buy_records:
                logging.info("user %s item category %s : %s" % (user_id, item_category, each_record))
            buy_cnt += len(buy_records)
        logging.info("user %s -- %d buy records" % (user_id, buy_cnt))

    #logging.info("g_user_buy_transection is (%d) %s" % (len(g_user_buy_transection), g_user_buy_transection.keys()))

    g_min_support = round(g_buy_record_cnt * 0.01)
    #g_min_support = 50
    logging.info("%s total buy record count is %d(%d), min support is %d" % (getCurrentTime(), g_buy_record_cnt, skiped_buy_cnt, g_min_support))
    print("%s total buy record count is %d(%d), min support is %d" % (getCurrentTime(), g_buy_record_cnt, skiped_buy_cnt, g_min_support))

    filehandle1.close()

    # logging.info("g_user_behavior_patten is %s" % g_user_behavior_patten)

    return 0

def saveRecordstoRedis():
    print("%s saveRecordstoRedis()" % getCurrentTime())
    all_users = list(g_user_buy_transection.keys())
    redis_cli.set("all_users", ",".join(all_users))

    for user_id, item_category_buy in g_user_buy_transection.items():
        item_categories_str = list(item_category_buy.keys())
        #item_categories_str = [str(x) for x in item_categories]
        redis_cli.hset(user_id, "item_category", ",".join(item_categories_str))

        for item_category, buy_records in item_category_buy.items():
            redis_cli.hset(user_id, item_category, buy_records)

    print("%s saveRecordstoRedis() Done!" % getCurrentTime())

    return 0

# 每条购物记录在 redis 中都表现为字符串 
#"[[(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)], [(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)], [(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)]]"
def loadRecordsFromRedis():
    global g_buy_record_cnt
    global g_min_support

    # 得到所有的用户
    all_users = redis_cli.get("all_users").decode()
    all_users = all_users.split(",")

    total_user = len(all_users)
    print("%s loadRecordsFromRedis, here total %d users" % (getCurrentTime(), total_user))

    #根据用户得到用户操作过的item category
    user_index = 0
    skiped_user = 0
    for user_id in all_users:
        if (user_id != '1774834'):
            continue

        g_user_buy_transection[user_id] = dict()
        item_categories = redis_cli.hget(user_id, "item_category").decode()
        if (len(item_categories) == 0):
            logging.info("User %s has no buy records!" % user_id)
            user_index += 1
            skiped_user += 1
            continue

        item_categories = item_categories.split(",")
        #得到用户在某个category上的购物记录，字符串形式，分解字符串
        for category in item_categories:
            g_user_buy_transection[user_id][category] = []
            buy_records = redis_cli.hget(user_id, category).decode()
            if (len(buy_records) <= 4):
                logging.info("user %s on category %s, buy records Length <= 4 %s" % (user_id, category, buy_records))

            #去掉开头和末尾的 [[ ]]
            buy_records = buy_records[2 : len(buy_records)-2]
           
            #得到 购买记录 list
            #buy_records[i] = "(1, 0.0, 35), (2, 0.0, 1), (3, 0.0, 1)"
            buy_records = buy_records.split("], [")
            for idx  in range(len(buy_records)):
                #去掉开头和末尾的 ( )
                buy_records[idx] = buy_records[idx][1 : len(buy_records[idx])-1]

                #得到三元组list： '1, 0.0, 35'， '2, 0.0, 1'
                behavior_tuple_list =  buy_records[idx].split("), (")

                user_buy_record = []
                for behavior_tuple in behavior_tuple_list:
                    behavior = behavior_tuple.split(", ")
                    behavior_type  = int(behavior[0])
                    behavior_time  = float(behavior[1])
                    behavior_count = int(behavior[2])
                    user_buy_record.append((behavior_type, behavior_time, behavior_count))
                g_user_buy_transection[user_id][category].append(user_buy_record)
                g_buy_record_cnt += 1

        user_index += 1
        print("%d / %d users read\r" % (user_index, total_user), end="")

    g_min_support = round(g_buy_record_cnt * 0.01)
    if (g_min_support < 1):
        g_min_support = 1

    logging.info("g_buy_record_cnt is %d, g_min_support is %d" % (g_buy_record_cnt, g_min_support))
    print("%s total buy count %d, min support %d " % (getCurrentTime(), g_buy_record_cnt, g_min_support))
    #redis_cli.close()
    return 0



#用户购买记录，按照时间排序，相同时间的情况下，1，2，3排在前，4在后
def sortAndCompressBuyRecord(user_buy_record):
    sorted_compressed_behavior = []

    for user_behavior in user_buy_record: #user_behavior is [behavior, time]
        sorted_len = len(sorted_compressed_behavior)
        if (sorted_len == 0):
            sorted_compressed_behavior.append([user_behavior, 1])
            continue

        if (user_behavior[1] > sorted_compressed_behavior[sorted_len - 1][0][1]):
            sorted_compressed_behavior.append([user_behavior, 1])
            continue

        if (user_behavior[1] < sorted_compressed_behavior[0][0][1]):
            sorted_compressed_behavior.insert(0, [user_behavior, 1])
            continue

        inserted = False
        idx = 0
        for behavior_consecutive in sorted_compressed_behavior:
            if (behavior_consecutive[0] == user_behavior):
                behavior_consecutive[1] += 1
                inserted = True
                break
            if (user_behavior[1] < behavior_consecutive[0][1] or \
                (user_behavior[1] == behavior_consecutive[0][1] and user_behavior[0] < behavior_consecutive[0][0])):
                break
            idx += 1

        if (not inserted):
            sorted_compressed_behavior.insert(idx, [user_behavior, 1])


    return sorted_compressed_behavior

#以单项形式得到所有的 behavior_consecutive
def createC1():
    print("%s creating C1" % getCurrentTime())
    C1 = {}
    for user_id, item_category_buy in g_user_buy_transection.items():
        for item_category, buy_records in item_category_buy.items():
            for each_record in buy_records:
                for behavior_consecutive in each_record:
                    #logging.info("each_record %s" % each_record)
                    if behavior_consecutive not in C1:
                        C1[behavior_consecutive] = 1
                    else:
                        C1[behavior_consecutive] += 1

                    # found = False

                    # for c1_item in C1:
                    #     if (behavior_consecutive in c1_item):
                    #         found = True                                           
                    #         break
                    # if (not found):
                    #     C1.append([behavior_consecutive])
    return C1

#从Ck中的每一项都是一个[], [] 中包含一个或多个 behavior_consecutive， 查找这些behavior_consecutive的组合是否满足minsupoprt
# ssCnt 的结构为：
#  |<----                 frequence item                                                       -------->|  |<-- 支持度
#  |  |<---    behavior_consecutive            ---->|  |<---        behavior_consecutive        ---->|  |  |
# [[  [(3, datetime.datetime(2014, 12, 14, 7, 0)), 1]， [(4, datetime.datetime(2014, 12, 14, 7, 0)), 1]  ], 1]
#       3:操作类型， datetime.datetime(2014, 12, 14, 7, 0)： 操作时间， 1： 操作次数

def scanMinSupportForC1(C1):
    tmp_C1 = []
    for behavior_consecutive in C1.keys():
        if C1[behavior_consecutive] < g_min_support:
            #del(C1[behavior_consecutive])
            tmp_C1.append(behavior_consecutive)

    for behavior_consecutive in tmp_C1:
        del(C1[behavior_consecutive])

    return C1


def scanMinSupport(Ck, k):
    ssCnt = []
    totalCk = len(Ck)
    index = 0
    for ck_item in Ck:
        for user_id, item_category_buy in g_user_buy_transection.items():
            for item_category, buy_records in item_category_buy.items():
                for each_record in buy_records:
                    if (isCKItemInBuyRecord(ck_item, each_record)):
                        AddItemSupport(ck_item, ssCnt)
        index += 1
        print("%s %d / %d C%d items checked!\r" % (getCurrentTime(), index, totalCk, k), end="")

    retList = []
    totalSup = 0
    for behavior_consecutive_support in ssCnt:
        if (behavior_consecutive_support[1] >= g_min_support):
            retList.append(behavior_consecutive_support[0])
            totalSup += behavior_consecutive_support[1]

    logging.info("total C%d has %d items, %d appeared in buy records, %d meet min-support" % (k, len(Ck), len(ssCnt), len(retList)))
    print("%s %d C%d items appeared in buy records, %d meet min-support" % (getCurrentTime(), len(ssCnt), k, len(retList)))
    for retList_item in retList:
        logging.debug(retList_item)

    return retList

# ck_item 中的每个元素都包含有 k 项 behavior_consecutive， 查看它们是否都在 buy_record 中
def isCKItemInBuyRecord(ck_item, buy_record):
    # if (len(ck_item) > 1):
    #     logging.info("ck_item is %s, buy_record is %s" % (ck_item, buy_record))
    for each_item in ck_item:
        inRecord = False
        if (each_item in buy_record):
            inRecord = True

        if (not inRecord):
            #logging.info("ck_item %s is Not in buy_record %s" % (ck_item, buy_record))
            return False

    # if (len(ck_item) > 1):
    #     logging.info("ck_item %s is in buy_record %s" % (ck_item, buy_record))
    return True


def isCKItemInBuyRecord(ck_item, buy_record):
    # if (len(ck_item) > 1):
    #     logging.info("ck_item is %s, buy_record is %s" % (ck_item, buy_record))
    for each_item in ck_item:
        inRecord = False
        for behavior_consecutive in buy_record:
            #logging.info("isCKItemInBuyRecord each_item %s, behavior_consecutive %s" % (each_item, behavior_consecutive))
            if (each_item == behavior_consecutive):
                inRecord = True
                break
        if (not inRecord):
            #logging.info("ck_item %s is Not in buy_record %s" % (ck_item, buy_record))
            return False

    if (len(ck_item) > 1):
        logging.debug("ck_item %s is in buy_record %s" % (ck_item, buy_record))
    return True

def AddItemSupport(behavior_consecutive, support_dict):
    #logging.info("AddItemSupport behavior_consecutive %s" % behavior_consecutive)
    # logging.info("AddItemSupport support_dict %s" % support_dict)

    for item_support in support_dict:
        # item_support[0] 频繁项
        # item_support[1] 频繁项的支持度
        frequence_item = item_support[0]
        if (len(frequence_item) != len(behavior_consecutive)):
            continue
        found = True
        for idx in range(len(frequence_item)):

            if (frequence_item[idx] != behavior_consecutive[idx]):
                found = False
                break

        if (found):
            #logging.info("found behavior_consecutive %s in support_dict %s" % (behavior_consecutive, support_dict))
            item_support[1] += 1
            return

    #logging.info("not found behavior_consecutive %s in %s" % (behavior_consecutive, support_dict))
    support_dict.append([behavior_consecutive.copy(), 1])
    #logging.info("after append: %s" % (support_dict))

    return 0

# Lk 内每个元素均为k-1项， 将它们合并成 k 项
# 若有两个k-1项集，每个项集按照“属性-值”（一般按值）的字母顺序进行排序。
# 如果两个k-1项集的前k-2个项相同，而最后一个项不同，则证明它们是可连接的，即两这个k-1项集可以连接，即可连接生成 k 项集。
# 使如有两个3项集：｛a, b, c｝{a, b, d}，这两个3项集就是可连接的，它们可以连接生成4项集｛a, b, c, d｝。
# 又如两个3项集｛a, b, c｝｛a, d, e｝，这两个3项集是不能连接生成3项集的。
def aprioriGen(Lk, K):
    logging.info(" aprioriGen length L%d is %d" %  (K-1, len(Lk)))
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            if (len(Lk[i]) != len(Lk[j])):
                logging.error("aprioriGen len(Lk[%d]) != len(Lk[%d]) %s " % (i, j, Lk))
                raise VauleError("Length in Lk does not equal!")

            canMerge = True
            item_len = len(Lk[i])
            #两个k-1项集的前k-2个项相同，而最后一个项不同, 才可合并
            for idx in range(item_len - 1):
                if (Lk[i][idx] != Lk[j][idx]):
                    canMerge = False
                    break

            if (not canMerge):
                logging.notset("Can not mrege: Lk[%d] %s, Lk[%d] %s" %(i, Lk[i], j, Lk[j]))
                continue

            #前 k-2 项都相等，直接merge
            newItem = []
            for idx in range(item_len - 1):
                newItem.append(Lk[i][idx])

            #合并最后一项
            if (Lk[i][item_len - 1][0] <= Lk[j][item_len - 1][0]):
                newItem.append(Lk[i][item_len - 1])
                newItem.append(Lk[j][item_len - 1])
            else:
                newItem.append(Lk[j][item_len - 1])
                newItem.append(Lk[i][item_len - 1])

            # #按照时间 merge
            # while (idx_i < item_len and idx_j < item_len):
            #     if (Lk[i][idx_i][0][1] <= Lk[j][idx_j][0][1]):
            #         newItem.append(Lk[i][idx_i])
            #         idx_i += 1
            #     else:
            #         newItem.append(Lk[j][idx_j])
            #         idx_j += 1
            # newItem.extend(Lk[i][idx_i:])
            # newItem.extend(Lk[j][idx_j:])
            #if (item_len > 1):
                #logging.info("Lk[%d] %s, Lk[%d] %s merged to %s" %(i, Lk[i], j, Lk[j], newItem))
            retList.append(newItem)

    return retList

def aprioriAlgorithm():
    logging.info("entered aprioriAlgorithm")
    print("%s running aprioriAlgorithm" % getCurrentTime())

    C1 = createC1()

    logging.info("C1 has %d items" % (len(C1)))
    logging.debug(C1)
    print("%s C1 has %d items" % (getCurrentTime(), len(C1)))

    #L1 = scanMinSupport(C1, 1)
    L1 = scanMinSupportForC1(C1)
    print("%s %d C1 meet min support" % (getCurrentTime(), len(L1)))

    L1_keys = list(L1.keys())
    L = [ [[key] for key in L1_keys] ]
    logging.debug("L from C1(%d) is %s" % (len(L), L))
    k = 2
    while (len(L[k-2]) > 0):
        #将k-1 项合并成k 项
        Ck = aprioriGen(L[k-2], k)
        logging.info("C%d has %d items " % (k, len(Ck)))
        print("%s C%d has %d items " % (getCurrentTime(), k, len(Ck)))

        #检查 k 项列表中有哪些符合最小支持度
        Lk = scanMinSupport(Ck, k)
        logging.info("%d C%d meet min support" % (len(Lk), k))
        print("%s %d C%d meet min support" % (getCurrentTime(), len(Lk), k))

        L.append(Lk)
        k += 1

    logging.info("frequence items are: ")
    for l_item in L:
        for l_k_item in l_item:
            logging.info(l_k_item)

    print("%s aprioriAlgorithm done" % getCurrentTime())
    logging.info("%s aprioriAlgorithm done" % getCurrentTime())
    return L