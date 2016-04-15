from common import *
import math

#记录的用户在item上的购买行为
g_user_buy_transection = dict()

#用于验证的用户购买行为
g_user_buy_transection_verify = dict()


#最终的预测结果
g_final_forecast = dict()
g_buy_record_cnt_forecast = 0.0

#在用户的操作记录中，从最后一条购买记录到train时间结束之间的操作行为记录，
#以它们作为patten
g_user_behavior_patten = dict()

#总共的购买记录数
g_buy_record_cnt = 0
g_min_support = 0.0
g_buy_record_cnt_verify = 0.0

g_pattern_cnt = 0.0


g_frequent_item = []

def loadDataAndSaveToRedis(need_verify = True, user_opt_file_name = tianchi_fresh_comp_train_user):    
    global g_buy_record_cnt
    global g_min_support
    global g_pattern_cnt

    print("%s loadDataAndSaveToRedis" % getCurrentTime())

    filehandle1 = open(user_opt_file_name, encoding="utf-8", mode='r')

    user_behavior_csv = csv.reader(filehandle1)
    index = 0
    users = ['110883802']

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

        if (user_id not in  users):
            index += 1
            continue

        if (user_id not in user_behavior_record):
            user_behavior_record[user_id] = dict()

        if (item_id not in user_behavior_record[user_id]):
            user_behavior_record[user_id][item_id] = []

        #用户在某个 item id 上的每个操作以（操作类型，操作时间） 的二元组表示
        #user_behavior_seq 为该二元组序列，按照时间排序，相同时间的按照1，2，3，4排序
        user_behavior_seq = user_behavior_record[user_id][item_id]
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

    verify_date = datetime.date(2014, 12, 18)

    #根据操作序列得到用户的购买记录，以及pattern
    print("%s getting user buy records" % getCurrentTime())
    index = 0
    total_user = len(user_behavior_record)
    for user_id, item_id_list in user_behavior_record.items(): #用户， 该用户在哪些 item 上有操作

        for item_id, behavior_seq in item_id_list.items(): 
            #用户在某个 item 上只有一次非购物操作，略过
            if (len(behavior_seq) == 1 and behavior_seq[0][0] != BEHAVIOR_TYPE_BUY):
                continue

            #用户购买记录，按照时间排序，相同时间的话，1，2，3在前，4在后
            #将连续的行为压缩成 {behavior：count}的形式
            #[1,1,1,2,3,4] => [((1, time), 3), ((2, time), 1), ((3, time), 1), ((4, time), 1)]
            sorted_seq = sortAndCompressBuyRecord(behavior_seq)

            user_buy_record = []
            #logging.debug("user: %s, item_id: %s, behavior seq: %s, sorted seq: %s" % (user_id, item_id, behavior_seq, sorted_seq))
            for behavior_consecutive in sorted_seq:
                behavior_type = behavior_consecutive[0][0]
                behavior_time = behavior_consecutive[0][1]
                user_buy_record.append(behavior_consecutive)

                if (behavior_type != BEHAVIOR_TYPE_BUY):
                    continue

                #有些购物记录没有任何浏览记录，跳过
                if (len(user_buy_record) == 0):
                    g_buy_record_cnt += 1
                    skiped_buy_cnt += 1
                    continue

                for idx in range(len(user_buy_record)):
                    #重新生成新的三元组 (操作类型， 操作时间 2014-01-23， 操作次数)
                    user_buy_record[idx] = (user_buy_record[idx][0][0], \
                                            convertDatatimeToStr(user_buy_record[idx][0][1]),\
                                            user_buy_record[idx][1])

                #如果有连续的购买，则为每个购买行为生成一条购物记录
                buy_cnt = behavior_consecutive[1]

                if (need_verify and behavior_time.date() == verify_date):
                    if (user_id not in g_user_buy_transection_verify):
                        g_user_buy_transection_verify[user_id] = dict()

                    if (item_id not in g_user_buy_transection_verify[user_id]):
                        g_user_buy_transection_verify[user_id][item_id] = []

                    #用于验证的用户购买行为
                    for each_buy in range(buy_cnt):
                        g_user_buy_transection_verify[user_id][item_id].append(user_buy_record.copy())
                        g_buy_record_cnt += 1
                else:
                    #用户的购买记录
                    if (user_id not in g_user_buy_transection):
                        g_user_buy_transection[user_id] = dict()
                
                    if (item_id not in g_user_buy_transection[user_id]):
                        g_user_buy_transection[user_id][item_id] = []

                    for each_buy in range(buy_cnt):
                        g_user_buy_transection[user_id][item_id].append(user_buy_record.copy())
                        g_buy_record_cnt += 1

                user_buy_record.clear()

            if (len(user_buy_record) > 0):
                if (user_id not in g_user_behavior_patten):
                    g_user_behavior_patten[user_id] = dict()

                if (item_id not in g_user_behavior_patten[user_id]):
                    g_user_behavior_patten[user_id][item_id] = []

                for idx in range(len(user_buy_record)):                
                    #重新生成新的三元组
                    user_buy_record[idx] = (user_buy_record[idx][0][0], \
                                            convertDatatimeToStr(user_buy_record[idx][0][1]),\
                                            user_buy_record[idx][1])

                g_user_behavior_patten[user_id][item_id].append(user_buy_record.copy())
                g_pattern_cnt += 1
        index += 1
        print("%d /%d users checked\r" % (index, total_user), end="")

    #saveRecordstoRedis()

    logging.info("g_user_behavior_patten %s" % g_user_behavior_patten)
    logging.info("g_user_buy_transection %s" % g_user_buy_transection)

    #logginBuyRecords()

    g_min_support = round(g_buy_record_cnt * 0.01)
    #g_min_support = 50
    logging.info("%s total buy record count is %d(%d), min support is %d" % (getCurrentTime(), g_buy_record_cnt, skiped_buy_cnt, g_min_support))
    print("%s total buy record count is %d(%d), min support is %d, pattern count is %d" %\
          (getCurrentTime(), g_buy_record_cnt, skiped_buy_cnt, g_min_support, g_pattern_cnt))

    filehandle1.close()

    return 0


def convertDatatimeToStr(opt_datatime):
    return "%04d-%02d-%02d %02d" % (opt_datatime.year, opt_datatime.month, opt_datatime.day, opt_datatime.hour)

def logginBuyRecords():
    print("%s logginBuyRecords" % (getCurrentTime()))
    for user_id, item_id_buy in g_user_buy_transection.items():
        buy_cnt = 0
        for item_id, buy_records in item_id_buy.items():
            for each_record in buy_records:
                logging.info("user %s item id %s : %s" % (user_id, item_id, each_record))
            buy_cnt += len(buy_records)
        logging.info("user %s -- %d buy records" % (user_id, buy_cnt))

    for user_id, item_id_opt in g_user_behavior_patten.items():
        for item_id, behavior_pattern in item_id_opt.items():
            logging.info("user %s item category %s  behavior pattern: %s" % (user_id, item_id, behavior_pattern))

    print("%s logginBuyRecords Done" % (getCurrentTime()))
    return 0

def saveRecordstoRedis():
    print("%s saveBuyRecordstoRedis()" % getCurrentTime())
    all_users = list(g_user_buy_transection.keys())
    redis_cli.set("all_users", ",".join(all_users))
    total_user = len(all_users)

    save_one_time = 1000

    idx = 0
    pipe = redis_cli.pipeline()
    for user_id, item_id_buy in g_user_buy_transection.items():        
        item_id_str = list(item_id_buy.keys())
        pipe.hset(user_id, "item_id", ",".join(item_id_str))

        for item_id, buy_records in item_id_buy.items():
            pipe.hset(user_id, item_id, buy_records)

        idx += 1
        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d saved to redis\r" % (idx, total_user), end="")

    if (idx % save_one_time != 0):
        pipe.execute()

    print("")

    all_users_verify = list(g_user_buy_transection_verify.keys())
    total_verify_user = len(all_users_verify)

    redis_cli.set("all_users_verify", ",".join(all_users_verify))

    idx = 0
    for user_id, item_id_buy in g_user_buy_transection_verify.items():
        item_id_str = list(item_id_buy.keys())
        pipe.hset(user_id, "item_id_verify", ",".join(item_id_str))

        for item_id, buy_records in item_id_buy.items():
            pipe.hset(user_id, item_id+"_verify", buy_records)
        idx += 1
        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d verify user saved to redis\r" % (idx, total_verify_user), end="")

    if (idx % save_one_time != 0):
        pipe.execute()

    print("")

    print("%s save patterns to redis" % getCurrentTime())
    idx = 0
    total_user = len(g_user_behavior_patten)
    for user_id, item_id_opt in g_user_behavior_patten.items():        
        item_id_str = list(item_id_opt.keys())
        pipe.hset(user_id, "item_id_pattern", ",".join(item_id_str))

        for item_id, item_pattern in item_id_opt.items():
            pipe.hset(user_id, item_id+"_pattern", item_pattern)
        idx += 1

        if (idx % save_one_time == 0):
            pipe.execute()

        print("%d / %d user pattern saved to redis\r" % (idx, total_user), end="")

    print("")
    if (idx % save_one_time != 0):
        pipe.execute()

    print("%s saveBuyRecordstoRedis() Done!" % getCurrentTime())

    return 0

# 每条购物记录在 redis 中都表现为字符串 
#"[ [(1, 2014-01-01 23, 35), (2, 2014-01-02 22, 1)], [(1, 2014-01-02 23, 35), (2, 2014-01-03 14, 1)] ]"
def loadRecordsFromRedis(min_suport_factor, need_verify):
    global g_buy_record_cnt
    global g_min_support
    global g_buy_record_cnt_verify
    global g_pattern_cnt

    # 得到所有的用户
    all_users = redis_cli.get("all_users").decode()
    all_users = all_users.split(",")

    total_user = len(all_users)
    print("%s loadRecordsFromRedis, here total %d users" % (getCurrentTime(), total_user))

    #根据用户得到用户操作过的item id
    user_index = 0
    skiped_user = 0
    for user_id in all_users:
        if (user_id != '100673077'):
            continue

        #读取购物记录
        g_user_buy_transection[user_id] = dict()

        user_whole_info = redis_cli.hgetall(user_id)

        item_id_list = user_whole_info[bytes("item_id".encode())].decode()
        if (len(item_id_list) > 0):
            item_id_list = item_id_list.split(",")
            for item_id in item_id_list:
                item_buy_record = user_whole_info[bytes(item_id.encode())].decode()
                g_user_buy_transection[user_id][item_id] = getRecordsFromRecordString(item_buy_record)
                logging.info("%s %s buy record %s " % (user_id, item_id, g_user_buy_transection[user_id][item_id]))
                g_buy_record_cnt += len(g_user_buy_transection[user_id][item_id])
        else:
            user_index += 1
            skiped_user += 1

        #得到用户的patterns
        tmp = bytes("item_id_pattern".encode())
        if tmp not in user_whole_info:
            continue

        item_pattern_list = user_whole_info[tmp].decode()
        if (len(item_pattern_list) == 0):
            logging.info("user %s has no patterns!")
            continue

        item_pattern_list = item_pattern_list.split(",")
        g_pattern_cnt += len(item_pattern_list)

        for item_id in item_pattern_list:
            tmp = item_id + "_pattern"
            item_pattern = user_whole_info[bytes(tmp.encode())].decode()
            user_item_pattern = getRecordsFromRecordString(item_pattern)
            # 对于pattern 来说在每个 item_id 上只会有一条，所以直接用 [0]
            for behavior_consecutive in user_item_pattern[0]:
                if (behavior_consecutive not in g_user_behavior_patten):
                    g_user_behavior_patten[behavior_consecutive] = set()

                g_user_behavior_patten[behavior_consecutive].add((user_id, item_id))
            
            logging.info("%s pattern is %s" % (behavior_consecutive, g_user_behavior_patten[behavior_consecutive]))

        user_index += 1
        print("%d / %d users read\r" % (user_index, total_user), end="")

    print("")

    g_min_support = round(g_buy_record_cnt * min_suport_factor)
    if (g_min_support < 1):
        g_min_support = 3

    logging.info("g_buy_record_cnt is %d, g_min_support is %d" % (g_buy_record_cnt, g_min_support))
    print("%s total buy count %d, min support %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_min_support, g_pattern_cnt))
    logging.info("%s total buy count %d, min support %d, pattern count %d " % (getCurrentTime(), g_buy_record_cnt, g_min_support, g_pattern_cnt))

    if (not need_verify):
        return

    # 得到用于验证的用户
    all_users_verify = redis_cli.get("all_users_verify").decode()
    all_users_verify = all_users_verify.split(",")

    total_user = len(all_users_verify)
    idx = 0

    #读取用于验证的购物记录
    for user_id in all_users_verify:
        if (user_id != '100673077'):
            continue

        #读取用于验证的购物记录
        g_user_buy_transection_verify[user_id] = set()

        user_whole_info = redis_cli.hgetall(user_id)

        item_id_list = user_whole_info[bytes("item_id_verify".encode())].decode()
        if (len(item_id_list) == 0):
            continue

        item_id_list = item_id_list.split(",")
        for item_id in item_id_list:
            tmp = item_id + "_verify"
            item_buy_record = user_whole_info[bytes(tmp.encode())].decode()
            records = getRecordsFromRecordString(item_buy_record)

            g_buy_record_cnt_verify += len(records)
            g_user_buy_transection_verify[user_id].add(item_id)
        idx += 1

        print("%d / %d verify user read\r " % (idx, total_user), end="")

    print("%s total verify buy count %d" % (getCurrentTime(), g_buy_record_cnt_verify))

    #logginBuyRecords()
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

#以单项形式得到所有的 behavior_consecutive, C1 保存了 behavior consecutive 以及对应的 support
def createC1():
    print("%s creating C1" % getCurrentTime())
    C1 = {}
    #记录每个 behavior_consecutive 出现在哪些购物记录中，用三元组表示 set( (user id, category, record num), (user id, category, record num) )
    C1_appearance = {}
    C1_support = {}
    for user_id, item_id_buy in g_user_buy_transection.items():
        for item_id, buy_records in item_id_buy.items():
            for record_idx in range(len(buy_records)):
                for behavior_consecutive in buy_records[record_idx]:
                    #logging.info("each_record %s" % each_record)
                    if behavior_consecutive not in C1:
                        C1[behavior_consecutive] = 1
                    else:
                        C1[behavior_consecutive] += 1

                    if (behavior_consecutive not in C1_appearance):
                        C1_appearance[behavior_consecutive] = set()
                    C1_appearance[behavior_consecutive].add((user_id, item_id, record_idx))

    g_user_behavior_patten

    return C1, C1_appearance

# 统计每个购买记录中的behavior consecutive 总的出现次数（包括 buy record and pattern)
def createC1Ex():
    print("%s creating C1" % getCurrentTime())

    # 每个 behavior consecutive 在 buy records 中的出现次数， 即支持度
    C1 = {}

    # 每个购买记录中的behavior consecutive 在整个数据集中出现的次数
    C1_total = {}

    for user_id, item_id_buy in g_user_buy_transection.items():
        for item_id, buy_records in item_id_buy.items():
            for record_idx in range(len(buy_records)):
                for behavior_consecutive in buy_records[record_idx]:
                    #logging.info("each_record %s" % each_record)
                    if behavior_consecutive not in C1:
                        C1[behavior_consecutive] = 1
                    else:
                        C1[behavior_consecutive] += 1

    print("%s creating C1 total" % getCurrentTime())

    for behavior_consecutive, item_id_pattern in g_user_behavior_patten.items():
        if (behavior_consecutive in C1):
            C1_total[behavior_consecutive] = C1[behavior_consecutive] + len(item_id_pattern)

    return C1, C1_total

def outputC1Support(C1):
    logging.info("behaivor consecutive,  support")
    for behavior_consecutive, support in C1.items():
        logging.info("%s, %d" % (behavior_consecutive, support))
    return 0
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

def scanMinSupportWithC1appearance(Ck, k, C1_appearance):
    retList = []
    totalCk = len(Ck)
    index = 0
    for ck_item in Ck:        
        item_in_records = getRecordsContainItems(ck_item, C1_appearance)
        if (len(item_in_records) >= g_min_support):
            retList.append(ck_item)
            logging.debug("ck_item %s appeared in following buy records: %s" % (ck_item, item_in_records))

        index += 1
        print("%s %d / %d C%d items checked!\r" % (getCurrentTime(), index, totalCk, k), end="")

    print("")
       
    return retList

#计算若干 behavior_consecutive 的组合同时出现在哪些 buy records 中
def getRecordsContainItems(behavior_consecutives, C1_appearance):
    item_in_records = set()
    for each_behavior in behavior_consecutives:
        if (len(item_in_records) == 0):
            item_in_records = C1_appearance[each_behavior]
        else:
            item_in_records = item_in_records.union(C1_appearance[each_behavior]) ^ (item_in_records ^ C1_appearance[each_behavior])

    return item_in_records

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
                #logging.debug("Can not mrege: Lk[%d] %s, Lk[%d] %s" %(i, Lk[i], j, Lk[j]))
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

            retList.append(newItem)

    return retList

def Bayes(need_verify):
    global g_buy_record_cnt_forecast
    forecasted_buy_posibility = dict()
    total_records = g_buy_record_cnt + g_pattern_cnt
    buy_posibility = g_buy_record_cnt / total_records
    
    #C1_total 是 C1 的子集， 意思是出现在 buy records 中的 behavior consecutive 不一定出现在 patterns 中
    C1, C1_total = createC1Ex()
    print("%s calculating Bayes P(B|A) = P(A|B)*P(B)/P(A)" % getCurrentTime())

    C1_total_len = len(C1_total)

    idx = 0
    max_posibility = 0
    for behavior_consecutive, behavior_posibislity in C1_total.items():
        posibility = C1[behavior_consecutive] / g_buy_record_cnt * buy_posibility / (C1_total[behavior_consecutive]/total_records)

        for user_item in g_user_behavior_patten[behavior_consecutive]:
            if (user_item not in forecasted_buy_posibility):
                forecasted_buy_posibility[user_item] = posibility
            else:
                forecasted_buy_posibility[user_item] += posibility

        idx += 1
        print("%s %d / %d behavior consecutives calcuated\r" % (getCurrentTime(), idx, C1_total_len), end="")

    outputC1Support(C1)
    outputC1Support(C1_total)

    for user_item in forecasted_buy_posibility:
        forecasted_buy_posibility[user_item] = math.log(forecasted_buy_posibility[user_item])
        if (forecasted_buy_posibility[user_item] > max_posibility):
                max_posibility = forecasted_buy_posibility[user_item]
    
    logging.info("%s max posibility %.2f" % (getCurrentTime(), max_posibility))
    max_posibility = math.log(max_posibility)

    output_file_name = "%s\\..\\output\\forecast.Bayes.csv" % (runningPath)

    if (need_verify):
        output_file_name = "%s\\..\\output\\forecast.verify.Bayes.csv" % (runningPath)

    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    outputFile.write("user_id,item_id\n")
    
    logging.info("user id, item id, posibility")
    print("%s outputting final forecast..." % getCurrentTime())
    lines = 0
    for user_item, posibility in forecasted_buy_posibility.items():
        user_id = user_item[0]
        item_id = user_item[1]

        posibility /= max_posibility
        if (posibility >= 0.5):
            logging.info("%s, %s, %.4f" % (user_id, item_id, posibility))
            if (user_id not in g_final_forecast):
                g_final_forecast[user_id] = set()

            g_final_forecast[user_id].add(item_id)
            g_buy_record_cnt_forecast += 1

    for user_id, item_set in g_final_forecast.items():
        for item_id in item_set:
            outputFile.write("%s,%s\n" % (user_id, item_id))
            lines += 1
            print("%d lines\r" % lines, end="")

    return 0

def aprioriAlgorithm():
    logging.info("entered aprioriAlgorithm")
    print("%s running aprioriAlgorithm" % getCurrentTime())

    C1, C1_appearance, C1_support = createC1()
    outputC1Support(C1_support)

    logging.info("C1 has %d items" % (len(C1)))
    print("%s C1 has %d items" % (getCurrentTime(), len(C1)))

    L1 = scanMinSupportForC1(C1)
    print("%s C1 has %d items meet min support" % (getCurrentTime(), len(L1)))

    L1_keys = list(L1.keys())
    L = [ [[key] for key in L1_keys] ]    
    k = 2
    while (len(L[k-2]) > 0):
        #将k-1 项合并成k 项
        Ck = aprioriGen(L[k-2], k)
        print("%s C%d has %d items " % (getCurrentTime(), k, len(Ck)))

        #检查 k 项列表中有哪些符合最小支持度
        Lk = scanMinSupportWithC1appearance(Ck, k, C1_appearance)
        print("%s C%d has %d items meet min support" % (getCurrentTime(), k, len(Lk)))

        L.append(Lk)
        k += 1

    logging.info("frequent items are: ")
    for l_item in L:
        for l_k_item in l_item:
            logging.info(l_k_item)

    print("%s aprioriAlgorithm done" % getCurrentTime())
    logging.info("%s aprioriAlgorithm done" % getCurrentTime())
    return L

def saveFrequentItemToRedis(frequent_item):
    logging.info("frequent_item %s " % frequent_item)
    total = 0

    for freq_item in frequent_item:
        if (len(freq_item) == 0):
            continue

        key = "frequent_item_%d" % len(freq_item[0])
        redis_cli.set(key, freq_item)
        total += 1

    redis_cli.set("frequent_item_total", total)

    return 0

def loadFrequentItemsFromRedis():
    frequent_item = []
    frequent_item_total = int(redis_cli.get("frequent_item_total").decode())

    logging.info("frequent_item_total %d" % frequent_item_total)

    for idx in range(1, frequent_item_total + 1):
        freq_items = redis_cli.get("frequent_item_%d" % idx).decode()
        freq_items = getRecordsFromRecordString(freq_items)
        frequent_item.append(freq_items)
        logging.info("freq_items %s" % freq_items)
    return frequent_item

def matchPatternAndFrequentItem(frequent_item, factor, need_verify):
    print("%s matchPatternAndFrequentItemEx" % getCurrentTime())
    lines = 0

    global g_buy_record_cnt_forecast

    output_file_name = "%s\\..\\output\\forecast.%.03f.csv" % (runningPath, factor)

    if (need_verify):
        output_file_name = "%s\\..\\output\\forecast.verify.%.03f.csv" % (runningPath, factor)

    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    outputFile.write("user_id,item_id\n")
    total_forecastd_buy = 0
    for i in range(0, len(frequent_item)):
        for each_fre_item in frequent_item[i]:
            #查找每个频繁项出现在哪些 user patterns 中
            # user_item_patterns 为 set({(user1, item_id1}, (user2, item_id2)}), 表示频繁项符合 user 在
            # item 上的 pattern
            user_item_patterns = getRecordsContainItems(each_fre_item, g_user_behavior_patten)
            for user_item in user_item_patterns:
                if (user_item[1] not in global_train_item):
                    continue

                user_id = user_item[0]
                item_id = user_item[1]
                outputFile.write("%s,%s\n" % (user_id, item_id))
                lines += 1
                print("%d lines output\r" % lines, end="")

                if (user_id not in g_final_forecast):
                    g_final_forecast[user_id] = set()

                g_final_forecast[user_id].add(item_id)
                g_buy_record_cnt_forecast += 1

    print("")
    outputFile.close()

    return 0

def verificationForecast():
    hit_item_cnt = 0
    hit_user_cnt = 0
    missed_user_cnt = 0
    f1 = 0.0
    precision = 0.0
    recall = 0.0
    for user_id, forecasted_items in g_final_forecast.items():
        if (user_id not in g_user_buy_transection_verify):
            missed_user_cnt += 1
            continue

        hit_user_cnt += 1
        hit_item_cnt += len(forecasted_items.union(g_user_buy_transection_verify[user_id]) ^ \
                            (forecasted_items ^ g_user_buy_transection_verify[user_id]))

    if (hit_item_cnt != 0):
        precision = hit_item_cnt / g_buy_record_cnt_forecast
        recall = hit_item_cnt / g_buy_record_cnt_verify
        f1 = 2 * precision * recall / (precision + recall)
    else:
        print("hit item count is 0!")
        logging.info("hit item count is 0!")

    logging.info("actual buy count %d, final forecast buy count %d" % (g_buy_record_cnt_verify, g_buy_record_cnt_forecast))
    logging.info("precision %.4f, recall %.4f, f1 %.4f" % (precision, recall, f1))
    logging.info("missed user count %d, hit user count %d, hit item count %d" % (missed_user_cnt, hit_user_cnt, hit_item_cnt))

    print("actual buy count %d, final forecast buy count %d" % (g_buy_record_cnt_verify, g_buy_record_cnt_forecast))
    print("precision %.4f, recall %.4f, f1 %.4f" % (precision, recall, f1))
    print("missed user count %d, hit user count %d, hit item count %d" % (missed_user_cnt, hit_user_cnt, hit_item_cnt))

    return 0    


