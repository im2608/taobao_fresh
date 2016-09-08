# -*- coding: utf-8 -*-

from common import *
import math


#ç”¨äºŽéªŒè¯�çš„ç”¨æˆ·è´­ä¹°è¡Œä¸º
g_user_buy_transection_verify = dict()


#æœ€ç»ˆçš„é¢„æµ‹ç»“æžœ
g_final_forecast = dict()
g_buy_record_cnt_forecast = 0.0


g_min_support = 0.0
g_buy_record_cnt_verify = 0.0

g_frequent_item = []

#ä»¥å�•é¡¹å½¢å¼�å¾—åˆ°æ‰€æœ‰çš„ behavior_consecutive, C1 ä¿�å­˜äº† behavior consecutive ä»¥å�Šå¯¹åº”çš„ support
def createC1():
    print("%s creating C1" % getCurrentTime())
    C1 = {}
    #è®°å½•æ¯�ä¸ª behavior_consecutive å‡ºçŽ°åœ¨å“ªäº›è´­ç‰©è®°å½•ä¸­ï¼Œç”¨ä¸‰å…ƒç»„è¡¨ç¤º set( (user id, category, record num), (user id, category, record num) )
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

# ç»Ÿè®¡æ¯�ä¸ªè´­ä¹°è®°å½•ä¸­çš„behavior consecutive æ€»çš„å‡ºçŽ°æ¬¡æ•°ï¼ˆåŒ…æ‹¬ buy record and pattern)
def createC1Ex():
    print("%s creating C1" % getCurrentTime())

    # æ¯�ä¸ª behavior consecutive åœ¨ buy records ä¸­çš„å‡ºçŽ°æ¬¡æ•°ï¼Œ å�³æ”¯æŒ�åº¦
    C1 = {}

    # æ¯�ä¸ªè´­ä¹°è®°å½•ä¸­çš„behavior consecutive åœ¨æ•´ä¸ªæ•°æ�®é›†ä¸­å‡ºçŽ°çš„æ¬¡æ•°
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
#ä»ŽCkä¸­çš„æ¯�ä¸€é¡¹éƒ½æ˜¯ä¸€ä¸ª[], [] ä¸­åŒ…å�«ä¸€ä¸ªæˆ–å¤šä¸ª behavior_consecutiveï¼Œ æŸ¥æ‰¾è¿™äº›behavior_consecutiveçš„ç»„å�ˆæ˜¯å�¦æ»¡è¶³minsupoprt
# ssCnt çš„ç»“æž„ä¸ºï¼š
#  |<----                 frequence item                                                       -------->|  |<-- æ”¯æŒ�åº¦
#  |  |<---    behavior_consecutive            ---->|  |<---        behavior_consecutive        ---->|  |  |
# [[  [(3, datetime.datetime(2014, 12, 14, 7, 0)), 1]ï¼Œ [(4, datetime.datetime(2014, 12, 14, 7, 0)), 1]  ], 1]
#       3:æ“�ä½œç±»åž‹ï¼Œ datetime.datetime(2014, 12, 14, 7, 0)ï¼š æ“�ä½œæ—¶é—´ï¼Œ 1ï¼š æ“�ä½œæ¬¡æ•°

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

#è®¡ç®—è‹¥å¹² behavior_consecutive çš„ç»„å�ˆå�Œæ—¶å‡ºçŽ°åœ¨å“ªäº› buy records ä¸­
def getRecordsContainItems(behavior_consecutives, C1_appearance):
    item_in_records = set()
    for each_behavior in behavior_consecutives:
        if (len(item_in_records) == 0):
            item_in_records = C1_appearance[each_behavior]
        else:
            item_in_records = item_in_records.union(C1_appearance[each_behavior]) ^ (item_in_records ^ C1_appearance[each_behavior])

    return item_in_records

# Lk å†…æ¯�ä¸ªå…ƒç´ å�‡ä¸ºk-1é¡¹ï¼Œ å°†å®ƒä»¬å�ˆå¹¶æˆ� k é¡¹
# è‹¥æœ‰ä¸¤ä¸ªk-1é¡¹é›†ï¼Œæ¯�ä¸ªé¡¹é›†æŒ‰ç…§â€œå±žæ€§-å€¼â€�ï¼ˆä¸€èˆ¬æŒ‰å€¼ï¼‰çš„å­—æ¯�é¡ºåº�è¿›è¡ŒæŽ’åº�ã€‚
# å¦‚æžœä¸¤ä¸ªk-1é¡¹é›†çš„å‰�k-2ä¸ªé¡¹ç›¸å�Œï¼Œè€Œæœ€å�Žä¸€ä¸ªé¡¹ä¸�å�Œï¼Œåˆ™è¯�æ˜Žå®ƒä»¬æ˜¯å�¯è¿žæŽ¥çš„ï¼Œå�³ä¸¤è¿™ä¸ªk-1é¡¹é›†å�¯ä»¥è¿žæŽ¥ï¼Œå�³å�¯è¿žæŽ¥ç”Ÿæˆ� k é¡¹é›†ã€‚
# ä½¿å¦‚æœ‰ä¸¤ä¸ª3é¡¹é›†ï¼šï½›a, b, cï½�{a, b, d}ï¼Œè¿™ä¸¤ä¸ª3é¡¹é›†å°±æ˜¯å�¯è¿žæŽ¥çš„ï¼Œå®ƒä»¬å�¯ä»¥è¿žæŽ¥ç”Ÿæˆ�4é¡¹é›†ï½›a, b, c, dï½�ã€‚
# å�ˆå¦‚ä¸¤ä¸ª3é¡¹é›†ï½›a, b, cï½�ï½›a, d, eï½�ï¼Œè¿™ä¸¤ä¸ª3é¡¹é›†æ˜¯ä¸�èƒ½è¿žæŽ¥ç”Ÿæˆ�3é¡¹é›†çš„ã€‚
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
            #ä¸¤ä¸ªk-1é¡¹é›†çš„å‰�k-2ä¸ªé¡¹ç›¸å�Œï¼Œè€Œæœ€å�Žä¸€ä¸ªé¡¹ä¸�å�Œ, æ‰�å�¯å�ˆå¹¶
            for idx in range(item_len - 1):
                if (Lk[i][idx] != Lk[j][idx]):
                    canMerge = False
                    break

            if (not canMerge):
                #logging.debug("Can not mrege: Lk[%d] %s, Lk[%d] %s" %(i, Lk[i], j, Lk[j]))
                continue

            #å‰� k-2 é¡¹éƒ½ç›¸ç­‰ï¼Œç›´æŽ¥merge
            newItem = []
            for idx in range(item_len - 1):
                newItem.append(Lk[i][idx])

            #å�ˆå¹¶æœ€å�Žä¸€é¡¹
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
    
    #C1_total æ˜¯ C1 çš„å­�é›†ï¼Œ æ„�æ€�æ˜¯å‡ºçŽ°åœ¨ buy records ä¸­çš„ behavior consecutive ä¸�ä¸€å®šå‡ºçŽ°åœ¨ patterns ä¸­
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
        #å°†k-1 é¡¹å�ˆå¹¶æˆ�k é¡¹
        Ck = aprioriGen(L[k-2], k)
        print("%s C%d has %d items " % (getCurrentTime(), k, len(Ck)))

        #æ£€æŸ¥ k é¡¹åˆ—è¡¨ä¸­æœ‰å“ªäº›ç¬¦å�ˆæœ€å°�æ”¯æŒ�åº¦
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
            #æŸ¥æ‰¾æ¯�ä¸ªé¢‘ç¹�é¡¹å‡ºçŽ°åœ¨å“ªäº› user patterns ä¸­
            # user_item_patterns ä¸º set({(user1, item_id1}, (user2, item_id2)}), è¡¨ç¤ºé¢‘ç¹�é¡¹ç¬¦å�ˆ user åœ¨
            # item ä¸Šçš„ pattern
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


