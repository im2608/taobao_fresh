from common import *
import apriori


buy_records_mysql = dict()
buy_records_python = dict()


def verifyBuyRecords():
    file_user_buy_record_mysql = open("%s\\..\\input\\buy_record_mysql.csv" % runningPath, encoding="utf-8", mode='r')
    user_buy_records_csv = csv.reader(file_user_buy_record_mysql)

    file_buy_records_python = open("%s\\..\\input\\buy_record_python.csv" % runningPath, encoding="utf-8", mode='r')
    buy_records_python_csv = csv.reader(file_buy_records_python)

    index = 0

    for aline in user_buy_records_csv:
        if (index == 0):
            index += 1
            continue
        user_id = aline[0]
        buy_records_cnt = int(aline[1])

        buy_records_mysql[user_id] = buy_records_cnt;

    index = 0
    for aline in buy_records_python_csv:
        if (index == 0):
            index += 1
            continue

        user_id = aline[0]
        buy_records_cnt = int(aline[1])

        buy_records_python[user_id] = buy_records_cnt;

    logging.info("users %d from mysql" % len(buy_records_mysql))
    logging.info("users %d from python" % len(buy_records_python))

    for user_id, buy_records_cnt in buy_records_mysql.items():
        if (user_id not in buy_records_python):
            logging.info("Error: %s not in Buy-Records-Python" % user_id)
            continue

        if (buy_records_cnt != buy_records_python[user_id]):
            logging.info("Error: user %s Mysql (%d) != Python (%d)" % (user_id, buy_records_cnt, buy_records_python[user_id]))

    return 0


file_idx = 37
data_file = "%s\\..\\input\\splitedInput\\datafile.%03d" % (runningPath, file_idx)

#apriori.loadData(True)
apriori.loadRecordsFromRedis()
apriori.saveRecordstoRedis()
#verifyBuyRecords()