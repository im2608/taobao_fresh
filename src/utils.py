import os
import time
import sys
import datetime
import csv
import logging
from common import *
from taking_sample import *
import verify
from global_variables import *

#统计购物的时间段
def staticBuyTime():

    logging.basicConfig(level=logging.INFO,\
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                        datefmt='%a, %d %b %Y %H:%M:%S',\
                        filename='..\\log\\staticBuyTime.txt',\
                        filemode='w')

    runningPath = sys.path[0]
    tianchi_fresh_comp_train_user = "%s\\..\\input\\tianchi_fresh_comp_train_user.csv" % runningPath

    filehandle1 = open(tianchi_fresh_comp_train_user, encoding="utf-8", mode='r')

    user_behavior_csv = csv.reader(filehandle1)
    index = 0

    total_buy = 0

    user_behavior_record = dict()
    week_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i in week_days:
        user_behavior_record[i] = 0

    for i in range(0, 24):
        user_behavior_record[i] = 0

    print("loading file %s" % tianchi_fresh_comp_train_user)
    for aline in user_behavior_csv:
        if (index == 0):
            index += 1
            continue

        index += 1
        if (index % 100000 == 0):
            print("lines read \r", index, end="")

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        user_geohash  = aline[3]
        item_category = aline[4]
        behavior_time = datetime.datetime.strptime(aline[5], "%Y-%m-%d %H")
        if (behavior_type != 4):
            continue

        total_buy += 1

        user_behavior_record[week_days[behavior_time.weekday()]] += 1
        user_behavior_record[behavior_time.time().hour] += 1

    logging.info("total buy %d" % total_buy)
    for i in week_days:
        logging.info("%s buy %d, percentage: %.4f" % (i, user_behavior_record[i], user_behavior_record[i]/total_buy))

    logging.info("")
    for i in range(0, 24):
        logging.info("Time %d:00 buy %d, percentage: %.4f" % (i, user_behavior_record[i], user_behavior_record[i]/total_buy))

    exit(0)

# 输出forecast_date-1 day 加购物车并且没有购买的用户
def rule_12_18_cart(forecast_date):
    prediction = []

    one_day_before_forecast = forecast_date - datetime.timedelta(1)

    if (forecast_date == ONLINE_FORECAST_DATE):
        user_records = g_user_behavior_patten
    else:
        user_records = g_user_buy_transection

    print("(each_behavior[1].hour >=9")

    for user_id, user_opt_records in user_records.items():
        for item_id, user_opt_item_records in user_opt_records.items():
            if (forecast_date == ONLINE_FORECAST_DATE and item_id not in global_test_item_category):
                continue

            for each_record in user_opt_item_records:
                for each_behavior in each_record:
                    if (each_behavior[1].date() == one_day_before_forecast and                         
                        (each_behavior[1].hour >=12 or each_behavior[1].hour == 0) and
                        each_behavior[0] == BEHAVIOR_TYPE_CART):
                        prediction.append(((user_id, item_id), 1))
    return prediction

def test_rule_12_18_cart():
    start_from = 0
    user_cn = 0
    
    print("daysBetweenLastOptAndBuy (%d)..." % user_cn)

    forecast_date = datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()

    logging.basicConfig(level=logging.INFO,\
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                        datefmt='%a, %d %b %Y %H:%M:%S',\
                        filename='..\\log\\daysBetweenLastOptAndBuy.txt',\
                        filemode='w')
    loadTestSet()
    loadRecordsFromRedis(start_from, user_cn)

    prediction = rule_12_18_cart(forecast_date)

    output_file_name = "%s\\..\\output\\rule_last_cart_%s.csv" % (runningPath, datetime.date.today())
    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    outputFile.write("user_id,item_id\n")
    for user_item in prediction:
        user_id = user_item[0][0]
        item_id = user_item[0][1]
        outputFile.write("%s,%s\n" % (user_id, item_id))
    outputFile.close()

    # acutal_buy = takingPositiveSamplesOnDate(forecast_date, True)
    # verify.calcuatingF1(forecast_date, prediction, acutal_buy)

    exit(0)


# 统计每个小时用户加购物车然后在第二天购买的次数
def lastCartBeforBuy():
    user_cn = 0
    print("daysBetweenLastOptAndBuy (%d)..." % user_cn)
    last_opt_between_buy_dict = dict()
    last_opt_between_buy_dict[BEHAVIOR_TYPE_CART] = dict()
    last_opt_between_buy_dict[BEHAVIOR_TYPE_FAV] = dict()

    logging.basicConfig(level=logging.INFO,\
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                        datefmt='%a, %d %b %Y %H:%M:%S',\
                        filename='..\\log\\lastCartBeforBuy.log',\
                        filemode='w')

    output_file_name = "%s\\..\\lastCartBeforBuy.txt" % (runningPath)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')

    loadRecordsFromRedis(0, user_cn)
    for user_id, user_buy_records in g_user_buy_transection.items():
        for item_id, user_buy_item_records in user_buy_records.items():
            for each_record in user_buy_item_records:

                if (each_record[0][1].date() == each_record[-1][1].date()):
                    continue

                buy_date = each_record[-1][1].date()
                one_day_before_buy = buy_date - datetime.timedelta(1)

                for each_behavior in each_record:
                    behavior_type = each_behavior[0]
                    if (behavior_type not in last_opt_between_buy_dict):
                        continue

                    hour = each_behavior[1].hour
                    if (hour not in last_opt_between_buy_dict[behavior_type]):
                        last_opt_between_buy_dict[behavior_type][hour] = 0
                    last_opt_between_buy_dict[behavior_type][hour] += 1

    for behavior_type in last_opt_between_buy_dict:
        sorted_hour = sorted(last_opt_between_buy_dict[behavior_type].items(), key=lambda item:item[0])
        outputFile.write("behaviro type %d\n" % behavior_type)
        for hour in sorted_hour:
            outputFile.write("%d: %d\n" % (hour[0], hour[1]))

    outputFile.close()
    return 0

# 统计用户第一次 behavior 到购物之间的天数
def daysBetween1stBehaviorAndBuy():
    loadRecordsFromRedis(0, 0)

    print("%s calculating days between first behavior to buy..." % (getCurrentTime()))

    days_1st_beahvior_buy_dict = dict()
    
    total_buy = 0

    for user_id, item_id_buy in g_user_buy_transection.items():
        for item_id, buy_records in item_id_buy.items():
            for each_record in buy_records:
                timedelta = each_record[-1][1] - each_record[0][1]
                days = (each_record[-1][1] - each_record[0][1]).days
                if (days not in days_1st_beahvior_buy_dict):
                    days_1st_beahvior_buy_dict[days] = 0
                days_1st_beahvior_buy_dict[days] += 1
                if (total_buy % 1000 == 0):
                    print("    %d  buy records checkd\r" % (total_buy), end="")
                total_buy += 1

    for days, how_man_buy in days_1st_beahvior_buy_dict.items():
        logging.info("days, how_man_buy %d, %d" % (days, how_man_buy))

    days_list = days_1st_beahvior_buy_dict.keys()

    buy_vol = 0
    for days in days_list:
        buy_vol += days_1st_beahvior_buy_dict[days]
        logging.info("first %d days buy %d account for %.2f, total %d " % (days, buy_vol, buy_vol/total_buy, total_buy))

    exit(0)


def MostPopular():
    user_cn = 0
    print("MostPopular (%d)..." % user_cn)
    logging.basicConfig(level=logging.INFO,\
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                        datefmt='%a, %d %b %Y %H:%M:%S',\
                        filename='..\\log\\MostPopular.log',\
                        filemode='w')

    loadRecordsFromRedis(0, user_cn)

    # 用户购买不同item的数量
    most_buy_users_dict = dict()

    # 购买item的用户数
    most_sell_items_dict = dict()

    for user_id, user_buy_records in g_user_buy_transection.items():
        most_buy_users_dict[user_id] = len(user_buy_records)

    for item_id, user_buy_records in g_user_buy_transection_item.items():
        most_sell_items_dict[item_id] = len(user_buy_records)

    output_file_name = "%s\\..\\MostPopular.txt" % (runningPath)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')

    sorted_buy_users = sorted(most_buy_users_dict.items(), key=lambda item:item[1], reverse=True)
    for user_buy_cnt in sorted_buy_users:
        user_id = user_buy_cnt[0]
        buy_cnt = user_buy_cnt[1]
        outputFile.write("user %s bought %d itmes : %s\n" % (user_id, buy_cnt, list(g_user_buy_transection[user_id].keys())))

    sorted_item_sell = sorted(most_sell_items_dict.items(), key=lambda item:item[1], reverse=True)
    for item_sell_cnt in sorted_item_sell:
        item_id = item_sell_cnt[0]
        sell_cnt = item_sell_cnt[1]
        outputFile.write("item %s sell %d counts : %s\n" % (item_id, sell_cnt, list(g_user_buy_transection_item[item_id].keys())))


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

if __name__ == '__main__':  
    # daysBetween1stBehaviorAndBuy()
    # staticBuyTime()
    # verification()
    # test_rule_12_18_cart()
    # lastCartBeforBuy()
    MostPopular()