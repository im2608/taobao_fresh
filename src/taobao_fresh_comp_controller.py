
import subprocess  
import os
import time
import sys
import datetime
import csv
import logging
from common import *
from verify import *

runningPath = sys.path[0]
sys.path.append("%s\\samples\\" % runningPath)


from taking_sample import *

def waitSubprocesses(runningSubProcesses):
    for start_from_user_cnt in runningSubProcesses:
        sub = runningSubProcesses[start_from_user_cnt]
        ret = subprocess.Popen.poll(sub)
        if ret == 0:
            logging.info("subprocess (%s, %s) ended" % (start_from_user_cnt[0], start_from_user_cnt[1]))
            runningSubProcesses.pop(start_from_user_cnt)
            return start_from_user_cnt
        elif ret is None:
            time.sleep(1) # running
        else:
            logging.info("subprocess (%s, %s) terminated" % (start_from_user_cnt[0], start_from_user_cnt[1]))
            runningSubProcesses.pop(start_from_user_cnt)
            return start_from_user_cnt
    return (0, 0)




# 统计用户第一次接触item到购买item之间的天数
def daysBetween1stBehaviorAndBuy():
    loadRecordsFromRedis(0, 17654)

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


def totalVerify(predicted_user_item, end_date):

    loadRecordsFromRedis(0, total_users)

    forecast_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() + datetime.timedelta(1)

    print("totalVerify(): forecast date %s" % forecast_date)

    actual_user_item = takingPositiveSamplesOnDate(forecast_date, True)

    calcuatingF1(forecast_date, predicted_user_item, actual_user_item)

    return 0    

def verification():
    actual_buy = "F:\\doc\\ML\\taobao\\fresh_comp_offline\\taobao_fresh\\input\\tianchi_fresh_comp_train_item.csv"

    acutal_buy_handle = open(actual_buy, encoding="utf-8", mode='r')
    acutal_buy_csv = csv.reader(acutal_buy_handle)

    acutal_buy_dict = dict()

    total_buy = 0

    index = 0

    print("loading file %s" % actual_buy)
    for aline in acutal_buy_csv:
        total_buy += 1
        if (index == 0):
            index += 1
            continue

        index += 1
        item_id = aline[0]

        # if (item_id in acutal_buy_dict):
        #     print("%s has duplication" % item_id)

        acutal_buy_dict[item_id] = 1

    logging.info("acutal_buy_dict len is %d" % len(acutal_buy_dict))

    predicted_user_item = "F:\\doc\\ML\\taobao\\fresh_comp_offline\\taobao_fresh\\output\\forecast.GBDT.LR.4.2016-08-30.1.csv"

    actualbuy_python_handle = open(predicted_user_item, encoding="utf-8", mode='r')
    actualbuy_python_csv = csv.reader(actualbuy_python_handle)

    print("loading file %s" % predicted_user_item)
    index = 0
    hit_count = 0
    for aline in actualbuy_python_csv:
        if (index == 0):
            index += 1
            continue

        index += 1

        user_id = aline[0]
        item_id = aline[1]

        if (item_id not in acutal_buy_dict):
            print("%s is not in test set!" % item_id)

    exit(0)


def submiteOneSubProcess(start_from, user_cnt):
    if (model == 1):
        model_cmd = "taobao_fresh_comp.py"
    elif (model == 2):
        model_cmd = "taobao_fresh_comp_blend_mean.py"
    elif (model == 3):
        model_cmd = "taobao_fresh_comp_blend.py"

    cmdLine = "python %s start_from=%d user_cnt=%d slide=%d topk=%d min_proba=%.2f start=%s end=%s output=1" % (model_cmd, 
        start_from, user_cnt, slide_windows_days, topK, min_proba, start_date, end_date)

    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(start_from, user_cnt)] = sub
    logging.info("running %s" % cmdLine)
    time.sleep(1)
    return

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# daysBetween1stBehaviorAndBuy()
# staticBuyTime()
# verification()

model = int(sys.argv[1].split("=")[1])

logging.basicConfig(level=logging.INFO,\
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
                    datefmt='%a, %d %b %Y %H:%M:%S',\
                    filename='..\\log\\controller.txt',\
                    filemode='w')

start_date = "2014-12-10"
end_date = "2014-12-18"

if (end_date == '2014-12-18'):
    total_users = 17654
else:
    total_users = 1765

users_one_time = 2000
slide_windows_days = 4
topK = 5000
user_for_subprocess = []
start_from = 0
min_proba = 0.5

while (start_from < total_users):
    if (start_from + users_one_time < total_users):
        user_for_subprocess.append((start_from, users_one_time))
        start_from += users_one_time
    else:
        user_for_subprocess.append((start_from, total_users - start_from))
        break

print("user for subprocess are: %s" % user_for_subprocess)

runningSubProcesses = {}
for start_from_user_cnt in user_for_subprocess:
    start_from = start_from_user_cnt[0]
    user_cnt = start_from_user_cnt[1]

    submiteOneSubProcess(start_from, user_cnt)
    logging.info("after submiteOneSubProcess, runningSubProcesses len is %d" % len(runningSubProcesses))
    if (len(runningSubProcesses) == 10):
        while True:
            finished_start_from_user_cnt = waitSubprocesses(runningSubProcesses)
            if ((finished_start_from_user_cnt[0] >= 0 and finished_start_from_user_cnt[1] > 0)):
                logging.info("after waitSubprocesses, runningSubProcesses len is %d" % len(runningSubProcesses))
                break
            if (len(runningSubProcesses) == 0):
                break

while True:
    start_from_user_cnt = waitSubprocesses(runningSubProcesses)
    if (start_from_user_cnt[0] >=0 and start_from_user_cnt[1] > 0):
        logging.info("after waitSubprocesses, runningSubProcesses len is %d" % len(runningSubProcesses))
    if len(runningSubProcesses) == 0:
        break

forecasted_user_item_prob = dict()

output_file_format = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv"
for start_from_user_cnt in user_for_subprocess:
    start_from = start_from_user_cnt[0]
    user_cnt = start_from_user_cnt[1]
    file_idx = 0
    output_file_name = output_file_format % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    
    # 找到 file_index 最大的文件
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = output_file_format % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

    file_idx -= 1
    output_file_name = output_file_format % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    if (not os.path.exists(output_file_name)):
        print("WARNNING: output file does not exist! %s" % output_file_name)
        continue

    print("reading (%d, %d), %s" % (start_from, user_cnt, output_file_name))
    filehandle = open(output_file_name, encoding="utf-8", mode='r')
    csv_reader = csv.reader(filehandle)

    for aline in csv_reader:
        user_id = aline[0]
        item_id = aline[1]
        probility = float(aline[2])
        if ((user_id, item_id) not in forecasted_user_item_prob or 
            probility > forecasted_user_item_prob[(user_id, item_id)]):
            forecasted_user_item_prob[(user_id, item_id)] = probility

file_idx = 0
output_file_name = "%s\\..\\output\\forecast.GBDT.LR.%d.%s.%d.csv" % (runningPath, slide_windows_days, datetime.date.today(), file_idx)

while (os.path.exists(output_file_name)):
    file_idx += 1
    output_file_name = "%s\\..\\output\\forecast.GBDT.LR.%d.%s.%d.csv" % (runningPath, slide_windows_days, datetime.date.today(), file_idx)

sorted_prob = sorted(forecasted_user_item_prob.items(), key=lambda item:item[1], reverse=True)

if (topK > len(sorted_prob)):
    topK = len(sorted_prob)


print("output forecast file %s" % output_file_name)
outputFile = open(output_file_name, encoding="utf-8", mode='w')
outputFile.write("user_id,item_id\n")

for index in range(topK):
    outputFile.write("%s,%s\n" % (sorted_prob[index][0][0], sorted_prob[index][0][1]))

outputFile.close()

if (end_date != '2014-12-18'):
    print("=================================================================")
    print("Total verify (%s, %s): topK %d, min_proba %.2f" % (start_date, end_date, topK, min_proba))
    print("=================================================================")
    totalVerify(sorted_prob[0:topK], end_date)