import subprocess  
import os
import time
import sys
import datetime
import csv

def waitSubprocesses(runningSubProcesses):
    for start_from_user_cnt in runningSubProcesses:
        sub = runningSubProcesses[start_from_user_cnt]
        ret = subprocess.Popen.poll(sub)
        if ret == 0:
            print("subprocess (%s, %s) ended" % (start_from_user_cnt[0], start_from_user_cnt[1]))
            return start_from_user_cnt
        elif ret is None:
            time.sleep(1) # running
        else:
            print("subprocess (%s, %s) terminated" % (start_from_user_cnt[0], start_from_user_cnt[1]))
            runningSubProcesses.pop(start_from_user_cnt)
            return start_from_user_cnt
    return (0, 0)


total_users = 17654
users_one_time = 50
slide_windows_days = 4
topK = 1000
user_for_subprocess = {0:users_one_time, 
                       2000:users_one_time, 
                       # 4000:users_one_time, 
                       # 6000:users_one_time, 
                       # 8000:users_one_time, 
                       # 10000:users_one_time, 
                       # 12000:users_one_time, 
                       # 14000:users_one_time, 
                       # 16000:1654
                       }

runningPath = sys.path[0]

runningSubProcesses = {}
for start_from, user_cnt in user_for_subprocess.items():
    cmdLine = "python taobao_fresh_comp.py start_from=%d user_cnt=%d slide=%d topk=%d" % (start_from, user_cnt, slide_windows_days, topK)
    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(start_from, user_cnt)] = sub
    print("running %s" % cmdLine)
    time.sleep(1)

while True:
    start_from_user_cnt = waitSubprocesses(runningSubProcesses)
    if (start_from_user_cnt[0] >=0 and start_from_user_cnt[1] > 0):
        runningSubProcesses.pop(start_from_user_cnt)

    if len(runningSubProcesses) == 0:
        break

forecasted_user_item_prob = dict()

for start_from, user_cnt in user_for_subprocess.items():
    file_idx = 0
    output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    
    # 找到 file_index 最大的文件
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

    file_idx -= 1
    output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
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

print("output forecast file %s" % output_file_name)
outputFile = open(output_file_name, encoding="utf-8", mode='w')
outputFile.write("user_id,item_id\n")

if (topK > len(sorted_prob)):
    topK = len(sorted_prob)

for index in range(topK):
    outputFile.write("%s,%s\n" % (sorted_prob[index][0][0], sorted_prob[index][0][1]))

outputFile.close()