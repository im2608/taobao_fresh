from common import *


def doesOperatedCategoryBeforeDate(user_id, item_id, checking_date):

    item_cagetory = global_train_item_category[item_id]
    logging.info("user %s first operated and bought (%s, %s) on same day %s" % (user_id, item_id, item_cagetory, checking_date))
    
    if (user_id in g_user_buy_transection):
        for item_id_opt, item_buy_records in g_user_buy_transection[user_id].items():
            if (global_train_item_category[item_id_opt] != item_cagetory or item_id == item_id_opt):
                continue

            for each_record in item_buy_records:
                if (each_record[0][1].date() < checking_date):
                    logging.info("user %s operated (%s, %s) before %s" % (user_id, item_id_opt, item_cagetory, checking_date))
                    return True

    if (user_id in g_user_behavior_patten):
        for item_id_opt, item_pattern_records in g_user_behavior_patten[user_id].items():
            if (global_train_item_category[item_id_opt] != item_cagetory or item_id == item_id_opt):
                continue
    
            for each_record in item_pattern_records:
                if (each_record[0][1].date() < checking_date):
                    logging.info("user %s operated (%s, %s) before %s" % (user_id, item_id_opt, item_cagetory, checking_date))
                    return True

    return False

# 如果用户第一次操作和购买item在同一天，则检查用户在之前是否操作过该category
def countUserCategoryOnSameDay(start_date, end_date):
    date_idx = start_date
    total_buy_same_day = 0
    operated_before = 0
    doesnot_operated_before = 0
    while (date_idx <= end_date):
        for user_id, item_buy_records in g_user_buy_transection.items():
            for item_id, buy_records in item_buy_records.items():
                for each_record in buy_records:
                    # 当天第一次操作item，当天购买
                    if (each_record[0][1].date() == date_idx and 
                        each_record[-1][1].date() == date_idx):
                        total_buy_same_day += 1
                        if (doesOperatedCategoryBeforeDate(user_id, item_id, date_idx)):
                            operated_before += 1
                        else:
                            doesnot_operated_before += 1 
        date_idx += datetime.timedelta(1)

    print("total buy on same day %s, operated before %d, does not operated before %d" % (total_buy_same_day, operated_before, doesnot_operated_before))

    exit(0)