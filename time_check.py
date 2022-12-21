import datetime
import sys
import time

def check():
    today = datetime.datetime.today()
    if today.weekday() >= 5:
        sys.exit("주말")

    today_9am = datetime.datetime.combine(datetime.date.today(), datetime.time(9,0,0))
    time_diff = today - today_9am
    while time_diff.total_seconds() <= 0:
        time.sleep(time_diff.total_seconds())

    today_3pm = datetime.datetime.combine(datetime.date.today(), datetime.time(15,20,0))
    if (today - today_3pm).total_seconds() >= 0:
        sys.exit("장 마감")

    print(f"현재시간: {datetime.datetime.today()}")