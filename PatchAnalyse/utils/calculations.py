import time

def time_normalization(alltime):
    a = int(alltime / 3600)
    b = int((alltime - a * 3600) / 60)
    c = alltime % 60
    HMS = str(a) + "小时:" + str(b) + "分钟:" + str(c) + "秒"
    return HMS


def total_consumption(time_start=None,time_end=None):
    time_consuming = time_end - time_start
    a = int(time_consuming / 3600)
    b = int((time_consuming - a * 3600) / 60)
    c = time_consuming % 60
    HMS = str(a) + "小时:" + str(b) + "分钟:" + str(c) + "秒"
    return time_consuming,HMS

def time_predict(time_start,time_end,now_times,total_times):
    """
    预估程序运行还需多少时间
    :return:
    """
    time_consuming,HMS=total_consumption(time_start,time_end)
    time_rest_second=time_consuming*(total_times-1-now_times)
    a = int(time_rest_second / 3600)
    b = int((time_rest_second - a * 3600) / 60)
    c = time_rest_second % 60
    HMS = str(a) + "小时:" + str(b) + "分钟:" + str(c) + "秒"
    return HMS,time_rest_second
