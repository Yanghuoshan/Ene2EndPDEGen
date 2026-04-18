import time


def display_current_data_time():
    """显示当前时间"""
    local_time = time.localtime()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    print(current_time)