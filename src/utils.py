import time


def display_current_data_time():
    """显示当前时间"""
    local_time = time.localtime()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    print(current_time)


def count_parameters(model, name="Model"):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in {name}: {total_params:,}")