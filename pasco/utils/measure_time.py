import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Thời gian thực hiện {func.__name__}: {end_time - start_time:.6f} giây")
        return result
    return wrapper
def measure_time_for_class(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value):  # Chỉ áp dụng với phương thức
            setattr(cls, attr_name, measure_time(attr_value))
    return cls