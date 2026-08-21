import time

def make_timer():
    last = [time.perf_counter()]
    def tick(label=""):
        now = time.perf_counter()
        dt = now - last[0]
        last[0] = now
        if label:
            print(f"{label}: {dt*1000:.2f} ms")
        return dt
    return tick
    

tick = make_timer()

#example usage:
'''
tick("spline")
time.sleep(1)
tick("curvature")
time.sleep(1.1)
tick("minimize")
'''