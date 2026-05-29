import threading
import requests
import time

# 公网 IP
url = "http://120.46.131.120/api/hello"  

def pull_requests():
    while True:
        try:
            # 疯狂发送请求
            requests.get(url, timeout=2)
        except:
            pass

print("🔥 压测开始，正在通过 50 个线程并发轰炸后端，持续 2 分钟...")

# 开启 50 个线程同时发请求，这足以让单核 100m 的 Pod 瞬间 CPU 飙过 60%
for i in range(50):
    t = threading.Thread(target=pull_requests)
    t.daemon = True
    t.start()

time.sleep(120)  # 持续轰炸 2 分钟
print("🛑 压测结束！流量已停止。")