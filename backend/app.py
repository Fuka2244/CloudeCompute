from flask import Flask, jsonify
import redis
import os

app = Flask(__name__)

# 从环境变量获取 Redis 配置 (docker-compose 注入的)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
if REDIS_PASSWORD == "":
    REDIS_PASSWORD = None

# 连接 Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

@app.route('/api/hello')
def hello():
    # try:
    #     # 每次访问 API，Redis 中的 'hits' 键值加 1
    count = r.incr('hits')
    return jsonify({
        "status": "success",
        "message": "Flask 后端与 Redis 连接正常！", 
        "hits": count
    })
    # except redis.exceptions.ConnectionError:
    #     return jsonify({"status": "error", "message": "无法连接到 Redis"}), 500

if __name__ == '__main__':
    # 使用 0.0.0.0 允许外部容器访问
    app.run(host='0.0.0.0', port=5000)