FROM nginx:1.25-alpine
# 将自定义 nginx 配置覆盖默认配置
COPY nginx.conf /etc/nginx/conf.d/default.conf
# 将前端静态文件复制到 Nginx 默认目录
COPY static/ /usr/share/nginx/html/
# 暴露 80 端口
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]