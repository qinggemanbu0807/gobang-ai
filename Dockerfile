# 1. 使用官方 Python 轻量镜像作为基础
FROM python:3.10-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 安装系统依赖（为了让一些 Python 库能正常编译）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 拷贝你的代码到容器里
COPY . .

# 5. 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/temp_code
# 6. 暴露 Streamlit 默认端口 8501
EXPOSE 8501

# 7. 启动命令
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]