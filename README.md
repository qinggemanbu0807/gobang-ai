# 五子棋 AI 项目

一个基于 Streamlit 的五子棋游戏，支持玩家与 AI 对战，并允许通过编写 Python 代码来自定义 AI 策略。

## 功能特性

- 🎮 15x15 标准五子棋棋盘
- 🤖 内置简单 AI 对手
- 💻 支持自定义 Python 代码定义 AI 策略
- 🔌 预留 Qwen API 调用接口
- 📊 实时显示游戏状态和落子历史
- 🐳 Docker 容器安全执行用户代码（`run_code_safely` 函数）

## 安装步骤

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Docker（可选，用于安全代码执行）

如果使用 `run_code_safely` 函数在 Docker 容器中执行代码，需要安装 Docker：

- **Windows/Mac**: 下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 
  ```bash
  sudo apt-get update
  sudo apt-get install docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  ```

确保 Docker 服务正在运行：
```bash
docker --version
docker ps
```

### 3. 配置环境变量（可选）

如果需要使用 Qwen API，可以设置环境变量：

```bash
# Windows PowerShell
$env:QWEN_API_KEY="your-api-key-here"
$env:QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Linux/Mac
export QWEN_API_KEY="your-api-key-here"
export QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

或者在应用界面的侧边栏中直接配置。

## 运行方法

在项目根目录下运行：

```bash
streamlit run app.py
```

浏览器会自动打开应用界面（通常是 `http://localhost:8501`）。

## 使用说明

### 基本游戏

1. 点击侧边栏的坐标输入框选择落子位置（行和列，范围 0-14）
2. 点击"落子"按钮下棋
3. AI 会自动下棋
4. 先连成五子的一方获胜

### 自定义 AI 策略

1. 在侧边栏的"AI 策略代码"文本框中编写 Python 代码
2. 代码需要定义一个变量 `next_move = (row, col)` 表示下一步棋的位置
3. 代码中可以使用以下变量：
   - `board`: 15x15 的 numpy 数组，0表示空位，1表示黑子，2表示白子
   - `current_player`: 当前玩家（1表示黑子，2表示白子）
4. 勾选"使用自定义 AI 代码"复选框
5. AI 将使用你编写的代码进行下棋

### 代码示例

```python
# 随机选择空位
empty_positions = [(r, c) for r in range(15) for c in range(15) if board[r][c] == 0]
if empty_positions:
    import random
    next_move = random.choice(empty_positions)
else:
    next_move = (7, 7)
```

## Qwen API 接口

项目中预留了 `call_qwen_api()` 函数用于调用 Qwen API。可以在自定义 AI 代码中使用：

```python
# 示例：使用 Qwen API 生成策略（需要实现具体的调用逻辑）
# messages = [{"role": "user", "content": "..."}]
# result = call_qwen_api(messages)
```

## Docker 安全代码执行

项目提供了 `run_code_safely(user_code)` 函数，可以在 Docker 容器中安全执行用户代码。

### 功能特点

- ✅ 使用 `python:3.9-slim` 轻量级镜像
- ✅ 2 秒执行超时限制
- ✅ 128MB 内存限制
- ✅ 禁用网络访问
- ✅ 只读文件系统
- ✅ 自动清理临时文件和容器

### 使用示例

```python
from app import run_code_safely

user_code = """
print("Hello, World!")
for i in range(5):
    print(i)
"""

success, output = run_code_safely(user_code)
if success:
    print("执行成功:")
    print(output)
else:
    print("执行失败:")
    print(output)
```

### Dockerfile

项目包含 `Dockerfile`，可以用于构建自定义执行环境：

```bash
docker build -t python-code-executor .
```

默认情况下，`run_code_safely` 函数直接使用 `python:3.9-slim` 镜像，无需构建。

## 项目结构

```
cloud_python_lab/
├── app.py              # 主应用文件
├── requirements.txt    # Python 依赖列表
├── Dockerfile          # Docker 镜像配置文件
└── README.md          # 项目说明文档
```

## 注意事项

- 棋盘坐标从 (0, 0) 开始，到 (14, 14) 结束
- 黑子先手（玩家），白子后手（AI）
- 自定义代码执行在受限环境中，仅包含基本的 Python 内置函数和 numpy
- 如果自定义代码执行失败，将自动回退到简单 AI 策略
- 使用 `run_code_safely` 函数需要 Docker 环境，确保 Docker 已安装并运行
- Docker 容器执行有 2 秒超时和 128MB 内存限制，不适合执行复杂或长时间运行的任务

## 开发计划

- [ ] 添加点击棋盘直接落子的功能
- [ ] 实现更强大的 AI 算法（如 Minimax）
- [ ] 集成 Qwen API 实现智能 AI
- [ ] 添加游戏回放功能
- [ ] 支持保存/加载游戏
