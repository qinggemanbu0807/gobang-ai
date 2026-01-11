import streamlit as st
import numpy as np
from typing import Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import os
import io
import contextlib
import re
import tempfile
import docker
import time
from pathlib import Path
import nbformat
from nbconvert import PythonExporter
# ... 其他已有的 import ...

# 页面配置
st.set_page_config(
    page_title="五子棋 AI",
    page_icon="🎮",
    layout="wide"
)

# 初始化状态
if 'board' not in st.session_state:
    st.session_state.board = np.zeros((15, 15), dtype=int)  # 0: 空, 1: 黑子, 2: 白子
if 'current_player' not in st.session_state:
    st.session_state.current_player = 1  # 1: 黑子(玩家), 2: 白子(AI)
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'move_history' not in st.session_state:
    st.session_state.move_history = []

# Qwen API 配置（预留接口）
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 更新 Qwen API Key（从 session_state 中获取）
if 'qwen_api_key' in st.session_state:
    QWEN_API_KEY = st.session_state.qwen_api_key


def call_qwen_api(messages: list, model: str = "qwen-turbo") -> Optional[str]:
    """
    预留的 Qwen API 调用接口
    """
    # 获取最新的 API Key
    api_key = os.getenv("QWEN_API_KEY", "")
    if 'qwen_api_key' in st.session_state:
        api_key = st.session_state.qwen_api_key
    
    if not api_key:
        return None
    
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=QWEN_API_BASE
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Qwen API 调用失败: {str(e)}")
        return None


def board_to_text(board: np.ndarray) -> str:
    """
    将棋盘状态转换为文本描述
    """
    lines = []
    lines.append("当前棋盘状态（15x15）：")
    lines.append("   " + " ".join([f"{i:2d}" for i in range(15)]))
    for r in range(15):
        row_str = f"{r:2d} "
        for c in range(15):
            if board[r][c] == 0:
                row_str += " . "
            elif board[r][c] == 1:
                row_str += " ● "  # 黑子
            else:
                row_str += " ○ "  # 白子
        lines.append(row_str)
    return "\n".join(lines)


def run_notebook_logic(uploaded_file):
    """
    解析上传的 .ipynb，并在当前 Docker 目录内运行
    """
    try:
        # 1. 解析 Notebook 提取代码
        nb = nbformat.reads(uploaded_file.read().decode("utf-8"), as_version=4)
        exporter = PythonExporter()
        source_code, _ = exporter.from_notebook_node(nb)
        
        # 2. 动态获取当前目录（防止路径报错的关键！）
        # 在 Docker 里，这通常是 /app
        current_dir = os.path.dirname(os.path.abspath(__file__))
        user_code_path = os.path.join(current_dir, "user_code.py")
        
        # 3. 写入文件
        with open(user_code_path, "w", encoding="utf-8") as f:
            f.write(source_code)
            
        # 4. 执行命令：直接 python user_code.py
        import subprocess
        result = subprocess.run(
            ["python", "user_code.py"], 
            capture_output=True, 
            text=True,
            cwd=current_dir # 强制在当前工作目录运行
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"执行报错:\n{result.stderr}"
            
    except Exception as e:
        return f"系统处理失败: {str(e)}"


def get_qwen_move(board_state: np.ndarray, current_player: int = 2) -> Optional[Tuple[int, int]]:
    """
    使用 Qwen API 获取下一步棋的坐标
    
    Args:
        board_state: 15x15 的二维数组，0表示空位，1表示黑子，2表示白子
        current_player: 当前玩家（1表示黑子，2表示白子）
    
    Returns:
        下一步棋的坐标 (row, col)，如果失败返回 None
    """
    # 获取最新的 API Key
    api_key = os.getenv("QWEN_API_KEY", "")
    if 'qwen_api_key' in st.session_state:
        api_key = st.session_state.qwen_api_key
    
    if not api_key:
        return None
    
    # 将棋盘转换为文本格式
    board_text = board_to_text(board_state)
    
    # 统计已下棋子数量，判断当前是第几步
    move_count = np.sum(board_state > 0)
    player_name = "白子" if current_player == 2 else "黑子"
    
    # 构造提示词
    system_prompt = """你是一位五子棋高手。你需要分析当前棋盘状态，并给出最优的下一步落子位置。

规则：
1. 棋盘是15x15的网格，坐标从(0,0)到(14,14)
2. 0表示空位，1表示黑子，2表示白子
3. 你需要让同一颜色的棋子连成5个（横、竖、斜都可以）即可获胜
4. 同时要阻止对手连成5个

请仔细分析棋盘，考虑：
- 是否有立即获胜的机会
- 是否需要阻止对手获胜
- 如何形成自己的攻击组合
- 如何阻止对手形成威胁

请只返回坐标，格式为：(row, col)，例如：(7, 7) 或 (3, 10)
不要返回其他解释文字，只返回坐标。"""

    user_prompt = f"""当前棋盘状态：

{board_text}

当前是第 {move_count + 1} 步，轮到 {player_name}（玩家 {current_player}）下棋。

请给出最优的下一步落子坐标，格式：(row, col)"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 调用 Qwen API
    response = call_qwen_api(messages, model="qwen-turbo")
    
    if not response:
        return None
    
    # 解析返回的坐标
    try:
        # 尝试从响应中提取坐标
        # 匹配 (数字, 数字) 格式
        match = re.search(r'\((\d+)\s*,\s*(\d+)\)', response)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            # 验证坐标是否有效
            if 0 <= row < 15 and 0 <= col < 15:
                return (row, col)
        
        # 如果没有找到括号格式，尝试其他格式
        match = re.search(r'(\d+)\s*,\s*(\d+)', response)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            if 0 <= row < 15 and 0 <= col < 15:
                return (row, col)
        
        # 如果还是没找到，尝试提取两个数字
        numbers = re.findall(r'\d+', response)
        if len(numbers) >= 2:
            row = int(numbers[0])
            col = int(numbers[1])
            if 0 <= row < 15 and 0 <= col < 15:
                return (row, col)
        
        return None
    except Exception as e:
        st.warning(f"解析 Qwen 返回结果失败: {str(e)}，原始响应: {response}")
        return None


def check_winner(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """检查是否有玩家获胜"""
    directions = [
        [(0, 1), (0, -1)],   # 水平
        [(1, 0), (-1, 0)],   # 垂直
        [(1, 1), (-1, -1)],  # 主对角线
        [(1, -1), (-1, 1)]   # 副对角线
    ]
    
    for direction_pair in directions:
        count = 1  # 包括当前落子
        for dx, dy in direction_pair:
            r, c = row + dx, col + dy
            while 0 <= r < 15 and 0 <= c < 15 and board[r][c] == player:
                count += 1
                r += dx
                c += dy
        
        if count >= 5:
            return True
    
    return False


def execute_user_code(code: str, board: np.ndarray, current_player: int) -> Optional[Tuple[int, int]]:
    """
    执行用户编写的 Python 代码，返回下一步棋的坐标 (row, col)
    """
    try:
        # 创建安全的执行环境
        safe_globals = {
            'np': np,
            'board': board.copy(),
            'current_player': current_player,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'min': min,
            'max': max,
        }
        
        # 捕获输出
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with contextlib.redirect_stderr(output):
                exec(code, safe_globals)
        
        # 尝试获取返回值
        result = safe_globals.get('next_move')
        output_str = output.getvalue()
        
        if result and isinstance(result, (tuple, list)) and len(result) == 2:
            row, col = result
            if isinstance(row, (int, np.integer)) and isinstance(col, (int, np.integer)):
                return int(row), int(col)
        
        return None, output_str
    except Exception as e:
        return None, f"代码执行错误: {str(e)}"


def run_code_safely(user_code: str) -> Tuple[bool, str]:
    """
    在 Docker 容器中安全执行用户代码
    
    Args:
        user_code: 用户编写的 Python 代码字符串
    
    Returns:
        (success, output): 成功标志和输出内容
        success: True 表示执行成功，False 表示执行失败或超时
        output: stdout 的输出内容，如果失败则包含错误信息
    """
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    code_file = os.path.join(temp_dir, "user_code.py")
    
    try:
        # 写入用户代码到临时文件
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(user_code)
        
        # 初始化 Docker 客户端
        try:
            client = docker.from_env()
        except Exception as e:
            return False, f"Docker 客户端初始化失败: {str(e)}。请确保 Docker 已安装并正在运行。"
        
        # 确保 Docker 镜像存在
        try:
            client.images.get("python:3.9-slim")
        except docker.errors.ImageNotFound:
            try:
                st.info("正在拉取 python:3.9-slim 镜像，请稍候...")
                client.images.pull("python:3.9-slim")
            except Exception as e:
                return False, f"拉取 Docker 镜像失败: {str(e)}"
        
        # 准备容器配置
        # 使用低级别 API 创建 host_config
        host_config = client.api.create_host_config(
            mem_limit='128m',  # 限制内存 128MB
            memswap_limit='128m',  # 限制交换内存
            network_mode='none',  # 禁用网络访问
            cpu_period=100000,  # CPU 限制配置
            cpu_quota=50000,  # 限制 CPU 使用率（50%）
            pids_limit=10,  # 限制进程数
            read_only=True,  # 只读文件系统
            tmpfs={'/tmp': 'size=64m'},  # 临时文件系统
            binds=[f'{temp_dir}:/code:ro'],  # 只读挂载代码目录
        )
        
        # 创建并启动容器
        container = None
        try:
            # 使用低级别 API 创建容器
            container_dict = client.api.create_container(
                image='python:3.9-slim',
                command=['python', 'user_code.py'],
                working_dir='/code',
                host_config=host_config,
            )
            container_id = container_dict['Id']
            container = client.containers.get(container_id)
            container.start()
            
            # 等待容器执行完成，最多等待 2 秒
            start_time = time.time()
            timeout = 2.0
            
            while container.status == 'running':
                if time.time() - start_time > timeout:
                    # 超时，强制停止容器
                    try:
                        container.stop(timeout=1)
                    except Exception:
                        pass
                    try:
                        container.remove()
                    except Exception:
                        pass
                    return False, f"代码执行超时（超过 {timeout} 秒）"
                
                time.sleep(0.1)
                container.reload()
            
            # 获取容器输出
            logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='ignore')
            
            # 检查容器退出状态
            container.reload()
            exit_code = container.attrs.get('State', {}).get('ExitCode', -1)
            
            # 清理容器
            try:
                container.remove()
            except Exception:
                pass
            
            if exit_code == 0:
                return True, logs
            else:
                return False, f"代码执行失败（退出码: {exit_code}）\n{logs}"
                
        except docker.errors.ContainerError as e:
            if container:
                try:
                    container.remove()
                except Exception:
                    pass
            return False, f"容器执行错误: {str(e)}"
        except Exception as e:
            if container:
                try:
                    container.stop(timeout=1)
                    container.remove()
                except Exception:
                    pass
            return False, f"执行过程中发生错误: {str(e)}"
            
    except Exception as e:
        return False, f"准备执行环境失败: {str(e)}"
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def ai_move_simple(board: np.ndarray, current_player: int) -> Tuple[int, int]:
    """
    简单的AI落子策略（示例）
    """
    # 检查是否有获胜机会
    for r in range(15):
        for c in range(15):
            if board[r][c] == 0:
                board[r][c] = current_player
                if check_winner(board, r, c, current_player):
                    board[r][c] = 0
                    return r, c
                board[r][c] = 0
    
    # 检查是否需要防守
    opponent = 3 - current_player
    for r in range(15):
        for c in range(15):
            if board[r][c] == 0:
                board[r][c] = opponent
                if check_winner(board, r, c, opponent):
                    board[r][c] = 0
                    return r, c
                board[r][c] = 0
    
    # 随机选择空位
    empty_positions = [(r, c) for r in range(15) for c in range(15) if board[r][c] == 0]
    if empty_positions:
        return empty_positions[np.random.randint(len(empty_positions))]
    
    return 7, 7  # 默认中心位置


def draw_board(board: np.ndarray, last_move: Optional[Tuple[int, int]] = None):
    """绘制棋盘"""
    fig = go.Figure()
    
    # 绘制棋盘网格
    for i in range(15):
        fig.add_trace(go.Scatter(
            x=[i, i],
            y=[0, 14],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 14],
            y=[i, i],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 绘制棋子
    for r in range(15):
        for c in range(15):
            if board[r][c] == 1:  # 黑子
                fig.add_trace(go.Scatter(
                    x=[c],
                    y=[r],
                    mode='markers',
                    marker=dict(size=25, color='black', symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            elif board[r][c] == 2:  # 白子
                fig.add_trace(go.Scatter(
                    x=[c],
                    y=[r],
                    mode='markers',
                    marker=dict(size=25, color='white', symbol='circle', 
                              line=dict(color='black', width=2)),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 标记最后一步
    if last_move:
        r, c = last_move
        fig.add_trace(go.Scatter(
            x=[c],
            y=[r],
            mode='markers',
            marker=dict(size=30, color='red', symbol='circle', 
                       line=dict(color='darkred', width=2), opacity=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, 14.5], showgrid=False, zeroline=False, 
                  showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-0.5, 14.5], showgrid=False, zeroline=False, 
                  showticklabels=False),
        plot_bgcolor='burlywood',
        width=600,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig


def reset_game():
    """重置游戏"""
    st.session_state.board = np.zeros((15, 15), dtype=int)
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.move_history = []


# 主界面
st.title("🎮 五子棋 AI 对战")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
    # ---.ipynb 上传功能 ---
    st.subheader("📁 云端小电脑")
    uploaded_ipynb = st.file_uploader("上传五子棋脚本 (.ipynb)", type="ipynb")
    
    if uploaded_ipynb:
        # 调用我们之前说的解析函数（记得把函数定义放在 app.py 前面）
        st.info("正在 Docker 环境中解析并运行...")
        # 这里的 run_notebook_logic 是我们要加的新函数
        # output = run_notebook_logic(uploaded_ipynb) 
        # st.success("运行成功！")
    
    st.divider() # 画条分割线，显得专业
    
    # --- 原有的 API Key 输入框 ---
    with st.expander("Qwen API 配置", expanded=True):
        api_key = st.text_input("API Key", value=st.session_state.get('qwen_api_key', ""), type="password")
        if api_key:
            st.session_state.qwen_api_key = api_key
            os.environ["QWEN_API_KEY"] = api_key

    if st.button("🔄 重置游戏"):
        st.session_state.board = np.zeros((15, 15), dtype=int)
        st.session_state.move_history = []
        st.session_state.game_over = False
        st.rerun()
    
    if st.button("🔄 重新开始", use_container_width=True):
        reset_game()
        st.rerun()
    
    st.divider()
    
    st.subheader("AI 策略代码")
    st.caption("编写 Python 代码来定义 AI 的下棋策略")
    st.caption("代码需要定义一个变量 `next_move = (row, col)` 表示下一步棋的位置")
    
    default_code = """# 示例：随机选择空位
empty_positions = [(r, c) for r in range(15) for c in range(15) if board[r][c] == 0]
if empty_positions:
    import random
    next_move = random.choice(empty_positions)
else:
    next_move = (7, 7)"""
    
    ai_code = st.text_area(
        "AI 代码",
        value=default_code,
        height=200,
        help="使用 board (numpy数组) 和 current_player (1或2) 来编写策略"
    )
    
    use_custom_ai = st.checkbox("使用自定义 AI 代码", value=False)
    
    st.divider()
    
    st.subheader("Qwen API 配置")
    # 获取当前 API Key（优先使用 session_state 中的值）
    current_api_key = st.session_state.get('qwen_api_key', QWEN_API_KEY)
    qwen_api_key_input = st.text_input("API Key", value=current_api_key, type="password", key="unique_qwen_api_key_field")
    
    # 如果输入发生变化，更新 session_state 和环境变量
    if qwen_api_key_input != current_api_key:
        st.session_state.qwen_api_key = qwen_api_key_input
        os.environ["QWEN_API_KEY"] = qwen_api_key_input
    
    use_qwen_ai = st.checkbox("使用 Qwen AI", value=False, 
                              help="使用 Qwen API 进行智能下棋")
    
    if use_qwen_ai and not qwen_api_key_input:
        st.warning("⚠️ 请先输入 Qwen API Key")
    
    st.divider()
    
    st.subheader("游戏状态")
    if st.session_state.game_over:
        if st.session_state.winner == 1:
            st.success("🏆 黑子（玩家）获胜！")
        elif st.session_state.winner == 2:
            st.info("🤖 白子（AI）获胜！")
        else:
            st.warning("平局")
    else:
        if st.session_state.current_player == 1:
            st.info("当前回合：黑子（玩家）")
        else:
            st.info("当前回合：白子（AI）")
    
    st.caption(f"已下步数: {len(st.session_state.move_history)}")

# 主区域
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("棋盘")
    
    # 使用 Plotly 绘制棋盘
    fig = draw_board(
        st.session_state.board,
        st.session_state.move_history[-1] if st.session_state.move_history else None
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 点击棋盘落子（简化版本：使用坐标输入）
    if not st.session_state.game_over and st.session_state.current_player == 1:
        st.subheader("落子")
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            row_input = st.number_input("行 (0-14)", min_value=0, max_value=14, value=7, step=1)
        with col_input2:
            col_input = st.number_input("列 (0-14)", min_value=0, max_value=14, value=7, step=1)
        
        if st.button("🎯 落子", use_container_width=True, type="primary"):
            if st.session_state.board[row_input][col_input] == 0:
                st.session_state.board[row_input][col_input] = 1
                st.session_state.move_history.append((row_input, col_input))
                
                # 检查是否获胜
                if check_winner(st.session_state.board, row_input, col_input, 1):
                    st.session_state.game_over = True
                    st.session_state.winner = 1
                    st.rerun()
                else:
                    st.session_state.current_player = 2
                    st.rerun()
            else:
                st.error("该位置已有棋子！")

with col2:
    st.subheader("代码执行结果")
    
    # 显示AI代码执行结果
    if st.session_state.current_player == 2 and not st.session_state.game_over:
        st.info("AI 正在思考...")
        
        # 优先使用 Qwen AI
        if use_qwen_ai and qwen_api_key_input:
            qwen_move = get_qwen_move(st.session_state.board, st.session_state.current_player)
            if qwen_move:
                row, col = qwen_move
                if 0 <= row < 15 and 0 <= col < 15 and st.session_state.board[row][col] == 0:
                    st.success(f"Qwen AI 选择位置: ({row}, {col})")
                    st.session_state.board[row][col] = 2
                    st.session_state.move_history.append((row, col))
                    
                    if check_winner(st.session_state.board, row, col, 2):
                        st.session_state.game_over = True
                        st.session_state.winner = 2
                    else:
                        st.session_state.current_player = 1
                    
                    st.rerun()
                else:
                    st.error(f"Qwen AI 返回了无效位置: ({row}, {col})，回退到简单 AI")
                    # 回退到简单 AI
                    row, col = ai_move_simple(st.session_state.board, st.session_state.current_player)
                    st.success(f"AI 选择位置: ({row}, {col})")
                    st.session_state.board[row][col] = 2
                    st.session_state.move_history.append((row, col))
                    
                    if check_winner(st.session_state.board, row, col, 2):
                        st.session_state.game_over = True
                        st.session_state.winner = 2
                    else:
                        st.session_state.current_player = 1
                    
                    st.rerun()
            else:
                st.warning("Qwen AI 调用失败，回退到简单 AI")
                # 回退到简单 AI
                row, col = ai_move_simple(st.session_state.board, st.session_state.current_player)
                st.success(f"AI 选择位置: ({row}, {col})")
                st.session_state.board[row][col] = 2
                st.session_state.move_history.append((row, col))
                
                if check_winner(st.session_state.board, row, col, 2):
                    st.session_state.game_over = True
                    st.session_state.winner = 2
                else:
                    st.session_state.current_player = 1
                
                st.rerun()
        elif use_custom_ai and ai_code:
            result = execute_user_code(
                ai_code,
                st.session_state.board,
                st.session_state.current_player
            )
            
            if isinstance(result, tuple) and len(result) == 2:
                move, output = result
                if move:
                    row, col = move
                    if 0 <= row < 15 and 0 <= col < 15 and st.session_state.board[row][col] == 0:
                        st.success(f"AI 选择位置: ({row}, {col})")
                        st.session_state.board[row][col] = 2
                        st.session_state.move_history.append((row, col))
                        
                        if check_winner(st.session_state.board, row, col, 2):
                            st.session_state.game_over = True
                            st.session_state.winner = 2
                        else:
                            st.session_state.current_player = 1
                        
                        if output:
                            st.code(output, language="text")
                        st.rerun()
                    else:
                        st.error(f"无效的位置: ({row}, {col})")
                        if output:
                            st.code(output, language="text")
                else:
                    st.error("代码未返回有效的下一步位置")
                    if output:
                        st.code(output, language="text")
        else:
            # 使用简单AI
            row, col = ai_move_simple(st.session_state.board, st.session_state.current_player)
            st.success(f"AI 选择位置: ({row}, {col})")
            st.session_state.board[row][col] = 2
            st.session_state.move_history.append((row, col))
            
            if check_winner(st.session_state.board, row, col, 2):
                st.session_state.game_over = True
                st.session_state.winner = 2
            else:
                st.session_state.current_player = 1
            
            st.rerun()
    
    # 显示历史记录
    if st.session_state.move_history:
        st.subheader("历史记录")
        with st.container(height=300):
            for i, (r, c) in enumerate(st.session_state.move_history):
                player = "黑子" if i % 2 == 0 else "白子"
                st.text(f"第 {i+1} 步: {player} -> ({r}, {c})")
