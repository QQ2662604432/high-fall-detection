#!/bin/bash
#
# 高空抛物检测系统 - Web UI 启动脚本（Linux/macOS）
# Usage: ./start_web.sh [--host HOST] [--port PORT]
#

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ========= 默认值 =========
HOST="0.0.0.0"
PORT=5000
DEBUG=""

# ========= 解析命令行参数 =========
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "未知参数：$1"
            echo "用法：./start_web.sh [--host HOST] [--port PORT] [--debug]"
            exit 1
            ;;
    esac
done

# ========= 检查虚拟环境 =========
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  警告：未找到虚拟环境（venv/）${NC}"
    echo "请先运行："
    echo "  ./setup.sh"
    echo ""
    read -p "是否继续使用当前 Python 环境？(y/N) " choice
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        exit 1
    fi
else
    echo -e "${GREEN}>>> 激活虚拟环境...${NC}"
    source venv/bin/activate
fi

# ========= 检查 Flask =========
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Flask 未安装，正在自动安装...${NC}"
    pip install flask
fi

# ========= 启动 Flask 应用 =========
echo ""
echo "=================================================="
echo -e "${GREEN}🚀 高空抛物检测系统 - Web UI 启动中...${NC}"
echo "=================================================="
echo ""

# 构建访问地址
echo "🌐 本地访问地址："
echo "   http://localhost:$PORT"
echo "   http://127.0.0.1:$PORT"
echo ""

if [ "$HOST" = "0.0.0.0" ]; then
    echo "📡 局域网访问地址（示例）："
    # 尝试获取局域网 IP
    if command -v ifconfig &> /dev/null; then
        LOCAL_IP=$(ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
    elif command -v ip &> /dev/null; then
        LOCAL_IP=$(ip addr show 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1)
    fi

    if [ -n "$LOCAL_IP" ]; then
        echo "   http://$LOCAL_IP:$PORT"
    else
        echo "   （无法自动获取 IP，请手动查看 ifconfig/ip addr）"
    fi
    echo ""
fi

echo "按 Ctrl+C 停止服务"
echo "=================================================="
echo ""

# 启动 Flask
export FLASK_APP=src/web_app.py
if [ -n "$DEBUG" ]; then
    echo "🔧 调试模式已开启"
    echo ""
fi

python -m src.web_app_launcher --host "$HOST" --port "$PORT" $DEBUG
