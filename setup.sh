#!/bin/bash
#
# 高空抛物检测系统 - 自动安装脚本（Linux/macOS）
# Usage: ./setup.sh
#

set -e  # 遇到错误立即退出

echo "=================================================="
echo "  高空抛物检测系统 - 环境安装"
echo "=================================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========== 步骤1：检查 Python 版本 ==========
echo -e "${YELLOW}>>> 检查 Python 版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 错误：未找到 Python 3！${NC}"
    echo "请先安装 Python 3.8 或更高版本。"
    echo "访问：https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ 错误：Python 版本过低（$PYTHON_VERSION）${NC}"
    echo "需要 Python 3.8 或更高版本。"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION 已安装${NC}"
echo ""

# ========== 步骤2：创建虚拟环境 ==========
echo -e "${YELLOW}>>> 创建虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo "虚拟环境已存在：venv/"
    echo "如果要重建，请先删除 venv/ 目录"
else
    python3 -m venv venv
    echo -e "${GREEN}✅ 虚拟环境创建成功：venv/${NC}"
fi
echo ""

# ========== 步骤3：激活虚拟环境 ==========
echo -e "${YELLOW}>>> 激活虚拟环境...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ 虚拟环境已激活${NC}"
echo ""

# ========== 步骤4：升级 pip ==========
echo -e "${YELLOW}>>> 升级 pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✅ pip 升级完成${NC}"
echo ""

# ========== 步骤5：安装依赖包 ==========
echo -e "${YELLOW}>>> 安装依赖包（可能需要几分钟）...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ 依赖包安装完成${NC}"
else
    echo -e "${RED}⚠️  警告：未找到 requirements.txt${NC}"
fi
echo ""

# ========== 步骤6：安装 Web UI 依赖 ==========
echo -e "${YELLOW}>>> 检查并安装 Web UI 依赖...${NC}"
pip install flask > /dev/null 2>&1
echo -e "${GREEN}✅ Web UI 依赖安装完成${NC}"
echo ""

# ========== 步骤7：设置启动脚本权限 ==========
echo -e "${YELLOW}>>> 设置启动脚本权限...${NC}"
chmod +x start_web.sh 2>/dev/null || true
echo -e "${GREEN}✅ 权限设置完成${NC}"
echo ""

# ========== 完成 ==========
echo "=================================================="
echo -e "${GREEN}✅ 安装完成！${NC}"
echo "=================================================="
echo ""
echo "📋 接下来的步骤："
echo ""
echo "  1. 激活虚拟环境："
echo "     source venv/bin/activate"
echo ""
echo "  2. 启动 Web UI："
echo "     ./start_web.sh"
echo ""
echo "  3. 在浏览器中访问："
echo "     http://localhost:5000"
echo ""
echo "=================================================="
