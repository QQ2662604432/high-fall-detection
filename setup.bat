@echo off
chcp 65001 >nul
echo ================================
echo   高空抛物检测系统 - 环境安装
echo ================================
echo.

rem ========== 步骤1：检查 Python ==========
echo ^>>> 检查 Python 版本...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python！
    echo 请先安装 Python 3.8 或更高版本。
    echo 访问：https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version 2>&1 | findstr /r "^Python 3" >nul
if errorlevel 1 (
    echo ❌ 错误：Python 版本过低！
    echo 需要 Python 3.8 或更高版本。
    pause
    exit /b 1
)

echo ✅ Python 已安装
echo.

rem ========== 步骤2：创建虚拟环境 ==========
echo ^>>> 创建虚拟环境...
if exist venv (
    echo 虚拟环境已存在：venv\
    echo 如果要重建，请先删除 venv\ 目录
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 虚拟环境创建失败！
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功：venv\
)
echo.

rem ========== 步骤3：激活虚拟环境 ==========
echo ^>>> 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败！
    pause
    exit /b 1
)
echo ✅ 虚拟环境已激活
echo.

rem ========== 步骤4：升级 pip ==========
echo ^>>> 升级 pip...
python -m pip install --upgrade pip >nul 2>&1
echo ✅ pip 升级完成
echo.

rem ========== 步骤5：安装依赖包 ==========
echo ^>>> 安装依赖包（可能需要几分钟)...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖包安装失败！
        pause
        exit /b 1
    )
    echo ✅ 依赖包安装完成
) else (
    echo ⚠️  警告：未找到 requirements.txt
)
echo.

rem ========== 步骤6：安装 Web UI 依赖 ==========
echo ^>>> 安装 Web UI 依赖...
pip install flask >nul 2>&1
echo ✅ Web UI 依赖安装完成
echo.

rem ========== 完成 ==========
echo ================================
echo ✅ 安装完成！
echo ================================
echo.
echo 📋 接下来的步骤：
echo.
echo   1. 激活虚拟环境（如果未激活）：
echo      venv\Scripts\activate.bat
echo.
echo   2. 启动 Web UI：
echo      start_web.bat
echo.
echo   3. 在浏览器中访问：
echo      http://localhost:5000
echo.
pause
