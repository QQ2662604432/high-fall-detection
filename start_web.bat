@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem ========= 默认值 =========
set HOST=0.0.0.0
set PORT=5000
set DEBUG=

rem ========= 解析命令行参数 =========
:parse_args
if "%~1"=="" goto :done_args
set ARG=%~1
if /i "%ARG%"=="--host" (
    shift
    set HOST=%~1
    shift
    goto :parse_args
)
if /i "%ARG%"=="--port" (
    shift
    set PORT=%~1
    shift
    goto :parse_args
)
if /i "%ARG%"=="--debug" (
    set DEBUG=--debug
    shift
    goto :parse_args
)
echo 未知参数：%ARG%
echo 用法：start_web.bat [--host HOST] [--port PORT] [--debug]
endlocal
exit /b 1

:done_args

rem ========= 检查虚拟环境 =========
if not exist venv (
    echo ⚠️  警告：未找到虚拟环境（venv\）
    echo 请先运行：
    echo   setup.bat
    pause
    endlocal
    exit /b 1
)

rem ========= 激活虚拟环境 =========
echo ^>>> 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败！
    pause
    endlocal
    exit /b 1
)
echo ✅ 虚拟环境已激活
echo.

rem ========= 检查 Flask =========
python -c "import flask" 2>nul
if errorlevel 1 (
    echo ⚠️  Flask 未安装，正在自动安装...
    pip install flask
    if errorlevel 1 (
        echo ❌ Flask 安装失败！
        pause
        endlocal
        exit /b 1
    )
    echo ✅ Flask 安装成功
)

rem ========= 启动 Flask 应用 =========
echo.
echo ================================
echo 🚀 高空抛物检测系统 - Web UI 启动中...
echo ================================
echo.
echo 🌐 本地访问地址：
echo    http://localhost:%PORT%
echo    http://127.0.0.1:%PORT%
echo.
echo 📡 局域网访问地址（示例）：
echo    http://%COMPUTERNAME%:%PORT%
echo.
echo 按 Ctrl+C 停止服务
echo ================================
echo.

rem 设置 FLASK_APP 环境变量
set FLASK_APP=src/web_app.py

rem 启动 Flask
python -m src.web_app_launcher --host %HOST% --port %PORT% %DEBUG%

endlocal
