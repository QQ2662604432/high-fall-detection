"""
Web UI 启动脚本 - 自动补全依赖并启动 Flask 应用
Usage: python -m src.web_app [--host HOST] [--port PORT] [--debug]
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def check_and_install_dependencies():
    """
    检查并自动安装依赖包。

    如果当前在虚拟环境中，直接安装 requirements.txt。
    如果不在虚拟环境中，提示用户激活或创建虚拟环境。

    Returns:
        bool: 依赖是否安装成功
    """
    print(">>> 检查依赖包...")

    # 检查是否在虚拟环境中
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("⚠️  警告：当前不在虚拟环境中！")
        print("建议在虚拟环境中运行：")
        print(f"  Linux/Mac: source venv/bin/activate")
        print(f"  Windows: venv\\Scripts\\activate.bat")
        print("")

        # 询问用户是否继续
        if sys.stdin.isatty():
            choice = input("是否继续在全局环境中安装？(y/N): ").strip().lower()
            if choice != 'y':
                print("")
                print("请先创建并激活虚拟环境：")
                print("  python -m venv venv")
                print("  (然后重新运行此脚本)")
                return False
        else:
            # 非交互式环境，默认不继续
            print("非交互式环境，跳过自动安装。")
            print("请确保以下依赖已安装：")
            print("  flask")
            return True

    # 检查 Flask 是否已安装
    try:
        import flask
        print(f"✅ Flask 已安装：v{flask.__version__}")
    except ImportError:
        print("⚠️  Flask 未安装，正在自动安装...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'flask'
            ])
            print("✅ Flask 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ Flask 安装失败：{e}")
            return False

    # 安装 requirements.txt 中的依赖
    req_file = PROJECT_ROOT / 'requirements.txt'
    if req_file.exists():
        print(f">>> 安装 requirements.txt 中的依赖...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
            ])
            print("✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败：{e}")
            return False

    return True


def start_flask_app(host: str, port: int, debug: bool):
    """
    启动 Flask Web 应用。

    Args:
        host: 监听地址（0.0.0.0 允许局域网访问）
        port: 监听端口
        debug: 是否开启调试模式
    """
    # 将项目根目录加入 sys.path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 导入并运行 Flask 应用
    try:
        from src.web_app import app

        print("")
        print("=" * 60)
        print(f"🚀 高空抛物检测系统 - Web UI 启动中...")
        print("=" * 60)
        print(f"")
        print(f"🌐 本地访问地址：")
        print(f"   http://localhost:{port}")
        print(f"   http://127.0.0.1:{port}")
        print(f"")
        if host == '0.0.0.0':
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                print(f"📡 局域网访问地址：")
                print(f"   http://{local_ip}:{port}")
            except Exception:
                pass
            print(f"")
        print(f"按 Ctrl+C 停止服务")
        print(f"=" * 60)
        print(f"")

        app.run(host=host, port=port, debug=debug)

    except ImportError as e:
        print(f"❌ 导入 Flask 应用失败：{e}")
        print(f"请确保 src/web_app.py 文件存在且无误。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败：{e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高空抛物检测系统 - Web UI 启动脚本")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="监听地址（默认 0.0.0.0，允许局域网访问）")
    parser.add_argument("--port", type=int, default=5000,
                        help="监听端口（默认 5000）")
    parser.add_argument("--debug", action="store_true",
                        help="开启调试模式")
    parser.add_argument("--no-auto-install", action="store_true",
                        help="跳过自动安装依赖")
    args = parser.parse_args()

    # 自动安装依赖
    if not args.no_auto_install:
        if not check_and_install_dependencies():
            sys.exit(1)

    # 启动 Flask 应用
    start_flask_app(args.host, args.port, args.debug)


if __name__ == "__main__":
    main()
