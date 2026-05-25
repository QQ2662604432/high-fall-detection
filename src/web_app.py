"""高空抛物检测系统 - Flask Web UI 应用

提供 Web 管理界面，支持：
1. 实时视频流（MJPEG 推流）
2. 检测控制（启动/停止）
3. 告警查看（列表、快照、视频）
4. 配置修改（读取/保存 config.yaml）

Usage: python -m src.web_app_launcher [--host HOST] [--port PORT]
"""

import os
import sys
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, Response, render_template, jsonify, request, send_file, abort

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 全局状态
current_frame = None
detection_running = False
detection_thread = None
alarms: List[Dict] = []  # 告警列表（内存存储，生产环境建议用数据库）
stats = {
    'fps': 0.0,
    'detections': 0,
    'tracks': 0,
    'start_time': None,
    'total_frames': 0,
}

# 创建 Flask 应用
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / 'src' / 'templates'),
    static_folder=str(PROJECT_ROOT / 'src' / 'static'),
)
app.secret_key = 'high-fall-detection-2024'


def gen_frames():
    """视频帧生成器（MJPEG 流）"""
    global current_frame
    while True:
        if current_frame is None:
            time.sleep(0.03)
            continue
        ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """首页（实时视频 + 控制面板）"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG 视频流端点"""
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/start', methods=['POST'])
def api_start():
    """启动检测"""
    global detection_running, detection_thread

    if detection_running:
        return jsonify({'success': False, 'message': '检测已在运行中'})

    # 在新线程中启动检测
    def run_detection():
        global current_frame, stats, detection_running

        # 导入检测主函数
        try:
            from src.main import main as detection_main
            # 注意：这里需要修改 main.py 支持线程模式
            # 暂时用占位符
            print("检测线程启动（占位符）")
            stats['start_time'] = time.time()

            # 模拟视频流（实际应调用真正的检测逻辑）
            cap = cv2.VideoCapture(0)  # 使用默认摄像头
            while detection_running:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame = frame
                stats['total_frames'] += 1
                time.sleep(0.03)

            cap.release()
        except Exception as e:
            print(f"检测线程错误: {e}")

    detection_running = True
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()

    return jsonify({'success': True, 'message': '检测已启动'})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """停止检测"""
    global detection_running

    if not detection_running:
        return jsonify({'success': False, 'message': '检测未运行'})

    detection_running = False
    return jsonify({'success': True, 'message': '检测已停止'})


@app.route('/api/status')
def api_status():
    """获取系统状态"""
    uptime = 0
    if stats['start_time']:
        uptime = int(time.time() - stats['start_time'])

    return jsonify({
        'running': detection_running,
        'fps': stats['fps'],
        'detections': stats['detections'],
        'tracks': stats['tracks'],
        'uptime': uptime,
        'total_frames': stats['total_frames'],
    })


@app.route('/alarms')
def alarms():
    """告警列表页面"""
    return render_template('alarms.html')


@app.route('/api/alarms')
def api_alarms():
    """获取告警 JSON 数据"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    total = len(alarms)
    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        'alarms': alarms[start:end],
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
    })


@app.route('/api/alarms/<alarm_id>/snapshot')
def api_alarm_snapshot(alarm_id):
    """下载告警快照图片"""
    for alarm in alarms:
        if alarm['id'] == alarm_id:
            snapshot_path = PROJECT_ROOT / alarm['snapshot_path']
            if snapshot_path.exists():
                return send_file(str(snapshot_path), mimetype='image/jpeg')
            else:
                abort(404, description="快照文件不存在")

    abort(404, description="告警不存在")


@app.route('/api/alarms/<alarm_id>/clip')
def api_alarm_clip(alarm_id):
    """下载告警视频"""
    for alarm in alarms:
        if alarm['id'] == alarm_id:
            clip_path = PROJECT_ROOT / alarm['clip_path']
            if clip_path.exists():
                return send_file(str(clip_path), mimetype='video/mp4')
            else:
                abort(404, description="视频文件不存在")

    abort(404, description="告警不存在")


@app.route('/config')
def config_page():
    """配置页面"""
    return render_template('config.html')


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """读取/修改配置"""
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'

    if request.method == 'POST':
        # 保存配置
        try:
            import yaml
            new_config = request.json

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False)

            return jsonify({'success': True, 'message': '配置已保存'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})

    else:
        # 读取配置
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return jsonify(config)
        except Exception as e:
            return jsonify({'success': False, 'message': f'读取失败: {str(e)}'}), 500


def add_alarm(alarm_data: Dict):
    """添加告警到列表（供检测模块调用）"""
    global alarms
    alarm_data['id'] = len(alarms) + 1
    alarm_data['timestamp'] = datetime.now().isoformat()
    alarms.append(alarm_data)

    # 只保留最近 1000 条告警
    if len(alarms) > 1000:
        alarms = alarms[-1000:]


if __name__ == '__main__':
    # 命令行参数解析
    parser = __import__('argparse').ArgumentParser(description="高空抛物检测系统 - Web UI")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="监听地址")
    parser.add_argument('--port', type=int, default=5000, help="监听端口")
    parser.add_argument('--debug', action='store_true', help="开启调试模式")
    args = parser.parse_args()

    # 启动 Flask 应用
    print(f"🌐 高空抛物检测系统 - Web UI 启动中...")
    print(f"📍 本地访问: http://localhost:{args.port}")
    print(f"📡 局域网访问: http://0.0.0.0:{args.port}")
    print(f"按 Ctrl+C 停止服务")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
