"""
QQ机器人推送模块

实现OneBot协议HTTP API调用，用于推送告警消息到QQ群。
使用轻量级HTTP POST调用，不依赖nonebot2完整框架。

OneBot API说明：
- 需要本地运行 OneBot 实现（如 go-cqhttp / Lagrange.Core）
- API地址默认为 http://localhost:8080
- 支持 HTTP POST 调用，无需完整Bot框架

Author: 寇豆码 (Alex)
Date: 2024
"""

import logging
import requests
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime

from src.utils import setup_logger
from src.alarm_handler import AlarmEvent


class QQBot:
    """
    QQ机器人告警推送。使用OneBot协议 HTTP API方式。
    
    功能：
    1. OneBot HTTP API 封装（轻量级，不依赖nonebot2完整框架）
    2. 告警消息格式化
    3. 支持发送文字消息和图片（CQ码格式）
    4. 提供 send_alarm(alarm) 接口
    
    OneBot API调用示例：
    - 发送群消息: POST http://localhost:8080/send_group_msg
    - 发送私聊消息: POST http://localhost:8080/send_private_msg
    
    CQ码格式（发送图片）：
    - [CQ:image,file=file:///path/to/image.jpg]
    
    Attributes:
        enabled: 是否启用QQ推送
        group_id: 目标群号
        api_url: OneBot API地址
        access_token: 访问令牌（可选）
        logger: 日志器实例
        
    Example:
        >>> config = {
        ...     "qq_bot": {
        ...         "enabled": True,
        ...         "group_id": "123456789",
        ...         "api_url": "http://localhost:8080",
        ...         "access_token": "your_token"
        ...     }
        ... }
        >>> qq_bot = QQBot(config)
        >>> qq_bot.send_alarm(alarm)
    """
    
    def __init__(self, config: dict) -> None:
        """
        初始化QQ机器人。
        
        Args:
            config: 配置字典，应包含 "qq_bot" 节
            
        Example:
            >>> config = {
            ...     "qq_bot": {
            ...         "enabled": True,
            ...         "group_id": "123456789",
            ...         "api_url": "http://localhost:8080",
            ...         "access_token": ""
            ...     }
            ... }
            >>> qq_bot = QQBot(config)
        """
        qq_config = config.get("qq_bot", {})
        self.enabled = qq_config.get("enabled", False)
        self.group_id = qq_config.get("group_id", "")
        self.api_url = qq_config.get("api_url", "http://localhost:8080")
        self.access_token = qq_config.get("access_token", "")
        
        # 初始化日志器
        if self.enabled:
            self.logger = setup_logger(__name__, config.get("logging", {}))
            self.logger.info(
                f"QQBot initialized: "
                f"api_url={self.api_url}, "
                f"group_id={self.group_id}, "
                f"enabled={self.enabled}"
            )
        else:
            # 创建一个null logger当未启用时
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
    
    def send_alarm(self, alarm: AlarmEvent) -> bool:
        """
        发送告警消息到QQ群。
        
        发送流程：
        1. 格式化告警消息
        2. 发送文字消息到群
        3. 发送快照图片（如果文件存在）
        
        Args:
            alarm: AlarmEvent对象
            
        Returns:
            bool: 是否发送成功
            
        Example:
            >>> success = qq_bot.send_alarm(alarm)
            >>> if success:
            ...     print("Alarm sent successfully")
        """
        if not self.enabled:
            self.logger.debug("QQ推送未启用，跳过")
            return False
        
        try:
            # 格式化消息
            message = self._format_message(alarm)
            
            # 发送文字消息
            resp = self._send_group_message(self.group_id, message)
            if resp.get("status") == "ok":
                self.logger.debug(f"Text message sent: {resp}")
            else:
                self.logger.warning(f"Text message send failed: {resp}")
            
            # 发送快照图片
            if alarm.snapshot_path and Path(alarm.snapshot_path).exists():
                img_resp = self._send_group_image(self.group_id, alarm.snapshot_path)
                if img_resp.get("status") == "ok":
                    self.logger.debug(f"Image sent: {img_resp}")
                else:
                    self.logger.warning(f"Image send failed: {img_resp}")
            else:
                self.logger.warning(f"Snapshot not found: {alarm.snapshot_path}")
            
            self.logger.info(f"告警已推送: {alarm.alarm_id}")
            return True
            
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"QQ推送失败（连接错误）: {e}")
            return False
        except requests.exceptions.Timeout as e:
            self.logger.error(f"QQ推送失败（超时）: {e}")
            return False
        except Exception as e:
            self.logger.error(f"QQ推送失败: {e}", exc_info=True)
            return False
    
    def _format_message(self, alarm: AlarmEvent) -> str:
        """
        格式化告警消息。
        
        Args:
            alarm: AlarmEvent对象
            
        Returns:
            str: 格式化的告警消息
            
        Example:
            >>> message = qq_bot._format_message(alarm)
        """
        return (
            f"🚨 高空抛物告警 🚨\n"
            f"告警ID: {alarm.alarm_id}\n"
            f"时间: {alarm.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"置信度: {alarm.confidence:.1%}\n"
            f"轨迹ID: {alarm.track_id}\n"
            f"请及时处理！"
        )
    
    def _send_group_message(self, group_id: str, message: str) -> Dict:
        """
        通过OneBot HTTP API发送群消息。
        
        API端点: POST /send_group_msg
        
        Args:
            group_id: 群号
            message: 消息内容
            
        Returns:
            Dict: API响应（JSON格式）
            
        Raises:
            requests.exceptions.ConnectionError: 连接失败
            requests.exceptions.Timeout: 请求超时
            
        Example:
            >>> resp = qq_bot._send_group_message("123456789", "Hello")
            >>> print(resp)
        """
        url = f"{self.api_url}/send_group_msg"
        
        # 构建请求payload
        payload = {
            "group_id": int(group_id),
            "message": message
        }
        
        # 构建请求头
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        # 发送POST请求
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        return resp.json()
    
    def _send_group_image(self, group_id: str, image_path: str) -> Dict:
        """
        发送群图片（OneBot CQ码格式）。
        
        API端点: POST /send_group_msg
        CQ码格式: [CQ:image,file=file:///path/to/image.jpg]
        
        注意：需要确保QQ机器人有权限读取该文件路径。
        
        Args:
            group_id: 群号
            image_path: 图片文件路径
            
        Returns:
            Dict: API响应（JSON格式）
            
        Raises:
            requests.exceptions.ConnectionError: 连接失败
            requests.exceptions.Timeout: 请求超时
            
        Example:
            >>> resp = qq_bot._send_group_image("123456789", "/path/to/image.jpg")
        """
        url = f"{self.api_url}/send_group_msg"
        
        # OneBot CQ码格式发送图片
        # 注意：file:// 协议需要OneBot实现支持
        # 某些实现可能需要使用 base64 或 http 协议
        abs_path = Path(image_path).resolve()
        message = f"[CQ:image,file=file://{abs_path}]"
        
        # 构建请求payload
        payload = {
            "group_id": int(group_id),
            "message": message
        }
        
        # 构建请求头
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        # 发送POST请求
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        return resp.json()
    
    def send_text(self, group_id: str, message: str) -> bool:
        """
        发送纯文本消息到群（通用接口）。
        
        Args:
            group_id: 群号
            message: 消息内容
            
        Returns:
            bool: 是否发送成功
            
        Example:
            >>> qq_bot.send_text("123456789", "测试消息")
        """
        if not self.enabled:
            self.logger.debug("QQ推送未启用，跳过")
            return False
        
        try:
            resp = self._send_group_message(group_id, message)
            return resp.get("status") == "ok"
        except Exception as e:
            self.logger.error(f"发送文本消息失败: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        测试与OneBot API的连接。
        
        Returns:
            bool: 连接是否正常
            
        Example:
            >>> if qq_bot.test_connection():
            ...     print("Connection OK")
        """
        if not self.enabled:
            self.logger.debug("QQ推送未启用，跳过连接测试")
            return False
        
        try:
            # 尝试调用一个简单的API（获取登录号信息）
            url = f"{self.api_url}/get_login_info"
            headers = {}
            if self.access_token:
                headers["Authorization"] = f"Bearer {self.access_token}"
            
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.json()
            
            if data.get("status") == "ok":
                self.logger.info(f"OneBot连接正常: {data.get('data', {})}")
                return True
            else:
                self.logger.warning(f"OneBot API返回异常: {data}")
                return False
                
        except Exception as e:
            self.logger.error(f"OneBot连接测试失败: {e}")
            return False
