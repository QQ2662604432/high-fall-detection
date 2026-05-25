#!/usr/bin/env python3
"""
高空抛物检测系统 - 主程序入口

Usage:
    python main.py [--config config/config.yaml] [--display]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

# 导入工具函数
from src.utils import setup_logger, ensure_dir

# 项目根目录
PROJECT_ROOT = Path(__file__).parent


def load_config(config_path: Path) -> dict:
    """
    加载配置文件。
    
    Args:
        config_path: 配置文件路径（Path对象）
        
    Returns:
        dict: 配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: 配置文件格式错误
        
    Example:
        >>> config = load_config(Path("config/config.yaml"))
        >>> print(config["video_sources"])
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


def create_output_dirs(config: dict) -> None:
    """
    创建输出目录结构。
    
    Args:
        config: 配置字典
        
    Example:
        >>> create_output_dirs(config)
    """
    # 主输出目录
    output_dir = Path(config.get("alarm", {}).get("output_dir", "output"))
    ensure_dir(output_dir)
    
    # 子目录
    ensure_dir(output_dir / "snapshots")
    ensure_dir(output_dir / "clips")
    
    # 日志目录
    log_config = config.get("logging", {})
    if log_config.get("save_to_file", False):
        log_dir = Path(log_config.get("log_dir", "output/logs"))
        ensure_dir(log_dir)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
        
    Example:
        $ python main.py --config config/custom.yaml --display
    """
    parser = argparse.ArgumentParser(
        description="高空抛物检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                              # 使用默认配置
  python main.py --config config/custom.yaml # 使用自定义配置
  python main.py --display                    # 显示实时画面
  python main.py --config config/config.yaml --display
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)"
    )
    
    parser.add_argument(
        "--display",
        action="store_true",
        help="是否显示实时画面"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    主函数。
    
    执行流程:
        1. 解析命令行参数
        2. 加载配置文件
        3. 初始化日志系统
        4. 创建输出目录
        5. 初始化各功能模块（TODO）
        6. 进入主循环（TODO）
        7. 清理资源
    """
    # Step 1: 解析命令行参数
    args = parse_arguments()
    
    # 如果命令行指定了--display，覆盖配置文件
    config_path = PROJECT_ROOT / args.config
    
    # Step 2: 加载配置
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"配置文件格式错误: {e}")
        sys.exit(1)
    
    # 如果命令行指定了--display，覆盖配置
    if args.display:
        config["display"] = True
    
    # Step 3: 初始化日志
    logger = setup_logger("HighFall", config.get("logging", {}))
    
    logger.info("=" * 60)
    logger.info("高空抛物检测系统 启动")
    logger.info("=" * 60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"实时显示: {config.get('display', True)}")
    
    # Step 4: 创建输出目录
    create_output_dirs(config)
    logger.info("输出目录已创建")
    
    # Step 5: TODO - 初始化各模块（后续任务实现）
    logger.info("初始化功能模块...")
    
    # QQ机器人模块
    # qq_bot = QQBot(config)
    
    # 告警处理模块
    # alarm_handler = AlarmHandler(config, qq_bot)
    
    # 区域管理模块
    # region_manager = RegionManager(config.get("regions", {}))
    
    # 背景建模模块
    # bg_model = BackgroundModel(config)
    
    # 检测模块
    # detector = Detector(config, region_manager)
    
    # 跟踪模块
    # tracker = Tracker(config)
    
    # 轨迹分析模块
    # trajectory_analyzer = TrajectoryAnalyzer(config)
    
    # Step 6: TODO - 初始化视频源（后续任务实现）
    # sources = config["video_sources"]
    # readers = [VideoReader(src["url"], config) for src in sources if src["enabled"]]
    
    # 检查视频源配置
    video_sources = config.get("video_sources", [])
    enabled_sources = [src for src in video_sources if src.get("enabled", True)]
    logger.info(f"已配置 {len(enabled_sources)} 个视频源")
    
    # Step 7: 主循环占位
    logger.info("进入主循环...")
    try:
        # TODO: 实现完整管道
        # for frame in video_reader:
        #     # 1. 背景差分
        #     fg_mask = bg_model.apply(frame)
        #     
        #     # 2. 检测前景物体
        #     detections = detector.detect(fg_mask, frame)
        #     
        #     # 3. 跟踪物体
        #     tracks = tracker.update(detections)
        #     
        #     # 4. 轨迹分析
        #     for track in tracks:
        #         if trajectory_analyzer.is_falling(track):
        #             alarm_handler.trigger_alarm(track, frame)
        #     
        #     # 5. 显示（可选）
        #     if config.get("display", True):
        #         cv2.imshow("HighFall Detection", frame)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        
        logger.warning("主循环尚未实现，这是程序骨架")
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
    finally:
        # TODO: 清理资源
        # for reader in readers:
        #     reader.release()
        # cv2.destroyAllWindows()
        
        logger.info("=" * 60)
        logger.info("高空抛物检测系统 已退出")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
