#!/usr/bin/env python3
"""
本地开发环境启动脚本
解决Python模块导入路径问题
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    
    # 添加项目根目录到Python路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    print(f"🚀 启动AI量化系统后端服务...")
    print(f"📁 项目根目录: {project_root}")
    print(f"🐍 Python路径: {env.get('PYTHONPATH', 'Not set')}")
    
    # 启动uvicorn
    cmd = [
        "python", "-m", "uvicorn",
        "backend.app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info"
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, env=env, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
