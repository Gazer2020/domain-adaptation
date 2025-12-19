import os
import subprocess
import time
import sys
from loguru import logger

# ================= 配置区域 =================
# 源代码入口相对于本脚本的路径
MAIN_SCRIPT = os.path.abspath("../src/main.py")

# 指定本次批量运行使用的数据集配置文件名和方法配置文件名 (无需 .yaml 后缀)
DATASET_NAME = "mini-office-31" 
METHOD_NAME = "ros"

# 任务列表：(Source, Target) - 只跑一个任务用于测试
TASKS = [
    ('amazon', 'webcam'),
]

# 待遍历的超参数列表 - 只跑一个
LAMBDAS = [0.1]

# 显卡设置
GPU_ID = "0" 
# ===========================================

def run_experiments():
    start_time = time.time()
    success_count = 0
    failed_tasks = []

    # 确保 PYTHONPATH 包含 src 目录，否则 main.py 无法 import 内部模块
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("../src")
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    logger.info(f"统计：共有 {len(TASKS) * len(LAMBDAS)} 个实验待跑。")

    for source, target in TASKS:
        for lmbd in LAMBDAS:
            task_id = f"{source}2{target}_lambda_{lmbd}"
            exp_name = f"TEST_{METHOD_NAME}_{task_id}_{int(time.time())}"
            
            logger.info("="*50)
            logger.info(f"正在启动任务: {task_id}")
            logger.info(f"实验名称: {exp_name}")
            logger.info("="*50)

            # 构建命令行指令 (符合 OmegaConf 覆盖语法)
            cmd = [
                sys.executable, MAIN_SCRIPT,
                f"dataset_config={DATASET_NAME}",
                f"method_config={METHOD_NAME}",
                f"dataset.source={source}",
                f"dataset.target={target}",
                f"dataset.setting=csda",
                f"method.params.trade_off={lmbd}",
                f"exp_name={exp_name}",
                f"batch_size=32"
            ]

            try:
                # subprocess.run 默认是阻塞的（串行执行）
                result = subprocess.run(cmd, env=env, check=True)
                
                if result.returncode == 0:
                    success_count += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"!!! 任务失败: {task_id}, 错误代码: {e.returncode}")
                failed_tasks.append(task_id)
            except KeyboardInterrupt:
                logger.warning("\n用户中断运行。")
                sys.exit(1)

    # 最终报告
    end_time = time.time()
    total_duration = (end_time - start_time) / 3600
    
    logger.info("#"*50)
    logger.info("Smoke Test 实验已完成！")
    logger.info(f"总耗时: {total_duration:.2f} 小时")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {len(failed_tasks)}")
    if failed_tasks:
        logger.info(f"失败任务列表: {failed_tasks}")
    logger.info("#"*50)

if __name__ == "__main__":
    # 简单的路径检查
    if not os.path.exists(MAIN_SCRIPT):
        logger.error(f"错误: 找不到入口文件 {MAIN_SCRIPT}")
        sys.exit(1)
        
    run_experiments()
