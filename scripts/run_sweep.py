import sys
import logging
import subprocess
from pathlib import Path

from hydra import compose, initialize


# logging.basicConfig(level=logging.INFO, format='[%(asctime)s][SCHEDULER] %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(overrides: list):
    """
    通过 subprocess 调用 main.py，并传递 Hydra 格式的参数
    """
    script_path = Path(__file__).resolve().parent.parent / "src" / "main.py"
    cmd = [sys.executable, str(script_path)] + overrides

    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        # 使用 subprocess 运行，确保每个实验有独立的显存空间
        subprocess.run(cmd, check=True)
        logger.info("Experiment finished successfully.\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed with error {e}\n")
        return False
    return True


def main():
    # 1. 初始化 Hydra 配置 (获取实验矩阵)
    # config_path 是相对于当前脚本的路径，指向 src/configs
    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(config_name="config")

        # --- 实验参数定义区域 ---
        # 你可以根据需要修改这里的任务列表
        method = 'mic'
        dataset = 'office-31'
        tasks = [("amazon", "webcam"), ("webcam", "dslr"), ("dslr", "amazon")]
        # -----------------------

        # 2. 首先执行：冒烟测试 (Smoke Test)
        logger.info("STEP 1: Starting Smoke Test (Tiny Dataset)")

        smoke_success = run_experiment(
            [
                f"method={method}",
                "dataset=mini-office-31",  # 使用超小型数据集配置
                "method.epochs=1",  # 只跑一个 epoch
                "batch_size=2",  # 极小 batch
                "exp_name=SMOKE_TEST",
            ]
        )

        if not smoke_success:
            logger.critical("Smoke Test failed! Aborting all formal experiments.")
            sys.exit(1)

        # 3. 继续执行：多任务并行/串行扫参
        logger.info("STEP 2: Starting Formal Experiments")

        total_tasks = len(tasks)
        for idx, (source, target) in enumerate(tasks):
            logger.info(f"Progress: [{idx+1}/{total_tasks}]")

            # 组合本次实验的参数
            overrides = [
                f"method={method}",
                f"dataset={dataset}",
                f"dataset.source={source}",
                f"dataset.target={target}",
                f"exp_name=EXP_{method}_{dataset}_{source}2{target}",
            ]

            run_experiment(overrides)

        logger.info("All tasks completed.")


if __name__ == "__main__":
    main()
