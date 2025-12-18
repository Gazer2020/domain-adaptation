import os
import subprocess
import time
import sys

# ================= 配置区域 =================
# 源代码入口相对于本脚本的路径
MAIN_SCRIPT = os.path.abspath("../src/main.py")

# 指定本次批量运行使用的数据集配置文件名和方法配置文件名 (无需 .yaml 后缀)
DATASET_NAME = "office31" 
METHOD_NAME = "dann"

# 任务列表：(Source, Target)
TASKS = [
    ('amazon', 'webcam'),
    ('amazon', 'dslr'),
    ('webcam', 'amazon'),
    ('webcam', 'dslr'),
    ('dslr', 'amazon'),
    ('dslr', 'webcam')
]

# 待遍历的超参数列表（例如损失权重 lambda）
LAMBDAS = [0.1, 0.5, 1.0]

# 显卡设置 (如果想指定某块卡，例如 "0")
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

    print(f"统计：共有 {len(TASKS) * len(LAMBDAS)} 个实验待跑。")

    for source, target in TASKS:
        for lmbd in LAMBDAS:
            task_id = f"{source}2{target}_lambda_{lmbd}"
            exp_name = f"{METHOD_NAME}_{task_id}_{int(time.time())}"
            
            print("\n" + "="*50)
            print(f"正在启动任务: {task_id}")
            print(f"实验名称: {exp_name}")
            print("="*50 + "\n")

            # 构建命令行指令 (符合 OmegaConf 覆盖语法)
            cmd = [
                sys.executable, MAIN_SCRIPT,
                f"dataset={DATASET_NAME}",
                f"method={METHOD_NAME}",
                f"dataset.source={source}",
                f"dataset.target={target}",
                f"method.params.trade_off={lmbd}",
                f"exp_name={exp_name}"
            ]

            try:
                # subprocess.run 默认是阻塞的（串行执行）
                result = subprocess.run(cmd, env=env, check=True)
                
                if result.returncode == 0:
                    success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"!!! 任务失败: {task_id}, 错误代码: {e.returncode}")
                failed_tasks.append(task_id)
            except KeyboardInterrupt:
                print("\n用户中断运行。")
                sys.exit(1)

    # 最终报告
    end_time = time.time()
    total_duration = (end_time - start_time) / 3600
    
    print("\n\n" + "#"*50)
    print("所有批量实验已完成！")
    print(f"总耗时: {total_duration:.2f} 小时")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_tasks)}")
    if failed_tasks:
        print(f"失败任务列表: {failed_tasks}")
    print("#"*50)

if __name__ == "__main__":
    # 简单的路径检查
    if not os.path.exists(MAIN_SCRIPT):
        print(f"错误: 找不到入口文件 {MAIN_SCRIPT}")
        sys.exit(1)
        
    run_experiments()