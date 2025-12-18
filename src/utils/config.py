import os
from omegaconf import OmegaConf

def get_config():
    # 1. 加载基础/默认配置 (包含通用设置如 seed, use_gpu 等)
    base_cfg = OmegaConf.create({
        "exp_name": "default",
        "mode": "vanilla",
        "device": "cuda",
        "seed": 42
    })

    # 2. 通过 CLI 获取数据集和方法的配置文件路径
    # 例如运行: python main.py dataset=office31 method=dann source=amazon target=webcam
    cli_cfg = OmegaConf.from_cli()
    
    # 确保传入了核心配置路径
    assert "dataset" in cli_cfg, "Must specify dataset config, e.g., dataset=office31"
    assert "method" in cli_cfg, "Must specify method config, e.g., method=dann"

    # 3. 加载具体的 YAML 文件
    dataset_path = f"configs/datasets/{cli_cfg.dataset}.yaml"
    method_path = f"configs/methods/{cli_cfg.method}.yaml"
    
    data_cfg = OmegaConf.load(dataset_path)
    meth_cfg = OmegaConf.load(method_path)

    # 4. 层级化合并 (右侧优先级高)
    # 合并顺序：基础 < 数据集 < 方法 < 命令行参数
    config = OmegaConf.merge(base_cfg, data_cfg, meth_cfg, cli_cfg)

    # 5. 冻结配置（防止程序运行中意外修改参数）
    OmegaConf.set_readonly(config, True)
    
    return config