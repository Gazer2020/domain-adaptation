import torch
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger


def parse_range(range_str):
    """
    将 '0-30' 转换为 [0, 1, ..., 30]
    将 '1,3,5-7' 转换为 [1, 3, 5, 6, 7]
    """
    result = []
    # 按照逗号分割
    parts = str(range_str).split(",")
    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def get_config():
    if not OmegaConf.has_resolver("parse_range"):
        OmegaConf.register_new_resolver("range", parse_range)

    # 1. 加载基础/默认配置 (包含通用设置如 seed, use_gpu 等)
    base_cfg = OmegaConf.create(
        obj={"exp_name": "default", "mode": "csda", "device": "cuda", "seed": 42}
    )

    # 2. 通过 CLI 获取数据集和方法的配置文件路径
    cli_cfg = OmegaConf.from_cli()

    # Determine config files
    dataset_name = cli_cfg.dataset_config
    method_name = cli_cfg.method_config

    if dataset_name is None:
        raise ValueError("Must specify dataset config file via 'dataset=name'")
    if method_name is None:
        raise ValueError("Must specify method config file via 'method=name'")

    # 3. 加载具体的 YAML 文件
    # 使用 pathlib 确保路径稳健
    # 假设项目根目录包含 src/
    # __file__ 是 src/utils/config.py
    # src_dir 是 src/
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent.parent
    # config dir relative to src
    config_dir = src_dir / "configs"

    dataset_path = config_dir / "datasets" / f"{dataset_name}.yaml"
    method_path = config_dir / "methods" / f"{method_name}.yaml"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {dataset_path}")
    if not method_path.exists():
        raise FileNotFoundError(f"Method config not found at: {method_path}")

    data_cfg = OmegaConf.load(dataset_path)
    meth_cfg = OmegaConf.load(method_path)

    # 4. 层级化合并 (右侧优先级高)
    # 合并顺序：基础 < 数据集 < 方法 < 命令行参数
    config = OmegaConf.merge(base_cfg, data_cfg, meth_cfg, cli_cfg)

    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA selected but not available. Switching to CPU.")
        config.device = "cpu"

    # 5. 冻结配置（防止程序运行中意外修改参数）
    OmegaConf.set_readonly(config, True)

    return config
