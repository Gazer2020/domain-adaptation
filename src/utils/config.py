from omegaconf import OmegaConf


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


OmegaConf.register_new_resolver("range", parse_range, replace=True)
