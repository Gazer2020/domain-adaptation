import torchvision.models as models


# Backbone registry
_BACKBONE_REGISTRY = {
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "resnet101": lambda: models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
}


def get_backbone(name: str = "resnet18"):
    """
    Get a backbone network by name.
    
    Args:
        name: Backbone name (resnet18, resnet50, resnet101)
        
    Returns:
        The backbone model with pretrained weights
        
    Raises:
        KeyError: If backbone name is not found
    """
    if name not in _BACKBONE_REGISTRY:
        available = list(_BACKBONE_REGISTRY.keys())
        raise KeyError(f"Backbone '{name}' not found. Available: {available}")
    return _BACKBONE_REGISTRY[name]()


def register_backbone(name: str):
    """
    Decorator to register a custom backbone.
    
    Usage:
        @register_backbone("my_backbone")
        def get_my_backbone():
            return MyBackboneModel()
    """
    def decorator(func):
        _BACKBONE_REGISTRY[name] = func
        return func
    return decorator


# Legacy functions for backward compatibility
def get_resnet18():
    return get_backbone("resnet18")


def get_resnet50():
    return get_backbone("resnet50")


def get_resnet101():
    return get_backbone("resnet101")
