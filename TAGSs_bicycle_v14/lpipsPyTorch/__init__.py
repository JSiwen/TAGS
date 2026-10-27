import torch

from .modules.lpips import LPIPS

_LPIPS_CACHE = {}


def get_lpips_criterion(net_type: str = "alex", version: str = "0.1", device=None):
    """Return a cached LPIPS criterion (eval mode, frozen weights)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        device_key = str(device)
    else:
        device_key = str(device)
        device = torch.device(device)
    key = (net_type, version, device_key)
    if key not in _LPIPS_CACHE:
        criterion = LPIPS(net_type, version).to(device)
        criterion.eval()
        for param in criterion.parameters():
            param.requires_grad = False
        _LPIPS_CACHE[key] = criterion
    return _LPIPS_CACHE[key]


def lpips(
    x: torch.Tensor,
    y: torch.Tensor,
    net_type: str = "alex",
    version: str = "0.1",
):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    criterion = get_lpips_criterion(net_type, version, x.device)
    return criterion(x, y)
