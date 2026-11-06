"""Load project-generated full training checkpoints across PyTorch versions."""
import inspect


def load_training_checkpoint(torch_module, path, map_location=None):
    # Old torch.load forwards unknown keywords to pickle.Unpickler. Modern
    # versions require weights_only=False for our NumPy/Python RNG state.
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        kwargs["weights_only"] = False
    return torch_module.load(path, **kwargs)
