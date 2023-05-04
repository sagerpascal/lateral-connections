try:
    # FFCV only for machines with GPUs
    from src.data.loader.ffcv_loader import get_ffcv_data_loaders, get_image_pipeline as get_ffcv_image_pipeline, \
        get_label_pipeline as get_ffcv_label_pipeline
except ImportError as e:
    from src.utils.custom_print import print_warn

    print_warn("FFCV is not installed. Please install it to use it.")
    print_warn(e)
    get_ffcv_data_loaders = None
    get_ffcv_image_pipeline = None
    get_ffcv_label_pipeline = None
from src.data.loader.torch_loader import get_torch_data_loaders
