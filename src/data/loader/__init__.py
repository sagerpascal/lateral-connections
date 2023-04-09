try:
    from data.loader.ffcv_loader import get_ffcv_data_loaders, get_image_pipeline as get_ffcv_image_pipeline, get_label_pipeline as get_ffcv_label_pipeline
except ImportError as e:
    print("FFCV is not installed. Please install it to use it.")
    print(e)
    get_ffcv_data_loaders = None
    get_ffcv_image_pipeline = None
    get_ffcv_label_pipeline = None
from data.loader.torch_loader import get_torch_data_loaders