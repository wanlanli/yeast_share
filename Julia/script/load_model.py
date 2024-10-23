def load_pytroch_model():
    import sys, os
    sys.path.insert(0, os.path.abspath('/home/liwa/project/OneFormer/detectron2/'))
    sys.path.insert(0, os.path.abspath('/home/liwa/project/OneFormer/'))
    sys.path.insert(0, os.path.abspath('/home/liwa/project/OneFormer/detectron2/projects/Panoptic-DeepLab/'))
    # os.environ['DETECTRON2_DATASETS'] = '/home/liwa/data/datasets/'

    from detectron2.config import get_cfg
    from detectron2.projects.panoptic_deeplab import (
        add_panoptic_deeplab_config,
    )
    from prediction import Predictor

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8700m"
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("/home/liwa/data/oneformer_output/output/config.yaml")

    cfg.MODEL.WEIGHTS = os.path.abspath("/home/liwa/data/oneformer_output/output/model_0059999.pth")
    predictor = Predictor(cfg)
    return predictor


def load_tensorflow_model():
    pass
