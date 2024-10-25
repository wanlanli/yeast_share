def load_pytroch_model():
    import sys, os
    sys.path.insert(0, os.path.abspath('/home/wlli/project/OneFormer/detectron2/'))
    sys.path.insert(0, os.path.abspath('/home/wlli/project/OneFormer/'))
    sys.path.insert(0, os.path.abspath('/home/wlli/project/OneFormer/detectron2/projects/Panoptic-DeepLab/'))
    # os.environ['DETECTRON2_DATASETS'] = '/home/wlli/Data/datasets/'

    from detectron2.config import get_cfg
    from detectron2.projects.panoptic_deeplab import (
        add_panoptic_deeplab_config,
    )
    from prediction import Predictor

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8700m"
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("/home/wlli/Data/oneformer_output/config.yaml")

    cfg.MODEL.DEVICE = "cpu"
    # into cpu model
    cfg.MODEL.SEM_SEG_HEAD.NORM = "BN"
    cfg.MODEL.INS_EMBED_HEAD.NORM = "BN"
    cfg.MODEL.RESNETS.NORM = "BN"

    cfg.MODEL.WEIGHTS = os.path.abspath("/home/wlli/Data/oneformer_output/model_0059999.pth")
    predictor = Predictor(cfg)
    return predictor


def load_tensorflow_model():
    import sys
    sys.path.append("/home/wlli/project_in_wd/deepYeast/deeplab/")
    import yaml
    from config_yml import ExperimentOptions
    from trainer.train import DeepCellModule
    model_dir = "/home/wlli/Data/deepyeast_model/v_1.0.0/checkpoint/"
    num_gpus = 0
    config_path = "/home/wlli/project_in_wd/deepYeast/deeplab/configs/config_wl.yaml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    configs = ExperimentOptions(config)
    predictor = DeepCellModule("test", configs, num_gpus, model_dir=model_dir)
    return predictor


def predict(model, image):
    pass
