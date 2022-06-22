import os, sys, getopt

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.evaluation import inference_on_dataset

from src.data import register_train, register_direct_test
from src.dataset_evaluators import MyDatasetEvaluators
from src.class_coco_evaluator import ClassSpecificCOCOEvaluator

import src.model

if __name__ == "__main__":
  cfg_path = ""

  try:
    opts, args = getopt.getopt(sys.argv[1:], "c:")

    for opt, arg in opts:
      if opt == "-c":
        cfg_path = arg

  except:
    print("-c config_path")
    sys.exit(2)

  setup_logger()

  register_train()
  register_direct_test()

  cfg = get_cfg()
  cfg.merge_from_file(cfg_path)
  cfg.DATASETS.TRAIN = ("train",)
  cfg.DATASETS.TEST = ("direct_test",)
  cfg.MODEL.META_ARCHITECTURE = "MyGeneralizedRCNN"
  cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
  cfg.SOLVER.IMS_PER_BATCH = 1

  print(cfg)

  model = build_model(cfg)

  coco_evaluator = ClassSpecificCOCOEvaluator("direct_test", ("bbox", ), False, output_dir=cfg.OUTPUT_DIR + "/coco_results_wearing/", kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS, class_ids = [1,2])
  hh_on_coco_evaluator = ClassSpecificCOCOEvaluator("direct_test", ("bbox", ), False, output_dir=cfg.OUTPUT_DIR + "/coco_results_wearing/", kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS, class_ids = [1])
  hh_off_coco_evaluator = ClassSpecificCOCOEvaluator("direct_test", ("bbox",), False, output_dir=cfg.OUTPUT_DIR + "/coco_results_wearing/", kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS, class_ids = [2])

  evaluators = MyDatasetEvaluators([coco_evaluator, hh_on_coco_evaluator, hh_off_coco_evaluator])
  loader = build_detection_test_loader(cfg,"direct_test", mapper = DatasetMapper(cfg, is_train=False, augmentations=[]))

  results = inference_on_dataset(model, loader, evaluators)
  print(results)