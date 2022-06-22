import torch, torchvision, detectron2, os, sys, getopt

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from src.data import register_train, register_test
from src.trainer import MyTrainer

if __name__ == "__main__":
  resume = False
  cfg_path = ""

  try:
    opts, args = getopt.getopt(sys.argv[1:], "c:r:")

    for opt, arg in opts:
      if opt == "-c":
        cfg_path = arg

      if opt == "-r":
        if arg == 'True':
          resume = True

  except:
    print("-c config_path -r if_resume")
    sys.exit(2)

  setup_logger()

  register_train()
  register_test()

  cfg = get_cfg()
  cfg.merge_from_file(cfg_path)

  print(cfg)

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  trainer = MyTrainer(cfg)
  trainer.resume_or_load(resume=resume)
  trainer.train()