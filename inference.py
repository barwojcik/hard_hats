import sys, getopt, cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer, ColorMode

import src.model

if __name__ == "__main__":
  cfg_path = ""
  im_path = ""

  try:
    opts, args = getopt.getopt(sys.argv[1:], "c:i:")

    for opt, arg in opts:
      if opt == "-c":
        cfg_path = arg
      if opt == "-i":
        im_path = arg

  except:
    print("-c config_path -i im_path")
    sys.exit(2)

  metadata = Metadata()
  metadata.set(thing_classes = ['hard hat: on', 'hard hat: off', 'hard hat: unknown'])
  metadata.set(thing_colors = [(0,255,0), (0,0,255), (255,0,0)])

  image = utils.read_image(im_path, format="BGR")

  cfg = get_cfg()
  cfg.merge_from_file(cfg_path)
  cfg.MODEL.META_ARCHITECTURE = "MyGeneralizedRCNN"  
  cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
  cfg.SOLVER.IMS_PER_BATCH = 1

  print(cfg)

  predictor = DefaultPredictor(cfg)
  out = predictor(image)

  vis = Visualizer(image, metadata=metadata, scale=2, instance_mode=ColorMode.SEGMENTATION)
  output_im = vis.draw_instance_predictions(out["instances"].to("cpu"))

  cv2.imwrite('output.png', output_im.get_image())


