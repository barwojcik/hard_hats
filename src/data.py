from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def register_train():
  register_coco_instances("train", {}, 
                        "dataset/annotations/train/train_hard hat+person.json",
                        "dataset/images/train")

  MetadataCatalog.get('train').keypoint_names = ["head"]
  MetadataCatalog.get('train').keypoint_flip_map = []
  MetadataCatalog.get('train').keypoint_connection_rules = []

def register_test():
  register_coco_instances("test", {}, 
                        "dataset/annotations/test/test_hard hat+person.json",
                        "dataset/images/test")

  MetadataCatalog.get('test').keypoint_names = ["head"]
  MetadataCatalog.get('test').keypoint_flip_map = []
  MetadataCatalog.get('test').keypoint_connection_rules = []

def register_direct_train():
  register_coco_instances("direct_train", {}, 
                        "dataset/annotations/train/train_wearing.json",
                        "dataset/images/train")

def register_direct_test():
  register_coco_instances("direct_test", {}, 
                        "dataset/annotations/test/test_wearing.json",
                        "dataset/images/test")
