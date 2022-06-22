from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class MyTrainer(DefaultTrainer):
  def build_evaluator(self, dataset_name):
    return COCOEvaluator(dataset_name, ("bbox",))