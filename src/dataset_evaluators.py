from detectron2.evaluation import DatasetEvaluators, COCOEvaluator
from detectron2.utils.comm import is_main_process

class MyDatasetEvaluators(DatasetEvaluators):
    def evaluate(self):
        results = []
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                results.append(result)
        return results