import torch
import numpy as np
from typing import Dict, List, Optional

from detectron2.structures import Instances 
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess


def checkIfIn(kp, bbox):
  if(bbox[0] <= kp[0] <= bbox[2]):
    if(bbox[1] <= kp[1] <= bbox[3]):
      return True
  return False


def instance_postprocess(results: Instances, output_height: int, output_width: int):
  instances = detector_postprocess(results, output_height, output_width)

  hhs = instances[instances.pred_classes == 0]
  ppl = instances[instances.pred_classes == 1]

  for idx in range(len(ppl)):
    new_cls = 2

    if ppl.pred_keypoints[idx][0][2].item() > 0.05:
      new_cls = 1

      kp = [ppl.pred_keypoints[idx][0][0].item(),
            ppl.pred_keypoints[idx][0][1].item()]

      for idx2 in range(len(hhs)):
        bbox = [hhs[idx2].pred_boxes.tensor[0,0].item(),
                hhs[idx2].pred_boxes.tensor[0,1].item(),
                hhs[idx2].pred_boxes.tensor[0,2].item(),
                hhs[idx2].pred_boxes.tensor[0,3].item()]

        if checkIfIn(kp, bbox):
          new_cls = 0
          break

    ppl.pred_classes[idx] = new_cls

  new_instances = Instances(instances.image_size)
  new_instances.set('pred_classes', ppl.pred_classes)
  new_instances.set('pred_boxes', ppl.pred_boxes)
  new_instances.set('scores', ppl.scores)
  
  return new_instances


# This is original detectron2 class where only postprocessing is changed to include hard hat wearing check
@META_ARCH_REGISTRY.register()
class MyGeneralizedRCNN(GeneralizedRCNN):
  def inference(
    self,
    batched_inputs: List[Dict[str, torch.Tensor]],
    detected_instances: Optional[List[Instances]] = None,
    do_postprocess: bool = True,
  ):
    """
    Run inference on the given inputs.
    Args:
        batched_inputs (list[dict]): same as in :meth:`forward`
        detected_instances (None or list[Instances]): if not None, it
            contains an `Instances` object per image. The `Instances`
            object contains "pred_boxes" and "pred_classes" which are
            known boxes in the image.
            The inference will then skip the detection of bounding boxes,
            and only predict other per-ROI outputs.
        do_postprocess (bool): whether to apply post-processing on the outputs.
    Returns:
        When do_postprocess=True, same as in :meth:`forward`.
        Otherwise, a list[Instances] containing raw network outputs.
    """
    assert not self.training

    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)

    if detected_instances is None:
        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        results, _ = self.roi_heads(images, features, proposals, None)
    else:
        detected_instances = [x.to(self.device) for x in detected_instances]
        results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

    if do_postprocess:
        assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        return MyGeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
    else:
        return results
  
  @staticmethod
  def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
      """
      Rescale the output instances to the target size.
      """
      # note: private function; subject to changes
      processed_results = []
      for results_per_image, input_per_image, image_size in zip(
          instances, batched_inputs, image_sizes
      ):
          height = input_per_image.get("height", image_size[0])
          width = input_per_image.get("width", image_size[1])
          r = instance_postprocess(results_per_image, height, width)
          processed_results.append({"instances": r})
      return processed_results