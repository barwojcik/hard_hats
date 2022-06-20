# Dataset

## Images

We used a publicly available dataset [1] for training and testing. This dataset contains 7035 images of different sizes, split into train (5269 images) and test (1766 images) part.

## Annotations

The annotations are provided in coco json format. We have annotated images for two different tasks, separate detection of hard hats, humans and head keypoint localization (*hard hat+person.json*), and direct detection of hard hat wearers and non-wearers (*wearing.json*).

Full breakdown of the instances:
|Instances|all|small|medium|large|
|---|---|---|---|---|
Train:
|hard hat|17,741|11,340|5,922|479|
|person|23,882|2,805|9,729|11,348|
|- w/ head keypoint|22,983|2,602|9,232|11,149|
|- w/ head keypoint wearing a hard hat|16,700|1,715|6,459|8,526|
Test:
|hard hat|5746|3727|1841|178|
|person|7992|1077|3200|3715|
|- w/ head keypoint|7775|1036|3065|3674|
|- w/ head keypoint wearing a hard hat|5353|509|2071|2773|
            
1. Xie, Liangbin, 2019, "Hardhat", https://doi.org/10.7910/DVN/7CBGOS, Harvard Dataverse
