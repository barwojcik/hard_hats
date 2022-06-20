# Hard hat wearing detection based on head keypoint localization :rescue_worker_helmet:

This is an implementation of a solution developed for hard hat wearing detection. A combination of object detection and head keypoint localization is proposed. In tests, this solution surpassed the previous methods based on the relative bounding box position of different instances, as well as direct detection of hard hat wearers and non-wearers.

<p align="center">
  <img src="/ilustrations/fig1-1.png" width=600/>
</p>

## Results

|Model|AP|AP50|AP75|APs|APm|APl|
|----|----|----|----|----|----|----|
Overall:
|Ours|**0.675**|**0.826**|**0.759**|0.211|**0.682**|**0.817**|
|Direct detection|0.663|**0.826**|0.757|**0.248**|0.670|0.809|
|Decision tree|0.664|0.815|0.746|0.222|0.668|0.799|
Hard hat wearer:
|Ours|0.710|0.871|0.805|0.247|0.693|0.805|
|Direct detection|**0.723**|**0.905**|**0.824**|**0.331**|**0.700**|**0.809**|
|Decision tree|0.698|0.860|0.794|0.248|0.681|0.789|
Hard hat non-wearer:
|Ours|**0.641**|**0.780**|**0.714**|0.175|**0.671**|**0.828**|
|Direct detection|0.603|0.747|0.690|0.165|0.639|0.810|
|Decision tree|0.630|0.769|0.698|**0.197**|0.655|0.808|

## Citing 

```BibTeX
@misc{wojcik2021hardhats,
  author = {Wójcik, Bartosz and Żarski, Mateusz and Książek, Kamil and Miszczak, Jarosław Adam and Skibniewski, Mirosław Jan},
  title = {Hard hat wearing detection based on head keypoint localization},
  publisher = {arXiv},
  year = {2021},
  doi = {10.48550/ARXIV.2106.10944},  
  url = {https://arxiv.org/abs/2106.10944},
}
