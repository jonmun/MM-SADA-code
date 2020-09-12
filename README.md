# Multi-Modal Domain Adaptation for Fine-grained Action Recognition
This is the official implementation for Multi-modal Self-Supervised Adversarial Domain Adapatation (MM-SADA). 
 ([Project](https://jonmun.github.io/mmsada/), [Publication](https://openaccess.thecvf.com/content_CVPR_2020/html/Munro_Multi-Modal_Domain_Adaptation_for_Fine-Grained_Action_Recognition_CVPR_2020_paper.html))

## BibTeX
If this repository was utilised, please cite:
```
@InProceedings{munro20multi,
author = "Munro, Jonathan and Damen, Dima",
title = "{M}ulti-modal {D}omain {A}daptation for {F}ine-grained {A}ction {R}ecognition",
booktitle = "Computer Vision and Pattern Recognition (CVPR)",
year = "2020"
}
```

## Requirements  
1. Download RGB and Optical Flow frames from the EPIC-KITCHENS-55 dataset from participants P01, P08 and P22. You can used the download script provided by the EPIC-KITCHENS team: https://github.com/epic-kitchens/epic-kitchens-download-scripts. The directory structure should be modified to match:

```
├── rgb
|   ├── P01
|   |   ├── P01_01
|   |   |   ├── frame_0000000000.jpg
|   |   |   ├── ...
|   |   ├── P01_02
|   |   ├── ...
|   ├── P08
|   |   ├── P08_01
|   |   ├── ...
|   ├── P22
|   |   ├── P22_01
|   |   ├── ...

├── flow
|   ├── P01
|   |   ├── P01_01
|   |   |   ├── u 
|   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   ├── ...
|   |   |   ├── v
|   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   ├── ...
|   |   ├── P01_02
|   ├── P08
|   |   ├── P08_01
|   |   ├── ..
|   ├── P22
|   |   ├── P22_01
|   |   ├── ..
```
        

2. Clone https://github.com/deepmind/kinetics-i3d into the current repo. This contains the pretrained I3D models on kinetics.

3. Install TensorFlow 1.12, Sonnet and other dependencies. The conda environment used to replicate the results is sepecified in: ```environment.yml```.

## Steps to Train
The folder ```scripts``` contains bash scripts to reproduce the ablation results. To re-create MM-SADA results:

1. Modify ```rgbdatapath```and ```flowdatapath``` variables in ```scripts/submit_mmsada.sh``` to match the location of EPIC-KITCHENS' RGB and Optical Flow frames.

2. Run the bash script:```scripts/submit_mmsada.sh``` to train MM-SADA

Note: The MM-SADA paper incorrectly states the Adam optimiser is used. To reproduce results, SGD with momentum optimiser should be used. All other hyper-parameters stated correctly in the paper. 

## Steps for Evaluation

1. Modify ```rgbdatapath```and ```flowdatapath``` variables in ```scripts/submit_test.sh``` to match the location of EPIC-KITCHENS' RGB and Optical Flow frames.

2. Run the bash script```scripts/submit_test.sh``` to obtain top-1 source and target domain accuracies from saved checkpoints. To evaluate a different experiment change ```experimentdirectory``` to the folder of the experiment to be evaluated.

3. Results are saved to the models results directory e.g. ```EPIC_D2_D3_mmsada/results_D2_D3_0.001_0.9_joint/logs/results.list```


## Acknowledgement
Research supported by EPSRC LOCATE (EP/N033779/1) and EPSRC Doctoral Training Partnershipts (DTP). The authors acknowledge and value the use of the ESPRC funded Tier 2 facility, JADE.

## Licence

All files in this repository are copyright by us and published under the Creative Commons Attribution-NonCommerial 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc/4.0/). This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes.

