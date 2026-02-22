# FastSpiker: Enabling Fast Training for Spiking Neural Networks on Event-based Data Through Learning Rate Enhancements for Autonomous Embedded Systems

Autonomous embedded systems (e.g., robots) typically necessitate intelligent computation with low power/energy processing for completing their tasks. Such requirements can be fulfilled by embodied neuromorphic intelligence with spiking neural networks (SNNs) because of their high learning quality (e.g., accuracy) and sparse computation. Here, the employment of event-based data is preferred to ensure seamless connectivity between input and processing parts. However, state-of-the-art SNNs still face a long training time to achieve high accuracy, thereby incurring high energy consumption and producing a high rate of carbon emission. Toward this, we propose FastSpiker, a novel methodology that enables fast SNN training on event-based data through learning rate enhancements targeting autonomous embedded systems. In FastSpiker, we first investigate the impact of different learning rate policies and their values, then select the ones that quickly offer high accuracy. Afterward, we explore different settings for the selected learning rate policies to find the appropriate policies through a statistical-based decision. Experimental results show that our FastSpiker offers up to l0.5x faster training time and up to 88.39% lower carbon emission to achieve higher or comparable accuracy to the state-of-the-art on the event-based automotive dataset (i.e., NCARS). In this manner, our Fast-Spiker methodology paves the way for green and sustainable computing in realizing embodied neuromorphic intelligence for autonomous embedded systems.

## Create Conda Environment (if required): 
```
conda create --name fastspiker python=3.8
```

## Installation: 
Ensure to fulfill the library requirements:
```
pip install numpy torch torchvision
```

## Preparation: 
Prepare the working folders as shown like this figure. 
<p align="left"><img width="40%" src="docs/folders.png"/></p>

To do this, first prepare the original N-CARS dataset (n-cars_test & n-cars_train), which can be downloaded from this [link](https://www.prophesee.ai/2018/03/13/dataset-n-cars/).

Then, generate the modified dataset (N_cars) using matlab scripts, which will create the N_cars folder.   

Afterwards, create "Trained_100" folder and run the example below.

## Example of command to run the code:
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --filenet ./net/net_1_4a32c3z2a32c3z2a_100_100_no_ceil.txt --fileresult ./results/exp_warmrestart_2peaks --batch_size 40 --lr 1e-3 --lr_decay_epoch 20 --lr_decay_value 0.5 --lr_policy 1 --threshold 0.4 --att_window 100 100 0 0 --sample_length 10 --sample_time 1 
```

## Citation
If you use FastSpiker in your research or find it useful, kindly cite the following [article](https://www.frontiersin.org/article/10.3389/frobt.2024.1401677):
```
@INPROCEEDINGS{Ref_Bano_FastSpiker_ICARCV24,
  author={Bano, Iqra and Wicaksana Putra, Rachmad Vidya and Marchisio, Alberto and Shafique, Muhammad},
  booktitle={2024 18th International Conference on Control, Automation, Robotics and Vision (ICARCV)}, 
  title={FastSpiker: Enabling Fast Training for Spiking Neural Networks on Event-based Data Through Learning Rate Enhancements for Autonomous Embedded Systems}, 
  year={2024},
  volume={},
  number={},
  pages={428-434},
  keywords={Training;Accuracy;Embedded systems;Event detection;Neuromorphics;Carbon dioxide;Spiking neural networks;Robots;Faces;Automotive engineering},
  doi={10.1109/ICARCV63323.2024.10821701}}

```

This work is inspired from the work of [SNN4Agents] (https://doi.org/10.3389/frobt.2024.1401677).
