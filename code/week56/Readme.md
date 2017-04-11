# Week 5 & 6: Image Semantic Segmentation

- Task (a): Run the provided code. Use the preconfigured experiment file (camvid_segmentation.py) to segment objects with the FCN8 model
   * [x]  Analyze the dataset
   * [x] Evaluate on train, val and test sets
> INSTRUCTIONS: There are 2 files and 1 folder on folder taskA: one bash script to execute the experiment, one txt file that contains the execution of previous bash script, and one results folder where are the results of experiments executed.
  
 - Task (b): Read two papers 
   * [x] Fully convolutional networks for semantic segmentation (Long et al. CVPR, 2015)
   * [x] Another paper of free choice
> INSTRUCTIONS: papers are summarieze on root directory /papers_summaries

 - Task (c): Train the network on a different dataset 
   * [x] Set-up a new experiment file to image semantic segmentation on another dataset (Cityscapes, KITTI,  Synthia, ...)
   * [x] Use the FCN8 model as before
> INSTRUCTIONS: There are 2 files and 1 folder on folder taskC: one bash script to execute the experiment, one txt file that contains the execution of previous bash script, and one results folder where are the results of experiments executed. We choose Cityscapes as dataset to compute previous experiment.
   
 - Task (d): Implement a new network 
   * [x] Select one network from the state of the art (SegnetVGG, DeepLab, ResnetFCN, ...)
   * [x] Integrate the new model into the framework
   * [x] Evaluate the new model on CamVid. Train from scratch and/or fine-tune
> INSTRUCTIONS: There are 2 files and 1 folder on folder taskD: one bash script to execute the experiment, one txt file that contains the execution of previous bash script, and one results folder where are the results of experiments executed. We choose SegnetVGG as network to compute previous experiment.
   
 - [x] Task (e): Boost the performance of your network
> INSTRUCTIONS: Implementation of SegNet architecture. Evaluation of performance on Camvid dataset, trained from scratch. Folder of results and scripts of execution of experiments are located on folder taskE. Main file included to implement architecture: models/segnet.py.
 
 - [x]  Task (f): Write the report for Weeks 5/6
