# Week 3 & 4: Object detection
- Task (a): run the provided code. Use the preconfigured experiment file tt100k_detection.py to detect traffic signs with the YOLOv2 model.
  * [x] Analyze the dataset
  * [x] Calculate the f-score and FPS on train, val and test sets

  YOLO architecture
  
   |Metric                 | Train         | Test      | Valid    |
   |:------------:         |:-------------:| :-----:   |:---:     |
   |  F-score              | 0.782425      | 0.703992  |0.266666  |
   |  FPS                  | 18.62149      | 15.70237  |10.51580  |
  
  Tiny-yolo architecture
  
   |Metric                 | Train         | Test      | Valid    |
   |:------------:         |:-------------:| :-----:   |:---:     |
   |  F-score              | 0.352980      | 0.273276  |0.138888  |
   |  FPS                  | 29.82082      | 23.35386  |14.37156  |
  
  * [x] Evaluate different network architectures:
    * [x] YOLO
    * [x] Tiny-yolo
> INSTRUCTIONS: There are one folder for each network architecture, taskA/yolo and taskA/tiny-yolo. On each folder, there are 2 files and 2 folders: one bash script to execute the experiment, one txt file that contains the execution of previous bash script, one results folder, where are the results of experiments executed, and one metrics folder, that contains the execution of evaluation detection score for train, test and validation dataset. 
    
- Task (b): Read two papers 
  * [x] You Only Look at Once (YOLO)
  * [x] Single Shot Multi-Box Detector (SSD), or another
- Task (c): Train the network on a different dataset 
  * [x] Set-up a new experiment file to detect among cars, pedestrians, and trucks on Udacity dataset
  * [x] Use the YOLOv2 model as before, but increment the number of epochs to 40
  * [x] Analyze the problems of the dataset as it is. Propose (and implement) solutions
> INSTRUCTIONS: There are 6 main files on folder taskC: 2 of them, are scripts to execute yolo architecture with Udacity dataseset, one of them with 10 epochs, and the other with 40 epochs. Also, there are 2 log files that contains the execution of previous scripts. Moreover, there are 2 folders with results of the experiments executed by scripts.  
  
- Task (d): Implement a new network
  * [x] We provide you a link to a Keras implementation of SSD. Other models will be highly valued
  * [x] Integrate the new model into the framework
  * [x] Evaluate the new model on TT100K and Udacity
> INSTRUCTIONS: New files included on ../models/ssd.py, ../metrics/ssd_training.py, and ../layers/ssd_layers.py. Files modified: model_factory.py and ../metrics/metrics.py. SSD implemented into framework, but not tested with any dataset. 
- [x] Task (e): Boost the performance of your network
> INSTRUCTIONS: Data augmentation applied on images activating next flags on config file: norm_featurewise_std_normalization (divide dst - dataset) and norm_featurewise_center (Substract mean - dataset). The aim is normalize image with respect to dataset to boost the performance of YOLO network.
- [x] Task (f): Write the report for Weeks 3/4
