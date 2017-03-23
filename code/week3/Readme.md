# Week 3: Object detection
- Task(a): run the provided code. Use the preconfigured experiment file tt100k_detection.py to detect traffic signs with the YOLOv2 model.
  * [x] Analyze the dataset
  * [ ] Calculate the f-score and FPS on train, val and test sets
  * [x] Evaluate different network architectures:
    * [x] YOLO
    * [x] Tiny-yolo 
There are one folder for each network architecture, taskA/yolo and taskA/tiny-yolo. On each folder, there are 3 files: one   bash script to execute the experiment, one txt file that contains the execution of previous bash script, and results folder, where the result of exepriment is saved. 
    
- Task(b): Read two papers 
  * [ ] You Only Look at Once (YOLO)
  * [ ] Single Shot Multi-Box Detector (SSD), or another
- Task(c): Train the network on a different dataset 
  * [ ] Set-up a new experiment file to detect among cars, pedestrians, and trucks on Udacity dataset
  * [ ] Use the YOLOv2 model as before, but increment the number of epochs to 40
  * [ ] Analyze the problems of the dataset as it is. Propose (and implement) solutions
- Task(d): Implement a new network
  * [ ] We provide you a link to a Keras implementation of SSD. Other models will be highly valued
  * [ ] Integrate the new model into the framework
  * [ ] Evaluate the new model on TT100K and Udacity
- Task (e): Boost the performance of your network
- Task (f): Write the report for Weeks 3/4
