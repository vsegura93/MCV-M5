#### SSD
As we mentioned in the last subsection, some of the state-of-the art methods needs an object proposal, that returns a set 
of bounding boxes that are object candidates before given any input to the neural network. Some techniques as sliding window 
and other object proposals are widely used in this field, but a novel technique has appeared and performs with respect to other
region proposals methods.This work instead of give a big number of possible bounding boxes per image, discretizes the output space 
of bounding boxes into a set of default boxes with different aspect ratio and scales directly on the feature map (at different 
scales).

In the same network at prediction time this method compute the probability of having an object (and the class) inside the region
and also produce some adjustments to the bounding box to better match the object shape. To do that, the network combines 
predictions from multiple feature map (different network depth) to obtain a very good performance doing it scalable. Due to the
discretization of possible bounding boxes it produces, the computation effort decrease being fast (more than YOLO). The core of 
the SSD predicts category (class) scores and box size adjustments (with respect to the predefined size) using small convolutional
filters applied to the feature maps. As the method is applied at different feature maps, makes predictions at different scales.

As in YOLO a single network performs the detection task, so the train can be done end-to-end, without any intermediate loss.
Just giving the input image and the ground truth boxes the network can learn itself to detect objects. The most interesting thing
about this method is that is performed at different scales so from a truncated base network (from the VGG 16 network architecture),
a set of convolutional layers were added, giving smaller feature maps each time. This requires to specify at train stage to which
scale the ground truth bounding box belongs to. Although the performance is increased with analyse different scales, we have to 
have into account bounding boxes we might discard, and this means give to the network the appropriate hard negative samples.

An interesting think of that approach is that the main goal of discretize bounding boxes adapts the object detector to our own 
task, as the aspect ratio of the bounding boxes are directly related with the objects we need to detect (as for example:
pedestrians, bicycles...). Object size is very important in this approach and for that reason the authors made lots of tests
with different data augmentation techniques to reason about the performance achieved doing that and they conclude data 
augmentation is crucial to detect small objects.
