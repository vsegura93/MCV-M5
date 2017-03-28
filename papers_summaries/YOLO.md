#### Very Deep Convolutional Networks for Large-scale Image Recognition
  There is one state-of-the art method that has the best performance in object detection and also is the fastest one. 
YOLO, you only look once, proposed a method for object detection that consists in a single neural network that having an image
as input give as a result a bounding box and the probability to belong to each class. The main idea is to explore directly the 
feature map and from there obtain bounding boxes and the class where the bounding box belong without having an object candidates
extraction method, seen the problem as a single regression problem. By this way we have an end-to-end system that can be trained
just one with backpropagation. This method extracts bounding boxes from the feature map result as output of a convolutional layer
and grid it. From that cell the algorithm predict B bounding boxes (for each one four values are predicted: center coordinates, 
height and width) and the confidence score for that boxes, resulting from the product between the probability of having an object
on the bounding box and the intersection over union (IoU) with respect to the groundtruth; so if there is not an object in the 
bounding box the confidence must be equal to 0. 

  Apart from the bounding box the output of the network per cell is a vector of C 
dimensions corresponding to the probability of the objects that the bounding box contains to belong to each class. By this way 
the output of the network is encoded into a S*S*(B*5)+C tensor, where S corresponds to the grid size, B the number of bounding 
boxes and C the number of classes. This work has second version that perform the first one and also mostly all the rest of the 
state-of-the art techniques. This neural network based model consist in a single convolutional network that simultaneously 
predicts multiple bounding boxes and class probability for those boxes. The network is optimized directly on the detection given
as an input a full image. Thanks to the special architecture of that approach despite the training can be an effort, in the 
inference time the network is very fast, achieving 45 frames per second in the original implementation and 155 frames per second 
in a small version (with less number of layers) despite of having double performance with respect to the other that performs the 
state of the art. The advantages of that network are: first YOLO is very fast and not requires a complex pipeline (no need object
proposal method as R-CNN). Second, reason over all the image when make predictions (other techniques uses sliding window or 
region proposal techniques, so a single image patch is pass-forward through the network). Third, in the shallow layers of YOLO
the network learns generalizable representation of objects so it presents good results over new domains. The last important think
is that as the bounding box is normalized with respect to the image size, so no depend on the grid size. The architecture of the
network consist basically in convolutional layers to extract the features and fully connected layers that predict the output 
probabilities and the coordinates. 
  
  An important issue is that during the training stage, the network is trained for a 
classification purpose on ImageNet dataset, with the objective to learn object features and then fine-tune the model to 
implement the specific task (adding four convolutional layers and two fully connected layers with random initialization).
Despite the high performance and the fast inference, this technique present some limitations. First of all, YOLO has spatial 
limitation on bounding box as each cell can just predict a certain number (predefined) of bounding boxes. Another drawback is
that the algorithm has some problem in little objects, taking a group of them as a single object, for example blocks of birds.
One limitation but not too worrying is that the error on small and large bounding boxes are treated by the same way (small error
on large bounding box is not bad as small error on small bounding boxes, that has a high effect on the intersection over union, 
IoU).

  Seen the advantages of YOLO we propose this neural network model to perform object detection in our task, applying it to a 
newer dataset, as is a unified model for object detection, simple to construct and can be trained directly on full images 
(without any intermediate supervision).
