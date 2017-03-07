# Computer Vision Master
- University: Universitat Autònoma de Barcelona
- Module: M5 Visual Recognition ([link module](http://pagines.uab.cat/mcv/content/m5-visual-recognition))
- Team number: 07

## Members
- Marc Grau (mgraugal@gmail.com)
- Marc Carné (marc.carne.herrera@gmail.com)
- Gonzalez Benito (gonzabenito@gmail.com)
- Víctor Segura (victor.seguratir@e-campus.uab.cat)

## Abstract
The goal of this project is to learn the basic concepts and techniques to build deep neural networks to detect, segment and recognize specific objects, focusing on images recorded by an on-board vehicle camera for autonomous driving.

## Report
Download the [Overleaf document](https://www.overleaf.com/read/qrjbtzwtjhmx)

## Slides
- Week 2: Object recognition ([link to google slides](https://docs.google.com/presentation/d/1vxO2lUGGYYm7yVjZjYvC0wNX4iJOQ6GoatuUw1deRxo/edit?usp=sharing))
- Week 3: Object detection ([link to google slides](https://docs.google.com/presentation/d/13U6bP7de293dzYGg2hDxrb1G0avuuknWLXXqld0XJSc/edit?usp=sharing))
- Week 4: Object detection ([link to google slides](https://docs.google.com/presentation/d/1b2lfVdsAQIWvKVn91t1U2P0ArWU6lN7sQeEBwFc4i4I/edit?usp=sharing))
- Week 5: Image semantic segmentation ([link to google slides](https://docs.google.com/presentation/d/1n6GLVJBKHYrHsap9NxRhFaoHRneM-YElXpyJ6k2GUUk/edit?usp=sharing))
- Week 6: Image semantic segmentation ([link to google slides](https://docs.google.com/presentation/d/1uuh2UWdc-UNsiDahuFWqYH3zyfSxUSrR9g7xiE7CoGM/edit?usp=sharing))

## Papers Summaries
#### Very Deep Convolutional Networks for Large-scale Image Rcognition
This paper shows a work oriented to study the effects of depth on the accuracy in CNNs. Their developers focused on networks with very small convolutional filters which allowed them to add more conv layers. Particularly, they explored the use of 16 and 19 layers based on the hypothesis that more layers with smaller filters provide better results than shallow nets.
In order to provide more discriminative power to the decision function, the stack of 3 small convolutional layers instead of a single big one was implemented. They also set the convolution stride at 1 and use an according padding to preserve spatial resolution. Max pooling are established at 2 by 2 elements with a stride of 2 also. The activation functions of the implementation consist entirely on ReLU. At the bottom of the network, 3 fully connected layers followed by a softmax which makes the decision are set.
Authors explain the training and testing details following their participation on the ImageNet Challenge (ILSVRC). For this, they performed pre processing on the training images by substracting the mean RGB value calculated on the training set from each pixel.
Trining was made following scale variation methods, with size jittering to provide the network with a certain size independence.
They finally conclude by comparing their results with the state of the art at the time of the publication. Results support their hypothesis on how it is more effective to generate a deeper net with small filters than using large shallow convolutional layers. Appendices incorporate further information on their results on tasks other than classification, such as localisation, use of VGG as feature extractors and the generalized implementation by using other smaller datasets than the ILSVRC one.

#### Deep Residual Learning for Image Recognition
ResNet is a convolutional neural network model that allows deeper architecture without degenerating the results (increasing the train error). The main idea of these networks is to fit the learning of the layers to a function that is a residual mapping (F(x)) instead of to an underlying mapping (H(x)). By this, we can express the underlying function as: H(x)=F(x)+x. Physically this is a shortcut in the structure. The advantage is that the solver (the part of the algorithm that train the network) optimize easily and faster that function), without degenerate the results. Using that approach we can have deeper networks with low training error and high detail in the descriptors, using from 18 layers to more than 500 or 1000 layers having good results. Still, we can suffer overfitting in the very deep architectures due to the features learned being too particular for the training set. 
In the paper the authors talked about a different block structure before the shortcut -how many convolutional layers and their size- and how to deal with the increasing dimension of the feature maps as in order to compute the residual the vectors or maps must have the same size. Authors propose to use a zero padding, the identity transform for the maps that have the same size and a projection shortcuts for those that have different size, or to use projection shortcuts for all the connections. At the end they present the results over different challenges showing a high performance on them.
