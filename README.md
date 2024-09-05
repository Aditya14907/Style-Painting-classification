# Style-Painting-classification

The rise of digitalization in the art world highlights the growing need for classifying paintings based on styles, artists, and genres. Style-based classification plays a crucial role in aiding both visitors and curators to explore and analyze artworks in museums more efficiently and at their own pace. Additionally, identifying the style of a painting can be challenging, as artists often develop unique interpretations of the same style, and multiple artists can share similar stylistic approaches. Accurate style classification can provide deeper insights into artistic trends and help preserve the integrity of fine art collections.

# Models
I have experimented on five models

* ResNet50
* InceptionV3
* EfficientNetB3
* ResNet152
* DenseNet161

# Dataset
The dataset used for this classification task is Wikiart dataset obtained from reference [WikiArt](https://www.wikiart.org/) .The dataset consists around 40,000 images with each image is labeled to its corresponding style.
We have used 12 style painting classes which includes Impressionism,Expressionism,Baroque,Realism and others.

The dataset is split into training,validation sets :

Training set : 80% of images
Validation set : 20% of images
Each image is pre-processed to a fixed input size suitable for respective model's architecture.
We have used ImageDataGenerator for data split and applied data augmentation of Height Shift rotate and Width Shift rotate.

Model Architectures
This project implements five deep learning architectures

# ResNet50
ResNet-50 (Residual network with 50 layers) is a deep convolutional neural network. It was introduced by [Kaiming He](https://arxiv.org/abs/1512.03385) in 2015. The main use of ResNet50 is to solve the problem of vanishing gradients using residual block that allow direct flow of information.
ResNet-50 takes an input of size 224x224 with 3 channels(RGB).It consists of several layers including convolutional layers, max pooling layers, bottleneck layer and skip connections.

![image](https://github.com/user-attachments/assets/4cd33944-6cec-4a1e-b24a-27dd82df0376)


# InceptionV3
InceptionV3 is a deep convolution neural network. It is the third version of the Inception architecture introduced by [Google](https://arxiv.org/abs/1512.00567) in 2015.It is also known as GoogLeNet. InceptionV3 is mainly designed for to decrease the computational cost and deep network with efficient accuracy.
InceptionV3 takes an input size of 299x299 with 3 channels. It consists of Inception Modules, Factorized Convolutions, Auxiliary classifiers and Label Smoothing.

![image](https://github.com/user-attachments/assets/0a83f410-56ee-4660-ac07-be83c67d1226)


# EfficientNetB3
EfficientNet-B3 is a deep convolutional neural network.It is the third version of EfficientNet family which was developed by [Google](https://arxiv.org/abs/1905.11946). EfficientNetB3 specifically strikes a balance between accuracy and efficiency by scaling the depth,width and resolution in a coordinated manner.
EfficientB3 takes an input size of 300x300 pixels with 3 channels(RGB). It consists of Swish Activation Function, MBConv layers, Compound Scaling, Baseline Model,Squeeze and Excitation(SE) Block.

![image](https://github.com/user-attachments/assets/ce26a340-32c9-4dcb-b9b5-a9aa2c07e0b2)

# ResNet152
It is same as the ResNet 50 but the architecture is more depth of layers stacking more layers. ResNet 152 has 152 layers and uses the same architecture as ResNet50 of bottleneck design,skip connection. ResNet 152 contains of 50 bottleneck blocks which making the network much deeper. This allows ResNet152 to learn more complex features but it also increases the computational cost.ResNet152 also takes an input size
of 224x224 pixels with 3 channels.

![image](https://github.com/user-attachments/assets/2d15a6bb-a462-44c1-a777-cedaf3f3f1af)



# DenseNet161
[DenseNet161](https://arxiv.org/abs/1608.06993) is a deep convolutional neural network. Unlike traditional networks in which each layer is connected to output layer but in DenseNet every layer is connected every other layers of upcoming layers this also known feed-forward fashion. The architecture guarantees that each layer has a direct access to the gradients from the loss function and the original input signal, improving gradient descent and making network easier to learn for training.

![image](https://github.com/user-attachments/assets/4bdb76b1-6ec7-4e1f-b6f1-ed04c763e058)


# Results
| Model Used     | Accuracy |
| -------------- | -------- |
| ResNet50       | 61.22%   |
| EfficientNetB3 | 66.71%   |
| InceptionV3    | 64.91%   |
| DenseNet161    | 60.31%   |
| ResNet152      | 61.31%   |

# Conclusions
I observed that there are limitations with deeper networks with DenseNet161 and ResNet152. These models exhibited the signs of overfitting, where the training accuracy continued to rise but the validation accuracy is stabilized for long time. This indicates that the models with more depth layers become more prone to memorizing the training data rather than generalizing to new, unseen examples.
