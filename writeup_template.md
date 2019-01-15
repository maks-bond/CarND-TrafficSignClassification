# **Traffic Sign Recognition** 

Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/1.jpg "Class distributions"
[image2]: ./writeup_imgs/RGB.jpg "Original RGB"
[image3]: ./writeup_imgs/Y.jpg "Y channel"
[image4]: ./TestSigns/1.png "Traffic Sign 1"
[image5]: ./TestSigns/2.png "Traffic Sign 2"
[image6]: ./TestSigns/3.png "Traffic Sign 3"
[image7]: ./TestSigns/4.png "Traffic Sign 4"
[image8]: ./TestSigns/5.png "Traffic Sign 5"
[image9]: ./writeup_imgs/conv1.jpg "Conv 1"
[image10]: ./writeup_imgs/activ1.jpg "Activ 1"
[image11]: ./writeup_imgs/pool1.jpg "Pool 1"

### Data Set Summary & Exploration

#### 1. Basic summary of the dataset

I used numpy and basic python functionality to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a plot showing the number of samples of each traffic sign type in the training, validation and test datasets:  

![alt text][image1]


It is visible that there are some classes which have more samples than others however overall distributions in the datasets look similar (whenever one class has more samples than other class in training data we can see the same relationship in validation and test datasets).

### Design and Test a Model Architecture

#### 1. Data preprocessing

Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, the image is converted to YCrCb color space. As a second step Y channel is extracted and the image is represented as gray scale image with one color channel.  
Such decision was made experimentally. Originally I was just using all thress channels of RGB color space and then tried Y channel of YCrCb color space like it was done in this paper: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf and the network performance was satisfactory using Y channel. Potentially I could try other color spaces and other channels but the performance using Y channel is good enough.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image by subtracting mean value of all pixels and dividing by standard deviation to bring all image pixel values to the same scale with 0-mean and standard deviation of 1.

# TODO:
I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:



The difference between the original data set and the augmented data set is the following ... 


#### 2. Model architecture

Final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x100 	|
| RELU					|												|
| DROPOUT               | 0.8 keep probability                          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x100 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x150  |
| RELU					|												|
| DROPOUT               | 0.8 keep probability                          |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 8x8x200    |
| RELU					|												|
| DROPOUT               | 0.8 keep probability                          |
| Max pooling	      	| 2x2 stride,  outputs 4x4x200  				|
| Fully connected		| Input 3200 after flattening. Output 200       |
| RELU					|												|
| DROPOUT               | 0.8 keep probability                          |
| Fully connected		| Input 200 after flattening. Output 84         |
| RELU					|												|
| DROPOUT               | 0.8 keep probability                          |
| Fully connected		| Input 84 after flattening. Output 43          |
 


#### 3. Training hyperparameters

Adam Optimizer is used for optimization and weight updates.  
Batch Size: 100  
Learning Rate: 0.001  
Number of Epochs: 20  
Dropout keep probability 0.8 for training and 1.0 for validation and testing.

#### 4. Accuracy results and experiments description
Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 97.9%
* test set accuracy of 95.3%

The final model has been obtained after sets of experments. Below is the brief description of every experiment setup, accuracy and transition to next experiment:
1. I started with LeNet model. The reason is that it classified numbers well and there was a chance that it would be able to classify german traffic signs as well which are 32x32 images as well. The only difference was that in LeNet the images have been grayscale but german traffic signs have 3 channels by default. So the model was updated to accept 32x32x3 images.  
Such model gave not very good results. I lost exact acuracy numbers. But validation and test accuracy have been around 70%.
I tried running LeNet on grayscale image (by converting RGB to grayscale first) which didn't produce any good results either.  
Also it is worth noting that originally I had very simple normalization by simply substracting 128 and dividing by 255.
2. After reading this paper http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf I realized that it might be worth trying enhancing number of filters in convultional layers.
This model was exactly the same as previous one but first conv layer now had 100 filters and second one 200.
Validation set accuracy was 97.4%. Test set accuracy was 84.5%.
It is worth noting here that originally I downloaded just training and test datasets and was splitting training dataset using sklearn module to obtaine validation dataset. It is hard to explain why but here I observed much better performance on validation data than on test data.
As a result I was not satisfied with such model.
3. Converted an image to YCrCb first and then applying same model as above produced validation accracy of 90.1% and test set accuracy of 69.2%.
4. Same model as above but used only Y channel of YCrCb.
Validation accuracy: 95.8%  
Test accuracy: 84.8%  
5. Added third convolutional layer. Updatd number of filters to be 100, 150, 200 for first, second and third conv layers respectively like in the final model.
Validation accuracy: 97.5%  
Test accuracy: 87.7%
6. Added dropout layers like in the final model.
Validation accuracy: 98.5%  
Test accuracy: 90.9%
7. Improved image normalization to subtract mean and divide by stddev. Also started using the dataset provided by the project description. This is the final model.
Validation accuracy: 97.9%  
Test accuracy: 95.3%


Some thoughts on the final model:  
1. Convolutional layer needs to have more filters which allows to capture different properties of an image. Since we have 43 classes we need to be able to capture diverse set of features. And having 6 and 16 filters like in LeNet is probably not enough to capture different features.
2. Dropout layer allows to train the network in more robust way by creating redundant connections which are activated under similar circumstances. That is acheived by randomly turning off some percentage of connections during training. Dropout layers should also reduce overfitting.
3. Ideally if I would be building production-ready model, I would experiment much more with model parameters and potentially find simplified model which still produces great results with the purpose of reducing training time and reducing predcition time.

### Test a Model on New Images

#### 1. Images discussion

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Images are reshaped to 32x32x3 and preprocessed using same logic as trainign/validation/test images.

Such images should be simple to classify in general however they have interesting properties like watermarks of websites which distribute them, background objects.

#### 2. Model performance on test images found in the web

The accuracy is 100%.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| No passing   									| 
| Road work     			| Road work 										|
| Children crossing					| Children crossing											|
| End of no passing	      		| End of no passing					 				|
| Wild animals crossing			| Wild animals crossing      							|


#### 3. Looking out softmax probabilities of each test image

In general model outputs pretty high probabilities for correct class. Which is vry good. Here is the output of top 5 probabilities for each test image:  
# TODO: Update this
For image # 1  the top 5 answers are:   
('No vehicles', 0.68922091)  
('No passing', 0.063223109)  
('Stop', 0.058636673)  
('Traffic signals', 0.051628381)  
('Priority road', 0.044560913)  
Correct answer is:  No passing  
  
For image # 2  the top 5 answers are: 
('Road work', 0.98606306)
('Turn right ahead', 0.0068569127)
('Road narrows on the right', 0.0017835007)
('Bicycles crossing', 0.0015924331)
('Keep left', 0.00073580601)
Correct answer is:  Road work

For image # 3  the top 5 answers are: 
('Children crossing', 0.99973696)
('Speed limit (20km/h)', 0.00017905852)
('Dangerous curve to the right', 3.5420413e-05)
('Go straight or right', 1.4326039e-05)
('Vehicles over 3.5 metric tons prohibited', 1.1186782e-05)
Correct answer is:  Children crossing

For image # 4  the top 5 answers are: 
('End of no passing', 1.0)
('End of all speed and passing limits', 5.0811014e-08)
('Dangerous curve to the right', 1.9846094e-09)
('End of no passing by vehicles over 3.5 metric tons', 1.5457834e-09)
('Keep right', 1.9596949e-11)
Correct answer is:  End of no passing

For image # 5  the top 5 answers are: 
('Wild animals crossing', 0.99528044)
('Speed limit (50km/h)', 0.003413972)
('Double curve', 0.00050660153)
('Speed limit (80km/h)', 0.0004538151)
('Speed limit (30km/h)', 0.00030400761)
Correct answer is:  Wild animals crossing

### Visualizing the Neural Network layers

Visualization of some trained neural netowrk layers have been performed based on activations of one test image to understand what kind of features does the network capture. Only layers before first pooling layer have been visualized as after first pooling layer the layer activations are not very representative as they encode relationship between activations of first convolutational layer (and subsequent additional layers: relu, dropout, pooling).
1. Visualization of the first convolutional layer (only 49 out of 100 filters are shown as plotting library can't visualize more):  
![alt text][image9]  

As we can see the network seem to have learned some filters which transform input image and apply some kind of 'edge detection' filters but with different properties.
2. Visualization of the relu activation layer after first convolutional layer:  
![alt text][image10]  
Activation relu layer removes a lot of pixel values (negative ones become 0) which seems to contribute to filtering which looks like real edge detection but with different gradient direction and magnitude.
3. Visualization of the max pool layer after first activation layer:  
![alt text][image11]  
Pooling layer just downsamples activations from previous layer. Preserving detected edges and features but reducing number of pixels which basically reduces amount of imfromation for subsequent layers. Bascially it allows to focus on important 'features' removing redundant data.
