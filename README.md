
**Behavioral Cloning Project**

In this project,a deep learning network model is used to predict the right steering angle that lets a car drives on tracks without exitng the boundaries in a simulated emvironment.
By cloning the human driving behavior during training mode,we will let the car drive by itself in autonomus mode using the model that leverages Keras deep learning netwrok framework.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Video Reference:

You can see how the car performs on an unseen track in a simulated environment here : 

Track1: 
[![Alt text](https://img.youtube.com/vi/7K5qkB90RzI/0.jpg)](https://www.youtube.com/watch?v=7K5qkB90RzI)

Track2:
[![Alt text](https://img.youtube.com/vi/6oR3BuE_IWM&t=33s/0.jpg)](https://www.youtube.com/watch?v=6oR3BuE_IWM&t=33s)


[//]: # (Image References)
[image1]: ./examples/center_2016_12_01_13_30_48_287.jpg 
[image2]: ./examples/Cropped-Image-Center.png
[image3]: ./examples/Cropped-and-Resized-Image-Center.png
[image4]: ./examples/nvidia_end_to_end_learning.jpg
 
Normal Center Image:
![alt text][image1]

Cropped Image: 
![alt text][image2]

Cropped and Resized Center Image 
![alt text][image3]

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the both tracks by executing python drive.py model.h5


####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

The actual architecture was borrowed from the Nvidia's "End to End Learning for Self-Driving Cars" paper

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64. 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 228). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 256-269). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator on both tracks and ensuring that the vehicle could stay on those tracks.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Using the Udacity's provided data that is labelled with road type- center,left and right along with steeting angles,I have been able to successfully train the deep learning netwrok model 
that let the car drives autonomusly on both tracks.The dataset contains a driving log which has file path of the images taken by three cameras, steering, throttle, brake and speed. There are 8,036 records. 
Each record has three corresponding images, center, left and right. And therefore, there are 24,108 images. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The data is not suffcient so we have augmented data by horizontal flipping.I have flipped the image around the Y axis and changed the sign of the corresponding steering angle.
I also have cropped the hood and most of the sky and scenery out of the image. Also,reshaped images to 32x32 to allow model to train on low intensive reources.
I have aso normalize pixels values, dividing them by 255.

The overall strategy for deriving a model architecture is to let the model learn useful road features with only steering angles.I will then train the weights of network to minimize the mean sqaured error.

The images are preprocessed with cropping, resizing. The number of the data was increased from 8,036 (original number) to 50,016. 

My first step is to use a convolution neural network model similar to the Nvidia's "End to End Learning for Self-Driving Cars". I thought this model might be appropriate because the model was used to map raw pixels from a single
front-facing camera directly to steering commands.That's what mentioned in the Nvidia's paper (./examples/nvidia_end_to_end_learning.pdf),"We trained a convolutional neural network (CNN) to map raw pixels from a single
front-facing camera directly to steering commands.The end-to-end approach proved surprisingly powerful. With minimum training data from humans the system learns to drive in traffic on local roads with or without lane markings and on highways." Since the data are being used to train my model have no lane markings,and limited training data and assumed its gathered from a signle front facing camera in simulated environement, I thought this model would be a perfect solution to implement to train the model and let the car drive in autonomus mode.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Using the data as it was, I found that my car was exiting boundaries. I noticed the the model was overfitting. 

To combat the overfitting, I modified the model so that it includes dropout and set the learning rate to as low as 1e-5.

Then I found no difference between the validation loss and loss result at the end of 50 epochs for both the training and the validation set. The two loss values are being computed on two distinct sets of samples - training and validation.
It might be due to the use a very small subset of training as a validation samples. It might be due to the loss for the training data is the loss for the last batch of data, while the loss for validation data is the average loss for the whole validation set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and to improve the driving behavior in these cases, I augmentated data with image flipping, add biases +/-0.25 to left and right of steering angles
to resposition the car back to center of the track.

I have trained the weights of the network to minimize the mean squared error between the steering command output by the network and the command of either the human driver, or the adjusted steering
command for off-center images like that was used in the Nvidia's  "End to End Learning for Self-Driving Cars" soultion.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final Keras model architecture (model.py lines 224-274) consisted of a convolution neural network with the following layers and layer sizes.
The CNN architecture contains 5x convolutional layers, 5x fully connected layers (Dense layers 2).It also contains max pooling layer x s, flatten layer x 1 with total 329,079 trainable parameters. 
Exponential linear unit (ELU) was chosen as the activation function and applied in all convolutional and Dense layers.
In order to prevent overfitting, dropout with 40% is applied on all of the fully connected layers.
I used Adam optimizer with a learning rate of 1e-5 and used Mean squared error (mse) to train the model. Batch size was set to 32 and number of epochs to 50. 

#### Model Summary
    ____________________________________________________________________________________________________

	Layer (type)                     Output Shape          Param #     Connected to                     
	lambda_1 (Lambda)                (None, 32, 32, 3)     0           lambda_input_1[0][0]             
	convolution2d_1 (Convolution2D)  (None, 16, 16, 24)    1824        lambda_1[0][0]                   
	elu_1 (ELU)                      (None, 16, 16, 24)    0           convolution2d_1[0][0]            
	maxpooling2d_1 (MaxPooling2D)    (None, 15, 15, 24)    0           elu_1[0][0]                      
	convolution2d_2 (Convolution2D)  (None, 8, 8, 36)      21636       maxpooling2d_1[0][0]             
	elu_2 (ELU)                      (None, 8, 8, 36)      0           convolution2d_2[0][0]            
	maxpooling2d_2 (MaxPooling2D)    (None, 7, 7, 36)      0           elu_2[0][0]                      
	convolution2d_3 (Convolution2D)  (None, 4, 4, 48)      43248       maxpooling2d_2[0][0]             
	elu_3 (ELU)                      (None, 4, 4, 48)      0           convolution2d_3[0][0]            
	maxpooling2d_3 (MaxPooling2D)    (None, 3, 3, 48)      0           elu_3[0][0]                      
	convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       maxpooling2d_3[0][0]             
	elu_4 (ELU)                      (None, 3, 3, 64)      0           convolution2d_4[0][0]            
	maxpooling2d_4 (MaxPooling2D)    (None, 2, 2, 64)      0           elu_4[0][0]                      
	convolution2d_5 (Convolution2D)  (None, 2, 2, 64)      36928       maxpooling2d_4[0][0]             
	elu_5 (ELU)                      (None, 2, 2, 64)      0           convolution2d_5[0][0]            
	maxpooling2d_5 (MaxPooling2D)    (None, 1, 1, 64)      0           elu_5[0][0]                      
	flatten_1 (Flatten)              (None, 64)            0           maxpooling2d_5[0][0]             
	dense_1 (Dense)                  (None, 1164)          75660       flatten_1[0][0]                  
	elu_6 (ELU)                      (None, 1164)          0           dense_1[0][0]                    
	dropout_1 (Dropout)              (None, 1164)          0           elu_6[0][0]                      
	dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
	elu_7 (ELU)                      (None, 100)           0           dense_2[0][0]                    
	dropout_2 (Dropout)              (None, 100)           0           elu_7[0][0]                      
	dense_3 (Dense)                  (None, 50)            5050        dropout_2[0][0]                  
	elu_8 (ELU)                      (None, 50)            0           dense_3[0][0]                    
	dropout_3 (Dropout)              (None, 50)            0           elu_8[0][0]                      
	dense_4 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
	elu_9 (ELU)                      (None, 10)            0           dense_4[0][0]                    
	dropout_4 (Dropout)              (None, 10)            0           elu_9[0][0]                      
	dense_5 (Dense)                  (None, 1)             11          dropout_4[0][0]                  
	Total params: 329,079
	Trainable params: 329,079
	Non-trainable params: 0


Here is a visualisation of model architecture 

Nvidia end to end learning CNN Architecture:

![alt text][image4]


####3. Creation of the Training Set & Training Process

I have used 85% for training and 15% for data validation. I have also used a random seed value as for the purpose of data reproducibilty during train-test split.
I also shuffled data set beforehand.I used fit generator to get more data points. To augment the data sat, I also flipped images and angles thinking that this would help model to learn better driving straight and recovering to center.

After the collection process, I have 50016 number of data points.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 50 as evidenced by below training result. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is the result of the trained model.
____________________________________________________________________________________________________
    Epoch 1/50
    50016/50016 [==============================] - 50s - loss: 0.0449 - val_loss: 0.0290
    Epoch 2/50
    50016/50016 [==============================] - 49s - loss: 0.0332 - val_loss: 0.0272
    Epoch 3/50
    50016/50016 [==============================] - 50s - loss: 0.0301 - val_loss: 0.0245
    Epoch 4/50
    50016/50016 [==============================] - 49s - loss: 0.0289 - val_loss: 0.0251
    Epoch 5/50
    50016/50016 [==============================] - 49s - loss: 0.0277 - val_loss: 0.0226
    Epoch 6/50
    50016/50016 [==============================] - 50s - loss: 0.0267 - val_loss: 0.0224
    Epoch 7/50
    50016/50016 [==============================] - 49s - loss: 0.0259 - val_loss: 0.0221
    Epoch 8/50
    50016/50016 [==============================] - 50s - loss: 0.0254 - val_loss: 0.0213
    Epoch 9/50
    50016/50016 [==============================] - 50s - loss: 0.0247 - val_loss: 0.0213
    Epoch 10/50
    50016/50016 [==============================] - 50s - loss: 0.0246 - val_loss: 0.0196
    Epoch 11/50
    50016/50016 [==============================] - 50s - loss: 0.0234 - val_loss: 0.0202
    Epoch 12/50
    50016/50016 [==============================] - 50s - loss: 0.0227 - val_loss: 0.0201
    Epoch 13/50
    50016/50016 [==============================] - 50s - loss: 0.0224 - val_loss: 0.0194
    Epoch 14/50
    50016/50016 [==============================] - 50s - loss: 0.0226 - val_loss: 0.0183
    Epoch 15/50
    50016/50016 [==============================] - 50s - loss: 0.0220 - val_loss: 0.0182
    Epoch 16/50
    50016/50016 [==============================] - 50s - loss: 0.0213 - val_loss: 0.0180
    Epoch 17/50
    50016/50016 [==============================] - 50s - loss: 0.0210 - val_loss: 0.0177
    Epoch 18/50
    50016/50016 [==============================] - 50s - loss: 0.0208 - val_loss: 0.0175
    Epoch 19/50
    50016/50016 [==============================] - 50s - loss: 0.0207 - val_loss: 0.0171
    Epoch 20/50
    50016/50016 [==============================] - 50s - loss: 0.0201 - val_loss: 0.0176
    Epoch 21/50
    50016/50016 [==============================] - 50s - loss: 0.0194 - val_loss: 0.0169
    Epoch 22/50
    50016/50016 [==============================] - 50s - loss: 0.0194 - val_loss: 0.0169
    Epoch 23/50
    50016/50016 [==============================] - 50s - loss: 0.0195 - val_loss: 0.0166
    Epoch 24/50
    50016/50016 [==============================] - 50s - loss: 0.0192 - val_loss: 0.0166
    Epoch 25/50
    50016/50016 [==============================] - 50s - loss: 0.0188 - val_loss: 0.0165
    Epoch 26/50
    50016/50016 [==============================] - 50s - loss: 0.0189 - val_loss: 0.0171
    Epoch 27/50
    50016/50016 [==============================] - 51s - loss: 0.0186 - val_loss: 0.0162
    Epoch 28/50
    50016/50016 [==============================] - 51s - loss: 0.0185 - val_loss: 0.0159
    Epoch 29/50
    50016/50016 [==============================] - 50s - loss: 0.0181 - val_loss: 0.0167
    Epoch 30/50
    50016/50016 [==============================] - 50s - loss: 0.0180 - val_loss: 0.0166
    Epoch 31/50
    50016/50016 [==============================] - 50s - loss: 0.0176 - val_loss: 0.0157
    Epoch 32/50
    50016/50016 [==============================] - 50s - loss: 0.0175 - val_loss: 0.0159
    Epoch 33/50
    50016/50016 [==============================] - 50s - loss: 0.0171 - val_loss: 0.0156
    Epoch 34/50
    50016/50016 [==============================] - 50s - loss: 0.0169 - val_loss: 0.0155
    Epoch 35/50
    50016/50016 [==============================] - 51s - loss: 0.0168 - val_loss: 0.0155
    Epoch 36/50
    50016/50016 [==============================] - 51s - loss: 0.0170 - val_loss: 0.0156
    Epoch 37/50
    50016/50016 [==============================] - 51s - loss: 0.0170 - val_loss: 0.0150
    Epoch 38/50
    50016/50016 [==============================] - 51s - loss: 0.0167 - val_loss: 0.0153
    Epoch 39/50
    50016/50016 [==============================] - 53s - loss: 0.0161 - val_loss: 0.0156
    Epoch 40/50
    50016/50016 [==============================] - 53s - loss: 0.0166 - val_loss: 0.0154
    Epoch 41/50
    50016/50016 [==============================] - 53s - loss: 0.0160 - val_loss: 0.0149
    Epoch 42/50
    50016/50016 [==============================] - 55s - loss: 0.0161 - val_loss: 0.0154
    Epoch 43/50
    50016/50016 [==============================] - 56s - loss: 0.0157 - val_loss: 0.0154
    Epoch 44/50
    50016/50016 [==============================] - 55s - loss: 0.0155 - val_loss: 0.0150
    Epoch 45/50
    50016/50016 [==============================] - 55s - loss: 0.0156 - val_loss: 0.0147
    Epoch 46/50
    50016/50016 [==============================] - 55s - loss: 0.0153 - val_loss: 0.0152
    Epoch 47/50
    50016/50016 [==============================] - 55s - loss: 0.0153 - val_loss: 0.0149
    Epoch 48/50
    50016/50016 [==============================] - 50s - loss: 0.0153 - val_loss: 0.0155
    Epoch 49/50
    50016/50016 [==============================] - 50s - loss: 0.0150 - val_loss: 0.0146
    Epoch 50/50
    50016/50016 [==============================] - 51s - loss: 0.0148 - val_loss: 0.0148