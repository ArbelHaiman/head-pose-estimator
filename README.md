# Head Pose Estimator
This project was a final project of a computer vision course during my Computer Science Master's degree.
Our mission to create an algorithm for estimating the translation and rotation(meaning, the extrinsic parameters)
of a face, in a face image.
The project is divided into 2 sub-mission:
1. Creating a dataset for the problem, consists of face images(The samples), and the translation and rotation vectors(The labels).
2. Creating a model which learns from the dataset, and able to predict the translation and rotation of a new face image.

## Project Process and Database Creation
### Theoretical Explanations
For calculating the 6 parameters of translation and rotation vectors, we need 3D-2D correspondences. We take the 3D point from a generic model of a face, and we take the 2D points by locating same points in the image, by an algorithm or manually.
After we have the pairs of correspondences, we can estimate the parameters.
The matrix which represent the transformation we try to calculate
is a 3 by 4 matrix, so we need to estimate a total of 12 parameters. each correspondence gives 2 equations, meaning we need at least
6 correspondences for this calculation. If we use more than that, then the result will be more accurate, because we can use optimization 
methods such as RANSAC.
I used the 68 facial landmarks model for caculating the transformation of the face in an image, meaning 68 3D-2D correspondences
for each image.

### Database Creation
The 2D points of each of the 68 facial landmarks can be found in an image using dlib 68 facial landmarks detector.
There are also databases, such as HELEN, AFW and IBUG, which are manually labeled(i.e., they consists of the
locations of the landmarks already).
In this project I used these databases as well as new images labeled by dlib's detector.
It is a critical disadvantage using only dlib's detector, since it only recognizes faces in 'easy' angles.
This results in a non-versatile database.
The 68 3D facial landmarks are taken from a 3D generic face model(they are the same for all images).

After we have 3D-2D correspondences for each image, we can calculate the transformation, meaning its label, using 
the solvepnp function, available in opencv library. The method returns the translation and rotation vectors, which are going to
be our labels.

### Augmentation Process
After we have all images and their corresponding 68 landmarks, we apply an augmentation process, in which we make our database larger and more versatile, by taking existed images and labels, and applying different kinds of transformations such as rotation in image plane, changing light conditions, mirroring, cropping, and more. We apply these transformations on images and on their corresponding labels, so at the end we have a larger and more versatile dataset for the system to learn from.

### Learning Process
After creating the database and augmenting the images, I used 2 CNNs(same architecture) for predicting the labels - one for rotation vector and one for translation vector. By trial and error, i it found out to be the best approach to separate the prediction of these two vectors, instead of predicting them using the same network, simultaneously. 

### More Tools and Technicals
I also added visualizing tools, which draw the 3D rotation vector in the face's center, which is given by the translation vector.
I used an a laptop with intel i7 9th generation and an NVIDIA RTX 2060 GPU for the training process, which took about 20 hours total.

## Files Summary
#### 1. pre_database_creator.py:
   Creates the database (images and 6 DoF vectors) from the images which already have the 68 facial landmarks.
#### 2. database_creator.py:
   Creates the database (images and 6 DoF vectors) from the images which do not have the 68 facial landmarks (it finds them).
#### 3. database_connector.py:
   Creates the combined database, composed of all images and their labels, including augmentations.
#### 4. model_creator.py:
   Creates the model architecture, and trains it.
#### 5. model_demonstrator.py:
   Applies prediction on test set, and show the images with the predictions.
#### 6. augmentation_utils.py:
   Contains all the functions to apply the augmentation operations.
#### 7. constants.py:
   Contains constants to be used in all other files.
#### 8. images directory:
   Contains images showing different stages in the process.
#### 9. model3D_aug_-00_00_01.mat:
   The 3D landmarks model, which is used for the computation of the head pose.


### An Example Of The Processing
#### 1. We take a face image
![alt text](https://github.com/ArbelHaiman/head-pose-estimator/blob/master/images/orginal.jpg)

#### 2. We find its 68 facial landmarks (or we already have it manually labeled)

#### 3. We augment it, by creating multiple copies of the image, in which we change its parameters (lighting, size, rotation, mirroring and more)

#### 4. We compute its 6DoF vector

#### 5. We train the model, then predict on unseen images
