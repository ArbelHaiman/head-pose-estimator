## head-pose
# head pose estimator
This project was a final project of a computer vision course during my Master's degree.
The goal was to create an algorithm that can  estimate the location and rotation(meaning, the extrinsic parameters)
of a face, in a face image.
The project is divided into 2 sub-mission:
1. Creating a dataset for the problem, consists of face images(The samples), and the translation and rotation vectors(The labels).
2. Creating a model which learns from the dataset, and able to predict the translation and rotation of a new face image.

## Database creation
# Some technicals:
For this calculation, we need 3D-2D correspondences. The matrix which represent the transformation we try to calculate
is a 3 by 4 matrix, so we need to estimate a total of 12 parameters. each correspondence gives 2 equations, meaning we need at least
6 correspondences for this calculation. If we use more than that, then the result will be more accurate, because we can use optimization 
methods such as RANSAC.
I used the 68 facial landmarks model for caculating the transformation of the face in an image, meaning 68 3D-2D correspondences
for each image.

The 2D points of each of the 68 facial landmarks can be found in an image using dlib 68 facial landmarks detector.
There are also databases, such as HELEN, AFW and IBUG, which are manually labeled(meaning, they consists of the
locations of the landmarks already).
In this project I used These databases as well as new images labeled by dlib's detector.
There is a critical disadvantage using only dlib's detector, since it only recognizes faces in easy angles.
This results in a non-versatile database.
The 68 3D facial landmarks are taken from a 3D generic face model(they are the same for all images).

After we have 3D-2D correspondences for each image, we can calculate the transformation, meaning its label, using 
the solvepnp method, available in opencv library. The method returns the translation and rotation vectors, which are going to
be our labels.

Now we apply an augmentation process, in which we make our database bigger and more versatile, by taking existed images and labels, and applying many kinds of different transformations such as rotation in image plane, changing light conditions, mirroring, cropping, and more. We apply these transformations on images and on their corresponding labels, so at the end we have a bigger and more versatile dataset for the system to learn from.

After creating the database, i used 2 CNNs(same architecture) for predicting the labels - ine for rotation vector and one for translation vector.

I also added visualizing tools, which draw the 3 axis on the image, in the directions of the rotation vector.
