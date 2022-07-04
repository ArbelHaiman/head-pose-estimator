
import os
import time
import shutil
from augmentation_utils import *
import constants

# This file creates a csv file with all the images and pose vectors, which already had facial landmarks inside matlab matrices

# max number of images to extract from the specified file
number_of_images = 20000

# directory to write to
img_dir = 'database_HELEN3'

# csv to write to
csv_path = 'pose_vectors_HELEN3.csv'
current_database = 'HELEN3'

# index to start database from
number_to_start = 0


def extract_landmarks(landmarks_path):
    """
    A function to extract landmarks and bbox from the .mat files in the images directory.
    :param landmarks_path: The path to the .mat file which contains the landmarks.
    :return: The bbox and landmarks extracted.
    """
    # extract the landmarks
    landmarks_matrix = sio.loadmat(landmarks_path)['pt2d'].transpose()
    
    # extract a bounding box
    x_min, y_min = np.amin(landmarks_matrix, axis=0)
    x_max, y_max = np.amax(landmarks_matrix, axis=0)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox, landmarks_matrix


def extract_lm_and_images_from_matlab_files():
    """
    A function to extract images, bboxes and landmarks from all regular images in directory.
    :return: A list of images, bboxes and landmarks for all regular images.
    """
    rootdir = current_database
    image_list = []
    lmks_list = []
    bbox_list = []
    counter = 0
    # going through all the files
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if counter >= number_of_images:
                return image_list, bbox_list, lmks_list
            
            # adding current landmarks
            if file.endswith('_0.mat'):
                bbox, current_landmarks = extract_landmarks(os.path.join(subdir, file))
                bbox_list.append(bbox)
                lmks_list.append(current_landmarks)
                counter += 1
            
            # adding current image
            elif file.endswith('_0.jpg'):
                image_list.append(cv2.imread(os.path.join(subdir, file)))

    return image_list, bbox_list, lmks_list


# creating the database images directory
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.makedirs(img_dir)

# csv to write pose vectors to
df = pd.DataFrame(columns={'image_name': [], 'rx': [], 'ry': [], 'rz': [], 'tx': [], 'ty': [], 'tz': []})
df.to_csv(csv_path, index=False)

pic_list = []
landmarks_list = []
bbox_list = []
final_images_list = []
final_lmks_list = []
DoF6_vec_list = []

database_counter = number_to_start

# generic model of face
model_path = 'model3D_aug_-00_00_01.mat'

# extracting generic model and camera matrix
camera_matrix, model_points = extract_model_and_camera_matrix()
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

time1 = time.time()

# extracting images, bboxes and landmarks
pic_list, bbox_list, landmarks_list = extract_lm_and_images_from_matlab_files()

# creating the database
# augmenting each image, to create a larger and more versatile database for the training
database_counter = augment_and_create_database(pic_list, landmarks_list, database_counter, img_dir, csv_path)

# measuring the time it took to create the database
time2 = time.time()
print('process with ' + str(number_of_images) + ' images took ' + str(math.floor((time2 - time1) / num_seconds_in_minute)) +
      ' minutes, and ' + str((time2 - time1) % num_seconds_in_minute) + ' seconds')
print('total number of images in final dataset: ' + str(database_counter - number_to_start))

# showing the results of images with pose vectors drawn on
data = np.genfromtxt(csv_path, delimiter=',')[1:, 1:]

for i in range(0, database_counter-number_to_start):
    img = draw_axis(cv2.imread(img_dir + '//image_' + str(i + number_to_start) + '.jpg'), data[i][:3],
                    data[i][3:], camera_matrix)
    cv2.imshow('example', img)
    cv2.waitKey()

