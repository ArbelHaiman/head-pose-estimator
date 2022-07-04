import os
import shutil
import dlib
from imutils import face_utils
import time
from augmentation_utils import *
import constants

# This file writes creates the database from images which didn't have labels.
# For these images, we compute the labels, then write the images and their labels, i.e. pose vectors, into a csv file

# The total number of images to go through
number_of_images = 11000

# The generic model we calculate the model according to.
model_path = 'model3D_aug_-00_00_01.mat'

# dlib's facial landmarks detector.
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# The final database and pose vectors to write to.
img_dir = 'database_dlib'
csv_path = 'pose_vectors_dlib.csv'

# The directory to sample images from
current_database = 'train1'

# The index to start the database from
number_to_start = 112080

def get_images_as_list():
    """
    A function to create a list of images.
    :return: a list of images.
    """
    rootdir = current_database
    image_list = []
    counter = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if counter >= number_of_images:
                return image_list
            #print(os.path.join(subdir, file))
            image_list.append(cv2.imread(os.path.join(subdir, file)))
            counter += 1

    return image_list


def rect_to_bb(rect):
    """

    :param rect:
    :return:
    """
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return [x, y, w, h]



def detect_bbox_and_lmks(image):
    """
    A function that detects the bbox and facial landmarks of a face in an image, using dlib's detector.
    :param image: The image to detect in.
    :return: The bbox and landmarks coordinates.
    """
    # detect the facial landmarks with dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    # we take only images with one face in it.
    if len(rects) != 1:
        return None, None
    rect = rects[0
                 
    # For each detected face, find the landmark.
    landmarks2d = predictor(gray, rect)
    landmarks2d = face_utils.shape_to_np(landmarks2d)
                 
    # converting rect to bbox
    rect = rect_to_bb(rect)
    return rect, landmarks2d


def get_list_with_bboxes_and_lmks(images_list):
    """
    A function to create a list of bboxes and landmarks for the images.
    :param images_list: The images to find bbox and landmarks in.
    :return: imges list, bbox list and landmarks list.
    """
    images = []
    landmarks_list = []
    bbox_list = []
    counter = 0
    
    # go through all the images, compute the lmks and bbox for each images and append to list
    for ind in range(0, len(images_list)):
        if counter > number_of_images:
            break
        bbox, landmarks = detect_bbox_and_lmks(images_list[ind])
        if bbox is not None:
            images.append(images_list[ind])
            landmarks_list.append(landmarks)
            bbox_list.append(bbox)
            counter += 1
            print(counter)
        else:
            print('couldn\'t find landmarks')
    
    # get final lists as numpy arrays
    images = np.array(images)
    landmarks_list = np.array(landmarks_list)
    bbox_list = np.array(bbox_list)

    return images, bbox_list, landmarks_list

# a counter to start the database from
database_counter = number_to_start

# extracting the generic model and the camera matrix
camera_matrix, model_points = extract_model_and_camera_matrix()
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# creating the database images directory
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.makedirs(img_dir)

# creating the pose csv
df = pd.DataFrame(columns={'image_name': [], 'rx': [], 'ry': [], 'rz': [], 'tx': [], 'ty': [], 'tz': []})
df.to_csv(csv_path, index=False)

time1 = time.time()
images = get_images_as_list()

# this is a filtered list, to not contain images in which landmarks haven't been found
good_images_list = []
bbox_list = []
lmks_list = []
final_images_list = []
final_lmks_list = []

# creating the database
good_images_list, bbox_list, lmks_list = get_list_with_bboxes_and_lmks(images)
DoF6_vec_list = []
database_counter = augment_and_create_database(good_images_list, lmks_list, database_counter, img_dir, csv_path)

# printing the time the process took
time2 = time.time()
print('process with ' + str(number_of_images) + ' took ' + str(math.floor((time2-time1) / num_seconds_in_minute)) + ' minutes, and ' +
      str((time2 - time1) % num_seconds_in_minute) + ' seconds')
print('total number of images in final dataset: ' + str(database_counter-number_to_start))

data = np.genfromtxt(csv_path, delimiter=',')[1:, 1:]

# showing the database images with pose vectors
for i in range(0, database_counter-number_to_start):
    img = draw_axis(cv2.imread(img_dir + '//image_' + str(i + number_to_start) + '.jpg'), data[i][:3],
                    data[i][3:], camera_matrix)
    cv2.imshow('example', img)
    cv2.waitKey()






