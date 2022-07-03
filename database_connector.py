import os
import shutil
from augmentation_utils import *

# This file contains operations for the merging of different small databases which I created during the project.

model_path = 'model3D_aug_-00_00_01.mat'
final_dataset_dir = 'final_database'
final_dataset_csv_path = 'final_pose_estimation_DB.csv'

afw_dir = 'database_AFW'
ibug_dir = 'database_IBUG'
helen1_dir = 'database_HELEN1'
helen2_dir = 'database_HELEN2'
helen3_dir = 'database_HELEN3'

afw_pose_csv = 'pose_vectors_AFW.csv'
ibug_pose_csv = 'pose_vectors_IBUG.csv'
helen1_pose_csv = 'pose_vectors_HELEN1.csv'
helen2_pose_csv = 'pose_vectors_HELEN2.csv'
helen3_pose_csv = 'pose_vectors_HELEN3.csv'


def create_full_dataset():
    """
    A function to create a final full database for the model to learn, out of small databases.
    :return: None
    """

    # creating full dataset images directory
    dir = 'final_database'
    if os.path.exists(final_dataset_dir):
        shutil.rmtree(final_dataset_dir)
    os.makedirs(final_dataset_dir)

    # creating full data set labels csv file
    df = pd.DataFrame(columns={'image_name': [], 'rx': [], 'ry': [], 'rz': [], 'tx': [], 'ty': [], 'tz': []})
    df.to_csv(final_dataset_csv_path, index=False)

    # reading the partial datasets
    labels_afw = pd.read_csv(afw_pose_csv)
    labels_ibug = pd.read_csv(ibug_pose_csv)
    labels_helen1 = pd.read_csv(helen1_pose_csv)
    labels_helen2 = pd.read_csv(helen2_pose_csv)
    labels_helen3 = pd.read_csv(helen3_pose_csv)

    # writing them to the final dataset
    labels_afw.to_csv(final_dataset_csv_path, mode='a', header=False, index=False)
    labels_ibug.to_csv(final_dataset_csv_path, mode='a', header=False, index=False)
    labels_helen1.to_csv(final_dataset_csv_path, mode='a', header=False, index=False)
    labels_helen2.to_csv(final_dataset_csv_path, mode='a', header=False, index=False)
    labels_helen3.to_csv(final_dataset_csv_path, mode='a', header=False, index=False)

    # moving all images to the final directory
    afw_images = os.listdir(afw_dir)
    ibug_images = os.listdir(ibug_dir)
    helen1_images = os.listdir(helen1_dir)
    helen2_images = os.listdir(helen2_dir)
    helen3_images = os.listdir(helen3_dir)

    for f in afw_images:
        shutil.move(afw_dir + '//' + f, final_dataset_dir)
    for f in ibug_images:
        shutil.move(ibug_dir + '//' + f, final_dataset_dir)
    for f in helen1_images:
        shutil.move(helen1_dir + '//' + f, final_dataset_dir)
    for f in helen2_images:
        shutil.move(helen2_dir + '//' + f, final_dataset_dir)
    for f in helen3_images:
        shutil.move(helen3_dir + '//' + f, final_dataset_dir)

    # now, the final dataset is ready
    return


def check_pose_and_image(index):
    """
    A function to check that the final database is synchronized with its labels.
    :param index: The index of the image to check its pose.
    :return: None
    """
    camera_matrix, model_points = extract_model_and_camera_matrix()
    data = np.genfromtxt(final_dataset_csv_path, delimiter=',')[1:, 1:]
    img = draw_axis(cv2.imread(final_dataset_dir + '//image_' + str(index) + '.jpg'), data[index][:3],
                    data[index][3:], camera_matrix)
    cv2.imshow('gf', img)
    cv2.waitKey()


create_full_dataset()
#for ind in range(0, 200):
    #check_pose_and_image(ind * 1200)
