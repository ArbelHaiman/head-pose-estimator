import cv2
import numpy as np
import scipy.io as sio
import pandas as pd
import math

'''
This file contains all the necessary actions on images for the augmentation process of the dataset.

'''

# The generic face model we project to.
model_path = 'model3D_aug_-00_00_01.mat'

# The swapping facial landmarks dictionary for the flipping stage.
swap_dict = {0: 16, 1: 15, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10, 7: 9, 8: 8, 17: 26, 18: 25, 19: 24, 20: 23, 21: 22,
             27: 27, 28: 28, 29: 29, 30: 30, 31: 35, 32: 34, 33: 33, 36: 45, 37: 44, 38: 43, 39: 42, 40: 47, 41: 46,
             48: 54, 49: 53, 50: 52, 51: 51, 55: 59, 56: 58, 57: 57, 60: 64, 61: 63, 62: 62, 65: 67}

number_of_landmarks = 68
normalization_factor = 255.0
cnn_input_size = 150

def show_image_with_bbox_and_lmks(image, bbox, lmks):
    """
    This method demonstrates the image with the bounding box and and the 68 facial landmarks,
    and it is used for validation of augmentation operations.
    :param image: The image to demonstrate.
    :param bbox: The bounding box coordinates of the image.
    :param lmks: A list of the 68 landmarks.
    :return: None
    """
    draw_image = np.copy(image)
    
    # drawing the facial landmarks
    for ind in range(0, len(lmks)):
        cv2.circle(draw_image, (int(round(lmks[ind][0])), int(round(lmks[ind][1]))), 2, (0, 255, 0), -1)
        
    # drawing the bounding box
    cv2.rectangle(draw_image, (int(round(bbox[0])), int(round(bbox[1]))),
                  (int(round(bbox[0] + bbox[2])), int(round(bbox[1] + bbox[3]))), (0, 255, 255), 1)
    
    # showing the image
    cv2.imshow("Output", draw_image)
    cv2.waitKey()
    
    return


def expand_bbox(image, bbox, delta_coeff):
    """
    A function to expand the bounding box of the face.
    :param image: The image to operate on.
    :param bbox: The original bounding box.
    :param delta_coeff: A number between 0 and 1, which is used to set the expansion rate of the bbox.
    :return: The coordinates of the expanded bbox.
    """
    # computing the width delta and the height delta
    # first, we compute the offset of the bbox from both sides of the image, then take the delta_coeff percentage of the minimum
    # Hence, we end up with a bbox which is expanded 40% of the original bbox minimal offset in the image, to each side
    delta_w = delta_coeff * min(image.shape[1] - (bbox[0] + bbox[2]), bbox[0])
    delta_h = delta_coeff * min(image.shape[0] - (bbox[1] + bbox[3]), bbox[1])
    expanded_bbox = [int(round(bbox[0] - delta_w)), int(round(bbox[1] - delta_h)),
                     int(round(bbox[2] + 2 * delta_w)), int(round(bbox[3] + 2 * delta_h))]

    return expanded_bbox


def translate_bbox(image, bbox):
    """
    A function to move the bounding box, for changing a little the location of the face in the bbox.
    :param image: The image to operate on.
    :param bbox: The original bbox.
    :return: The coordinates of the new bbox.
    """
    # get a random number for translation
    s = np.random.uniform(-0.1, 0.1, 2)
    new_bbox = [0, 0, 0, 0]
    
    # compute the translated bbox, in which we change the left bottom corner, and leave the width and height unchanged
    new_bbox = [bbox[0] + int(round(s[0] * bbox[2])), bbox[1] + int(round(s[1] * bbox[3])), bbox[2], bbox[3]]
    
    # if the translation ended up out of the image dimensions, we cut it at the edge of the image
    if new_bbox[0] + new_bbox[2] > image.shape[1]:
        new_bbox[2] = image.shape[1] - new_bbox[0]
    if new_bbox[1] + new_bbox[3] > image.shape[0]:
        new_bbox[3] = image.shape[0] - new_bbox[1]

    return new_bbox


def scale_image_and_lms(image, bbox, lmks):
    """
    A function to scale the image and the bbox and the facial landmarks.
    :param image: The image to operate on.
    :param bbox: The original bbox.
    :param lmks: The original landmarks.
    :return: A tuple consists of the scaled image, the scaled bbox, and the scaled landmarks.
    """
    # scaling factor
    s = 0.75
    
    # scaling the image, by the same factor along both axes 
    scaled_image = cv2.resize(image, (0, 0), fx=s, fy=s)
    
    # scaling the bbox
    scaled_bbox = [int(round(b * s)) for b in bbox]
    
    # scaling the landmarks
    scaled_lmks = (s * lmks).astype(int)

    return scaled_image, scaled_bbox, scaled_lmks


def rotate_image_and_lmks(image, bbox, lmks):
    """
    A function to rotate the image and the facial landmarks.
    :param image: The image to rotate.
    :param bbox: The original bbox.
    :param lmks: The original landmarks.
    :return: A tuple consists of the rotated image, the new bbox and the new landmarks.
    """
    # getting a random rotating angle, in degrees
    angle = 30 * np.random.normal(0, 1, 1)
    
    # grab the dimensions of the image and then determine the
    # center. we need it because we rotate the image around the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos_of_angle = np.abs(M[0, 0])
    sin_of_angle = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    # The new measured width and height are the sum of projection of original width and height, on x and y axes, respectively
    nW = int((h * sin_of_angle) + (w * cos_of_angle))
    nH = int((h * cos_of_angle) + (w * sin_of_angle))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation
    rotated_image = cv2.warpAffine(np.copy(image), M, (nW, nH))
    
    # rotate the landmarks
    rotated_lmks = np.matmul(np.hstack((lmks, np.ones((number_of_landmarks, 1)))), M.transpose()).astype(int)

    # rotating top left corner
    lt_c_r = np.matmul(M, [bbox[0], bbox[1], 1]).astype(int)

    # rotating low right corner
    rl_c_r = np.matmul(M, [bbox[2] + bbox[0], bbox[3] + bbox[1], 1]).astype(int)

    # rotating the right top corner
    rt_c_r = np.matmul(M, [bbox[2] + bbox[0], bbox[1], 1]).astype(int)

    # rotating the low left corner
    ll_c_r = np.matmul(M, [bbox[0], bbox[3] + bbox[1], 1]).astype(int)
    
    # find final dimensions and position of bounding box
    x_min, y_min = min(lt_c_r[0], rl_c_r[0], rt_c_r[0], ll_c_r[0]), min(lt_c_r[1], rl_c_r[1], rt_c_r[1], ll_c_r[1])
    x_max, y_max = max(lt_c_r[0], rl_c_r[0], rt_c_r[0], ll_c_r[0]), max(lt_c_r[1], rl_c_r[1], rt_c_r[1], ll_c_r[1])

    rotated_bbox = [max(x_min, 0), max(y_min, 0), x_max - max(x_min, 0), y_max - max(y_min, 0)]

    return rotated_image, rotated_bbox, rotated_lmks


def crop_image(image, bbox, lmks):
    """
    A function to crop the image according to its bbox.
    :param image: The image to crop.
    :param bbox: The bbox.
    :param lmks: The facial landmarks.
    :return: The cropped image and the new landmarks.
    """
    c_cropped_image = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
    
    # compute the position of the landmarks after cropping,
    # which is just translation to the bottom left corner of the bbox
    c_cropped_lmks = lmks - [bbox[0], bbox[1]]
    return c_cropped_image, c_cropped_lmks


def crop_bigger_image(image, bbox, lmks):
    """
    A function to crop an image according to a little bigger area than the bbox.
    :param image: The image to crop.
    :param bbox: THe bbox.
    :param lmks: The facial landmarks.
    :return: The cropped image and landmarks.
    """
    return crop_image(image, expand_bbox(image, bbox), lmks)


def flip_image_and_lmks(image, lmks):
    """
    A function to flip (mirror) an image an its landmarks, with respect to the y axis, i.e. horizontal flip.
    :param image: The image to flip.
    :param lmks: THe original landmarks.
    :return: The flipped image and landmarks.
    """
    # flipping the image
    c_flipped_image = cv2.flip(image, 1)
    
    # flipping horizontal coordinates of landmarks
    c_flipped_landmark = [image.shape[1], 0] - lmks
    # keep vertical coordinates of landmarks
    c_flipped_landmark[:, 1] = lmks[:, 1]
    
    # adjust the enumeration of the landmarks 
    for i in swap_dict:
        c_flipped_landmark[[i, swap_dict[i]]] = c_flipped_landmark[[swap_dict[i], i]]
    return c_flipped_image, c_flipped_landmark


def make_bbox_around_lmks(image, lmks):
    """
    A function to make sure the all landmarks are in the bbox. else - we make the bbox bigger.
    :param image: The original image.
    :param lmks: The facial landmarks.
    :return: The image, The new bbox and the landmarks.
    """
    # find minimum and maximum position of landmarks
    x_min, y_min = np.amin(lmks, axis=0)
    x_max, y_max = np.amax(lmks, axis=0)
    
    # adjust the bbox to contain all landmarks
    new_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return image, new_bbox, lmks



def show_image_with_lmks(image, lmks):
    """
    A function to demonstrate an image an its facial landmarks.
    :param image: The image to show.
    :param lmks: The facial landmarks.
    :return: None
    """
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    font_scale = 0.3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 1

    draw_image = np.copy(image)
    # go through the landmarks and demonstrate them
    for i in range(0, len(lmks)):
        draw_image = cv2.putText(draw_image, str(i),
                                 (int(round(lmks[i][0])), int(round(lmks[i][1]))),
                                 font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Output", draw_image)
    cv2.waitKey()
    return


def get_DoF6_from_landmarks(lmks):
    """
    A function to calculate the face pose out of the facial landmarks.
    :param lmks: The facial landmarks.
    :return: The pose vector of the face.
    """
    # compute the translation and rotation vector of the head, based on the model
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, lmks, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #print("Rotation Vector:\n {0}".format(rotation_vector))
    #print("Translation Vector:\n {0}".format(translation_vector))
    return np.vstack((rotation_vector, translation_vector))


def draw_axis(img, R, t, K):
    """
    A function to show the face pose in the image.
    :param img: The image to show pose in.
    :param R: The rotation matrix or vector.
    :param t: The translation vector.
    :param K: The camera matrix.
    :return: The image with pose drawn on.
    """

    points = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, -50], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, R, t, K.astype(float), (0, 0, 0, 0))
    axisPoints = axisPoints.astype(int)
    img_d = np.copy(img)
    img_d = cv2.line(img_d, tuple(axisPoints[3][0]), tuple(axisPoints[0][0]), (255, 0, 0), 3)
    img_d = cv2.line(img_d, tuple(axisPoints[3][0]), tuple(axisPoints[1][0]), (0, 255, 0), 3)
    img_d = cv2.line(img_d, tuple(axisPoints[3][0]), tuple(axisPoints[2][0]), (0, 0, 255), 3)
    return img_d


def adjust_gamma(image, gamma=1.0):
    """
    A function to adjust the lighting in images.
    :param image: The image to change its light.
    :param gamma: The gamma value to change the light according to.
    :return: The image with the new lighting.
    """
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / float(normalization_factor)) ** invGamma) * normalization_factor for i in np.arange(0, normalization_factor + 1)]).astype(np.uint8)
    table = table.reshape((-1, 1))
    
    # apply gamma correction using the lookup table
    return cv2.LUT(src=image, lut=table)


def get_lighter_and_darker_images(img_list):
    """
    A function to create a list of copies of different lighting for each image in the given list.
    :param img_list:
    :return: a list of different lighting images.
    """
    l_and_d_images = []
    
    # gamma values
    gamma_dict = {0: 0.25, 1: 0.5, 2: 0.75, 3: 1.25, 4: 1.5, 5: 1.75}
    
    number_of_copies_to_create = 2
    
    # create the darker and lighter images
    for ind in range(0, len(img_list)):
        choice = np.random.choice(len(gamma_dict), number_of_copies_to_create, replace=False)
        l_and_d_images.append(img_list[ind])
        l_and_d_images.append(adjust_gamma(img_list[ind], gamma_dict[choice[0]]))
        l_and_d_images.append(adjust_gamma(img_list[ind], gamma_dict[choice[1]]))

    return l_and_d_images


def scale_to_net_size(image, lmks):
    """
    A function to scale the image and the landmarks to the size of the input of the CNN.
    :param image: The image to scale.
    :param lmks: The original landmarks.
    :return: The scaled imgae and landmarks.
    """
    if image is None:
        return None, None
    w = image.shape[1]
    h = image.shape[0]
    new_lmks = np.copy(lmks)
    
    # resizing the image
    try:
        scaled = cv2.resize(image, (cnn_image_size, cnn_image_size))
    except:
        print(image, lmks)
    
    # compute landmarks new positions
    new_lmks[:, 0] = lmks[:, 0] * (cnn_image_size / w)
    new_lmks[:, 1] = lmks[:, 1] * (cnn_image_size / h)
    
    return scaled, new_lmks


def extract_model_and_camera_matrix():
    """
    A function to extract the generic model and the camera matrix, which are used for the calculation of the pose
    out of the landmarks.
    :return: The camera matrix and the 3d landmarks points of the generic model.
    """
    
    temp_model = sio.loadmat(model_path)['model3D'][0][0]
    camera_matrix = temp_model[1]
    landmarks_3d = temp_model[7]
    return camera_matrix, landmarks_3d


def write_to_database(image, DoF6_vec, index, csv_path, directory_path):
    """
    A function to write a new image and a pose vector to the database.
    :param image: The image to write.
    :param DoF6_vec: The pose vector.
    :param index: The index in the database.
    :param csv_path: The csv file for the pose vector labeling.
    :param directory_path: The directory to write the image to.
    :return: None
    """
    vec_dict = {'file name': 'image_' + str(index) + '.jpg', 'rx': DoF6_vec[0], 'ry': DoF6_vec[1],
                'rz': DoF6_vec[2], 'tx': DoF6_vec[3], 'ty': DoF6_vec[4], 'tz': DoF6_vec[5]}
    df = pd.DataFrame.from_dict(vec_dict)
    df.to_csv(csv_path, mode='a', header=False, index=False)
    cv2.imwrite(directory_path + '//image_' + str(index) + '.jpg', image)
    return


def add_to_final_list(counter, image, pose_vec, image_directory, csv_pose_file):
    """
    A function to add an image and its pose vector to the final database.
    :param counter: The index to add the image according to.
    :param image: The image to add.
    :param pose_vec: The pose vector of the image.
    :param image_directory: The image directory of the final database.
    :param csv_pose_file: The csv file contains the pose vectors.
    :return: The counter of the next image in the database.
    """
    # the counter is used to enumerate the images
    write_to_database(image, pose_vec, counter, csv_path=csv_pose_file,
                      directory_path=image_directory)
    counter += 1
    return counter


def augment_and_create_database(image_list, lmks_list, database_counter, image_directory, csv_pose_file):
    """
    A function which takes a list of images and landmarks, augments it, and creates a database consists of images and
     labels, i.e. pose vectors.
    :param image_list: The list of images to create the database from.
    :param lmks_list: The landmarks of the images.
    :param database_counter: The index to start the database from.
    :param image_directory: The image directory to put the images in.
    :param csv_pose_file: The csv to write the pose vectors to.
    :return: The index of the final image in the database plus 1.
    """
    for i in range(0, len(image_list)):
        print(i)

        ##################################
        # START OF AUGMENTATION PROCESS  #
        ##################################

        ###################################################################
        # applying some operations for the augmentation process.          #
        # we move the bbox a little, expanding it, and scaling the image. #
        ###################################################################

        image_right, right_bbox, right_lmks = make_bbox_around_lmks(image_list[i], lmks_list[i])
        expanded_bbox = expand_bbox(image_right, right_bbox)
        translated_bbox = translate_bbox(image_right, expanded_bbox)

        scaled_image, scaled_bbox, scaled_lmks = scale_image_and_lms(image_right, translated_bbox, right_lmks)

        ##################################
        # rotating in 2 different angles #
        ##################################
        # the function rotates the image in a random angle between -30 and 30 degrees
        rotated_image_1, rotated_bbox_1, rotated_lmks_1 = rotate_image_and_lmks(scaled_image, scaled_bbox, scaled_lmks)
        rotated_image_2, rotated_bbox_2, rotated_lmks_2 = rotate_image_and_lmks(scaled_image, scaled_bbox, scaled_lmks)

        #show_image_with_bbox_and_lmks(rotated_image_2, rotated_bbox_2, rotated_lmks_2)
        #print('bbox: ', rotated_bbox_2)
        #print('lmks: ', rotated_lmks_2)

        ####################################################
        # cropping each rotated image in 2 different sizes #
        ####################################################

        # small crop
        cropped_image_1, cropped_lmks_1 = crop_image(rotated_image_1, rotated_bbox_1, rotated_lmks_1)
        # big crop
        b_cropped_image_1, b_cropped_lmks_1 = crop_bigger_image(rotated_image_1, rotated_bbox_1, rotated_lmks_1)

        cropped_image_2, cropped_lmks_2 = crop_image(rotated_image_2, rotated_bbox_2, rotated_lmks_2)
        b_cropped_image_2, b_cropped_lmks_2 = crop_bigger_image(rotated_image_2, rotated_bbox_2, rotated_lmks_2)

        #######################
        # flipping each image #
        #######################

        flipped_image_1, flipped_lmks_1 = flip_image_and_lmks(cropped_image_1, cropped_lmks_1)
        flipped_image_2, flipped_lmks_2 = flip_image_and_lmks(b_cropped_image_1, b_cropped_lmks_1)
        flipped_image_3, flipped_lmks_3 = flip_image_and_lmks(cropped_image_2, cropped_lmks_2)
        flipped_image_4, flipped_lmks_4 = flip_image_and_lmks(b_cropped_image_2, b_cropped_lmks_2)

        #####################
        # scale to net size #
        #####################

        img1, lmks1 = scale_to_net_size(cropped_image_1, cropped_lmks_1)
        img2, lmks2 = scale_to_net_size(b_cropped_image_1, b_cropped_lmks_1)
        img3, lmks3 = scale_to_net_size(cropped_image_2, cropped_lmks_2)
        img4, lmks4 = scale_to_net_size(b_cropped_image_2, b_cropped_lmks_2)
        img5, lmks5 = scale_to_net_size(flipped_image_1, flipped_lmks_1)
        img6, lmks6 = scale_to_net_size(flipped_image_2, flipped_lmks_2)
        img7, lmks7 = scale_to_net_size(flipped_image_3, flipped_lmks_3)
        img8, lmks8 = scale_to_net_size(flipped_image_4, flipped_lmks_4)

        ##########################
        # calculate pose vectors #
        ##########################

        img1_pose = get_DoF6_from_landmarks(lmks1.astype(np.double))
        img2_pose = get_DoF6_from_landmarks(lmks2.astype(np.double))
        img3_pose = get_DoF6_from_landmarks(lmks3.astype(np.double))
        img4_pose = get_DoF6_from_landmarks(lmks4.astype(np.double))
        img5_pose = get_DoF6_from_landmarks(lmks5.astype(np.double))
        img6_pose = get_DoF6_from_landmarks(lmks6.astype(np.double))
        img7_pose = get_DoF6_from_landmarks(lmks7.astype(np.double))
        img8_pose = get_DoF6_from_landmarks(lmks8.astype(np.double))

        pose_list = [img1_pose, img2_pose, img3_pose, img4_pose, img5_pose, img6_pose, img7_pose, img8_pose]
        img_list = [img1, img2, img3, img4, img5, img6, img7, img8]

        #####################################################
        # getting lighter and darker versions of the images #
        #####################################################

        light_and_dark_image_list = get_lighter_and_darker_images(img_list)

        ################################################################################
        # we create a total of 40 versions of every image, in the augmentation process #
        ################################################################################
        for ind in range(0, len(light_and_dark_image_list)):
            database_counter = add_to_final_list(database_counter, light_and_dark_image_list[ind],
                                                 pose_list[int(math.floor(ind / 3))], image_directory=image_directory,
                                                 csv_pose_file=csv_pose_file)

    return database_counter


camera_matrix, model_points = extract_model_and_camera_matrix()
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion




