import joblib
from augmentation_utils import *
from keras import backend as K
import constants

csv_path = 'test_prediction.csv'

def use_model_on_test_set():
    """
    A function to predict poses on the test set.
    :return:
    """
    # loading the models
    rot_model = joblib.load('final_rot_model.pkl')
    loc_model = joblib.load('final_loc_model.pkl')

    img_list = []
    img_names = []

    # creating a list of images of the test set
    for i in range(0, 10):
        img_list.append(cv2.imread('test_set/image_0000' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))
        img_names.append('image_0000' + str(i) + '.png')

    for i in range(10, 100):
        img_list.append(cv2.imread('test_set/image_000' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))
        img_names.append('image_000' + str(i) + '.png')

    for i in range(100, 368):
        img_list.append(cv2.imread('test_set/image_00' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))
        img_names.append('image_00' + str(i) + '.png')

    img_arr = np.zeros((368, cnn_input_image_size, cnn_input_image_size, 1), dtype=np.uint8)
    for i in range(0, len(img_arr)):
        img_arr[i] = cv2.resize(img_list[i], (cnn_input_image_size, cnn_input_image_size)).reshape((cnn_input_image_size, cnn_input_image_size, 1))

    # predicting poses for test set
    img_arr = img_arr / float(image_normalization_factor)

    rot_prediction = rot_model.predict(img_arr)
    loc_prediction = loc_model.predict(img_arr)

    prediction = np.hstack((rot_prediction, loc_prediction))

    # writing predictions to csv file
    d1 = {'file_name': img_names, 'rx': prediction[:, 0], 'ry': prediction[:, 1], 'rz': prediction[:, 2],
          'tx': prediction[:, 3], 'ty': prediction[:, 4], 'tz': prediction[:, 5]}
    df = pd.DataFrame.from_dict(d1)
    df.to_csv(csv_path)

    # demonstrate the test set images with pose vector drawn on
    for i in range(0, len(img_arr)):
        img = cv2.resize(img_list[i], (cnn_input_image_size, cnn_input_image_size))
        est = draw_axis(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (cnn_input_image_size, cnn_input_image_size)),
                        prediction[i][:3], prediction[i][3:], camera_matrix)
        print(prediction[i])
        cv2.imshow('prediction', est)
        cv2.waitKey()
    return


# predicting poses for test set
use_model_on_test_set()
K.clear_session()
