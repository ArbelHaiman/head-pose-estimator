from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import joblib
from keras.optimizers import Adam
from augmentation_utils import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


total_examples_for_database = 316152
size_of_images = 150


def create_model():
    """
    A function to create a model for predicting the pose vector of a face image.
    :return: None
    """
    # checking that GPU is available.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # labels of database
    labels = np.genfromtxt("final_pose_estimation_DB.csv",
                           delimiter=",", skip_header=1)[0:total_examples_for_database, 1:]
    # dumping the name of the image
    labels = labels[:, 3:]

    # loading the images
    database = np.zeros((total_examples_for_database, size_of_images ** 2), dtype=np.uint8)
    for i in range(0, database.shape[0]):
        path = 'final_database//image_' + str(i) + '.jpg'
        print(i)
        current_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        current_image = current_image.flatten()
        database[i, :] = current_image

    X_train, X_test, y_train, y_test = train_test_split(database, labels, test_size=0.2)

    X_train = X_train.reshape(X_train.shape[0], size_of_images, size_of_images, 1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], size_of_images, size_of_images, 1) / 255.0

    ###################################################################
    # extracting the validation set for using in the training process #
    ###################################################################
    img_list = []
    for i in range(0, 10):
        img_list.append(cv2.imread('validation/image_0000' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))

    for i in range(10, 100):
        img_list.append(cv2.imread('validation/image_000' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))

    img_arr = np.zeros((100, 150, 150, 1), dtype=np.uint8)
    for i in range(0, 100):
        img_arr[i] = cv2.resize(img_list[i], (150, 150)).reshape((150, 150, 1))

    img_arr = img_arr / 255.0

    val_labels = np.genfromtxt("valid_set2.csv", delimiter=",", skip_header=1)[0:100, 5:8]
    print(val_labels[0])
    ###################################################################
    # end of this process #############################################
    ###################################################################

    batch_size = 128
    epochs = 80

    # data_to_learn = np.vstack((X_train, img_arr))
    # labels_to_learn = np.vstack((y_train, val_labels))
    # print(data_to_learn.shape)
    # print(labels_to_learn.shape)
    model = Sequential()

    # conv layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size_of_images, size_of_images, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3))

    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='mean_squared_error', optimizer=opt)
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks_list,
              validation_data=(img_arr, val_labels))

    for i in range(0, 10):
        image = X_test[i].reshape(1, size_of_images, size_of_images, 1)
        prediction = model.predict(image)
        print('real: ', y_test[i])
        print('pred: ', prediction)

    joblib.dump(model, 'final_loc_model.pkl')
    return


camera_matrix, model_points = extract_model_and_camera_matrix()
create_model()
