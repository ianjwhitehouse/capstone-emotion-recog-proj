import numpy as np
import cv2
import os

# for properly reading grayscale images
def imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# go through each folder in the specified master folder (train/test) and get image data/respective labels (as numpy arrays)
def get_data(train_test_folder):
    images = []
    emot_labels = []

    # go through each subfolder (named after each emotion) containing the respective images in the train/test folder
    for emotion in os.listdir(train_test_folder):
        emotion_path = os.path.join(train_test_folder, emotion)
        # read each image (.jpg and grayscale) as a numpy array
        for file_name in os.listdir(emotion_path):
            if file_name.endswith(".jpg"):
                file_path = os.path.join(emotion_path, file_name)
                # print(file_path)
                new_image = imread(file_path)
                images.append(new_image)
                emot_label = None

                # conditions for assigning labels
                if emotion == "happy":
                    emot_label = 0
                elif emotion == "sad":
                    emot_label = 1
                elif emotion == "neutral":
                    emot_label = 2
                elif emotion == "surprise":
                    emot_label = 3
                elif emotion == "angry":
                    emot_label = 4
                elif emotion == "disgust":
                    emot_label = 5
                elif emotion == "fear":
                    emot_label = 6
                
                # assign labels
                emot_labels.append(emot_label)

                # sanity check
                assert len(emot_labels) == len(images)
        
        print("Finished processing:", train_test_folder, "-", emotion)
        assert len(emot_labels) == len(images)

    assert len(emot_labels) == len(images)
    images = np.array(images)
    emot_labels = np.array(emot_labels)

    return images, emot_labels

# get training and testing data
X_train, y_train = get_data("train")
X_test, y_test = get_data("test")

# save numpy arrays for more convenient access to data in future sessions
np.savez_compressed("feri_ds.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)