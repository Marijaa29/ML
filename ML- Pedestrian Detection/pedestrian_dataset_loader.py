import os
import cv2
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split

class PedestrianDatasetLoader:
    def __init__(self, dataset_path):
        self.__dataset_path = dataset_path
        self.__mean_pedestrian_width = 0
        self.__mean_pedestrian_height = 0
        self.__number_of_pedestrian_samples = 0
        self.__number_of_no_pedestrian_samples = 0
    
    def __get_pedestrian_average_size(self):
        pedestrian_images_path = os.path.join(self.__dataset_path, "ped")
        pedestrian_image_counter = 0
        pedestrian_height_total = 0
        pedestrian_width_total = 0
        
        for filname in os.listdir(pedestrian_images_path):
            if filname.endswith(".png"):
                pedestrian_image_counter +=1
                image_path = os.path.join(pedestrian_images_path, filname)
                ped_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_height, image_width = ped_img.shape
                pedestrian_height_total += image_height
                pedestrian_width_total += image_width
            return pedestrian_width_total // pedestrian_image_counter, pedestrian_height_total // pedestrian_image_counter
                
    
    def __load_pedestrian_subset(self, load_pedestrian_part):
        
        if load_pedestrian_part:
            dir_name = "ped"
        else:
            dir_name = "no_ped"
        
        images_path = os.path.join(self.__dataset_path, dir_name)
        
        HOG_list = []
        
        for file in  os.listdir(images_path):
            if file.endswith(".png"):
                image_path = os.path.join(images_path, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.__mean_pedestrian_width, self.__mean_pedestrian_height))
                
                HOG_desc, hog_img = hog(image, visualize=True)
                # cv2.imshow("hog image", hog_img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                # print(HOG_desc.shape)
                HOG_list.append(HOG_desc)
        
        hogs = np.array(HOG_list)
        return hogs
                
    def load_pedestrian_dataset(self, test_size):
        print("Loading dataset...")
        self.__mean_pedestrian_width, self.__mean_pedestrian_height = self.__get_pedestrian_average_size()

        ped_dataset = self.__load_pedestrian_subset(True)
        no_ped_dataset = self.__load_pedestrian_subset(False)
        
        #print(ped_dataset.shape)
        #print(no_ped_dataset.shape)
        
        pedestrian_labels = np.ones((ped_dataset.shape[0]))
        no_pedestrian_labels = np.zeros((no_ped_dataset.shape[0]))
        
        self.__number_of_pedestrian_samples = len(pedestrian_labels)
        self.__number_of_no_pedestrian_samples = len(no_pedestrian_labels)

        
        X = np.concatenate((ped_dataset, no_ped_dataset))
        y = np.concatenate((pedestrian_labels, no_pedestrian_labels))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=1)
        print("Dataset loaded!")
        
        return  X_train, X_test, y_train, y_test
    
    def print_dataset_info(self):
        print()
        print("PEDESTRIAN DATASET INFO:")
        print("\tNumber of dataset samples: {}".format(self.__number_of_pedestrian_samples + self.__number_of_no_pedestrian_samples))
        print("\tNumber of pedestrian samples: {}".format(self.__number_of_pedestrian_samples))
        print("\tNumber of no-pedestrian samples: {}".format(self.__number_of_no_pedestrian_samples))
        print("\tAverage pedestrian height: {}".format(self.__mean_pedestrian_height))
        print("\tAverage pedestrian width: {}".format(self.__mean_pedestrian_width))
        print()
        
    def get_mean_pedestrian_height(self):
        return self.__mean_pedestrian_height
    
    def get_mean_pedestrian_width(self):
        return self.__mean_pedestrian_width
        
# loader = PedestrianDatasetLoader("dataset/dataset")
# loader.load_pedestrian_dataset(0.15)
# loader.print_dataset_info()