import joblib
import yaml
import cv2
from skimage.feature import hog

MODEL_PATH = "pedestrian_classifier.joblib"
CFG_PATH = "model_cfg.yaml"
TEST_IMAGE_PATH = "dataset/dataset/test_img.png"
SLIDING_WINDOW_STEP = 25

pedestrian_classifier = joblib.load(MODEL_PATH)

with open (CFG_PATH, "r") as file:
    model_cfg_dict = yaml.safe_load(file)
    
mean_pedestrian_height = model_cfg_dict["MEAN_PEDESTRIAN_HEIGHT"]
mean_pedestrian_width = model_cfg_dict["MEAN_PEDESTRIAN_WIDTH"]

test_image = cv2.imread(TEST_IMAGE_PATH)
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

image_height, image_width = test_image_gray.shape

max_pedestrian_probability = 0
final_top_left_bb = (0,0)

for i in range(0, image_height - mean_pedestrian_height, SLIDING_WINDOW_STEP):
    for j in range (0, image_width - mean_pedestrian_width, SLIDING_WINDOW_STEP):
        top_left_bb = (j,i)
        roi = test_image_gray[i: i+mean_pedestrian_height, j: j+mean_pedestrian_width]
        HOG_desc, hog_image = hog(roi, visualize=True)
        HOG_desc = HOG_desc.reshape((1,-1))
        roi_pedestrian_probabilities = pedestrian_classifier.predict_proba(HOG_desc)
        is_pedestrian_probability = roi_pedestrian_probabilities[0, 1]
        
        if is_pedestrian_probability > max_pedestrian_probability:
            max_pedestrian_probability = is_pedestrian_probability
            final_top_left_bb = top_left_bb
            
print("Max pedestrian probability: {}".format(max_pedestrian_probability))
cv2.rectangle(test_image, final_top_left_bb, (final_top_left_bb[0]+mean_pedestrian_width, final_top_left_bb[1]+mean_pedestrian_height), color=(0,255,0), thickness=3)
cv2.imshow("Final detection", test_image)
cv2.waitKey()
cv2.destroyAllWindows()
