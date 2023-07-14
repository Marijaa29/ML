from pedestrian_dataset_loader import PedestrianDatasetLoader
from sklearn.neural_network import MLPClassifier
import joblib 
import yaml

DATASET_PATH = "dataset/dataset"
DATASET_TEST_SIZE = 0.2
MODEL_PATH = "pedestrian_classifier.joblib"
CFG_FILE_PATH = "model_cfg.yaml"

def save_model_cfg(cfg_path, mean_pedestrian_height, mean_pedestrian_width):
    model_cfg = {"MEAN_PEDESTRIAN_HEIGHT": mean_pedestrian_height, "MEAN_PEDESTRIAN_WIDTH": mean_pedestrian_width}
    
    with open(cfg_path, "w") as file:
        yaml.dump(model_cfg, file)
        

pedestrian_dataset_loader = PedestrianDatasetLoader(DATASET_PATH)
X_train, X_test, y_train, y_test  = pedestrian_dataset_loader.load_pedestrian_dataset(DATASET_TEST_SIZE)
pedestrian_dataset_loader.print_dataset_info()

print("Training classifier...")

classifier = MLPClassifier(random_state=1,
              hidden_layer_sizes=(100, 50),
              solver="sgd",
              verbose=True,
              max_iter=300,
              batch_size=100,
              early_stopping=True,
              validation_fraction=0.12,
              learning_rate_init=0.01).fit(X_train, y_train)

print("Done training!")
# print(classifier.loss_curve_)
# print(classifier.validation_scores_)
classification_score = classifier.score(X_test, y_test)
print("Classification score: {}".format(classification_score))


print("Saving model and model cfg...")
joblib.dump(classifier, MODEL_PATH)
save_model_cfg(CFG_FILE_PATH, pedestrian_dataset_loader.get_mean_pedestrian_height(), pedestrian_dataset_loader.get_mean_pedestrian_width())
print("Model and model cfg saved!")