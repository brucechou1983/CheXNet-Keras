import numpy as np
import os
from callback import load_generator_data
from configparser import ConfigParser
from generator import custom_image_generator
from models.densenet121 import get_model
from utility import get_sample_counts
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    symlink_dir_name = "image_links"
    test_data_path = f"{output_dir}/{symlink_dir_name}/test/"

    step_test = int(test_counts / batch_size)
    print("** load test generator **")
    test_generator = custom_image_generator(
        ImageDataGenerator(horizontal_flip=True, rescale=1./255),
        test_data_path,
        batch_size=batch_size,
        class_names=class_names,
    )
    x, y = load_generator_data(test_generator, step_test, len(class_names))

    print("** load model **")
    model = get_model(class_names)
    if use_best_weights:
        print("** use best weights **")
        model.load_weights(best_weights_path)
    else:
        print("** use last weights **")
        model.load_weights(weights_path)

    print("** make prediction **")
    y_hat = model.predict(x)

    test_log_path = os.path.join(output_dir, "test.log")
    print(f"** write log to {test_log_path} **")
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[i], y_hat[i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(f"{class_names[i]}: {score}\n")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")


if __name__ == "__main__":
    main()
