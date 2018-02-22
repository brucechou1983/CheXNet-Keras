import csv
import os
from configparser import ConfigParser

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score

import grad_cam as gc
from callback import load_generator_data
from generator import custom_image_generator
from models.densenet121 import get_model
from utility import get_sample_counts


def grad_cam(model, class_names, y, y_hat, test_generator):
    print("** perform grad cam **")
    os.makedirs("imgdir", exist_ok=True)
    with open('predicted_class.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_header = ['ID', 'Most probable diagnosis']
        for i, v in enumerate(class_names):
            csv_header.append(f"{v}_Prob")
        csvwriter.writerow(csv_header)
        for i, v in enumerate(y_hat):
            predicted_class = np.argmax(v)
            labeled_class = np.argmax(y[i])
            print(
                f"** y_hat[{i}] = {v.round(3)} Label/Prediction: {class_names[labeled_class]}/{class_names[predicted_class]}")
            csv_row = [str(i + 1), f"{class_names[predicted_class]}"] + [str(vi.round(3)) for vi in v]
            csvwriter.writerow(csv_row)
            x_orig = test_generator.orig_input(i).squeeze()
            x_orig = cv2.cvtColor(x_orig, cv2.COLOR_GRAY2RGB)
            x = test_generator.model_input(i)
            cam = gc.grad_cam(model, x, x_orig, predicted_class, "conv5_blk_scale", class_names)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(x_orig, f"Labeled as:{class_names[labeled_class]}", (5, 20), font, 1,
                        (255, 255, 255),
                        2, cv2.LINE_AA)

            cv2.putText(cam, f"Predicted as:{class_names[predicted_class]}", (5, 20), font, 1,
                        (255, 255, 255),
                        2, cv2.LINE_AA)

            print(f"Writing cam file to imgdir/gradcam_{i}.jpg")

            cv2.imwrite(f"imgdir/gradcam_{i}.jpg", np.concatenate((x_orig, cam), axis=1))


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
        ImageDataGenerator(horizontal_flip=True, rescale=1. / 255),
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

    grad_cam(model, class_names, y, y_hat, test_generator)


if __name__ == "__main__":
    main()
