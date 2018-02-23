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
from models.keras import ModelFactory
from utility import get_sample_counts


def grad_cam(model, class_names, y, y_hat, x_model, x_orig, last_conv_layer):
    print("** perform grad cam **")
    y = np.swapaxes(np.array(y).squeeze(), 0, 1)
    y_hat = np.swapaxes(np.array(y_hat).squeeze(), 0, 1)
    print(f"** Shapes of y/y_hat are {np.shape(y)}/{np.shape(y_hat)} **")
    print(f"** Shapes of x_orig/x_model are {np.shape(x_orig)}/{np.shape(x_model)} **")
    os.makedirs("imgdir", exist_ok=True)
    with open('predicted_class.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_header = ['ID', 'Most probable diagnosis']
        for i, v in enumerate(class_names):
            csv_header.append(f"{v}_Prob")
        csvwriter.writerow(csv_header)
        for i, v in enumerate(y_hat):
            print(f"** y_hat[{i}] = {v}")
            print(f"** y[{i}] = {y[i]}")
            predicted_class = np.argmax(v)
            labeled_classes = ",".join([class_names[yi] for yi, yiv in enumerate(y[i]) if yiv == 1])
            if labeled_classes == "":
                labeled_classes = "Normal"
            print(f"** Label/Prediction: {labeled_classes}/{class_names[predicted_class]}")
            csv_row = [str(i + 1), f"{class_names[predicted_class]}"] + [str(vi.round(3)) for vi in v]
            csvwriter.writerow(csv_row)
            x_orig_i = 255 * x_orig[i].squeeze()
            x_model_i = x_model[i][np.newaxis, :, :, :]
            cam = gc.grad_cam(model, x_model_i, x_orig_i, predicted_class, last_conv_layer,
                              class_names)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(x_orig_i, f"Labeled as:{labeled_classes}", (5, 20), font, fontScale=0.5, color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(cam, f"Predicted as:{class_names[predicted_class]}", (5, 20), font, fontScale=0.5,
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            print(f"Writing cam file to imgdir/gradcam_{i}.jpg")

            cv2.imwrite(f"imgdir/gradcam_{i}.jpg", np.concatenate((x_orig_i, cam), axis=1))


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
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

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    print("** load test generator **")
    test_generator = custom_image_generator(
        ImageDataGenerator(horizontal_flip=True, rescale=1. / 255),
        test_data_path,
        batch_size=batch_size,
        class_names=class_names,
        cam=False,
        target_size=model_factory.get_input_size(base_model_name),
    )
    test_generator_orig = custom_image_generator(
        ImageDataGenerator(horizontal_flip=True, rescale=1. / 255),
        test_data_path,
        batch_size=batch_size,
        class_names=class_names,
        cam=True,
        target_size=model_factory.get_input_size(base_model_name),
    )
    x, y = load_generator_data(test_generator, step_test, len(class_names))
    x_orig, _ = load_generator_data(test_generator_orig, step_test, len(class_names))

    print("** make prediction **")
    y_hat = model.predict(x, verbose=1)

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

    grad_cam(model, class_names, y, y_hat, x, x_orig, model_factory.get_last_conv_layer())


if __name__ == "__main__":
    main()
