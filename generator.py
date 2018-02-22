import numpy as np


def custom_image_generator(generator, directory, class_names, batch_size=16, target_size=(224, 224),
                           color_mode="rgb", class_mode="binary", mean=None, std=None, cam=False):
    """
    In paper chap 3.1:

    we downscale the images to 224x224 and normalize based
    on the mean and standard deviation of images in the
    ImageNet training set
    """

    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])

    iterator = generator.flow_from_directory(directory=directory,
                                             target_size=target_size,
                                             color_mode=color_mode,
                                             class_mode=class_mode,
                                             batch_size=batch_size)
    # class index -> xxxx|xxxx
    class_indices_reversed = dict((v, k) for k, v in iterator.class_indices.items())

    for batch_x, batch_y in iterator:
        batch_y_multilabel = []
        for i in range(batch_y.shape[0]):
            # class index -> xxxx|xxxx -> one hot
            batch_y_multilabel.append(
                label2vec(class_indices_reversed[batch_y[i]], class_names))

        # now shape is (batch#, 14)
        batch_y_multilabel = np.array(batch_y_multilabel)
        # make the output [y1, y2, y3 ... y14] where yx shape is (batch#, 1)
        if not cam:
            yield (batch_x - mean) / std, [np.array(y) for y in batch_y_multilabel.T.tolist()]
        else:  # no normalization
            yield batch_x, [np.array(y) for y in batch_y_multilabel.T.tolist()]


def label2vec(label, class_names):
    vec = np.zeros(len(class_names))
    if label == "No Finding":
        return vec
    labels = label.split("|")
    for l in labels:
        vec[class_names.index(l)] = 1
    return vec
