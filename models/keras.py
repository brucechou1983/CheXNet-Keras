import importlib
from keras.layers.core import Dense
from keras.models import Model


class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
            ),
        )

    def get_input_shape(self, model_name):
        return self.models_[model_name]["input_shape"]

    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                  weights_path=None):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self.models_[model_name]['module_name']}"
            ),
            model_name)

        base_model = base_model_class(
            include_top=False,
            input_shape=self.models_[model_name]["input_shape"],
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        predictions = []
        for i, class_name in enumerate(class_names):
            prediction = Dense(1024)(x)
            prediction = Dense(1, activation="sigmoid", name=class_name)(prediction)
            predictions.append(prediction)
        model = Model(inputs=base_model.input, outputs=predictions)

        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            model.load_weights(weights_path)
        return model
