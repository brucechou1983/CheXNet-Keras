# ChexNet-Keras
This project is a tool to build CheXNet-like models, written in Keras.

<img width="450" height="450" src="https://stanfordmlgroup.github.io/projects/chexnet/img/chest-cam.png" alt="CheXNet from Stanford ML Group"/>

## What is [CheXNet](https://arxiv.org/pdf/1711.05225.pdf)?
ChexNet is a deep learning algorithm that can detect and localize 14 kinds of diseases from chest X-ray images. As described in the paper, a 121-layer densely connected convolutional neural network is trained on ChestX-ray14 dataset, which contains 112,120 frontal view X-ray images from 30,805 unique patients. The result is so good that it surpasses the performance of practicing radiologists.

## In this project, you can
1. Train/test a **baseline model** by following the quickstart. You can get a model with performance close to the paper.
2. Modify `multiply` and `use_class_balancing` parameters in `config.ini` to see if you can get better performance.
3. Modify `weights.py` to customize your weights in loss function.
4. Every time you do a new experiment, make sure you modify `output_dir` in `config.ini` otherwise previous training results might be overwritten. For more options check the parameter description in `config.ini`.

## Quickstart
**Note that currently this project can only be executed in Linux and macOS. You might run into some issues in Windows.**
1. Download **all tar files** and **Data_Entry_2017.csv** of ChestX-ray14 dataset from [NIH dropbox](https://nihcc.app.box.com/v/ChestXray-NIHCC). Put them under `./data` folder and untar all tar files.
2. Download DenseNet-121 ImageNet tensorflow pretrained weights from [DenseNet-Keras](https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc). Specify the file path in `config.ini` (field: `base_model_weights_file`)
3. Create & source a new virtualenv. Python >= **3.6** is required.
4. Install dependencies by running `pip3 install -r requirements.txt`.
5. Run `python train.py` to train a new model. If you want to run the training using multiple GPUs, just prepend `CUDA_VISIBLE_DEVICES=0,1,...` to restrict the GPU devices. `nvidia-smi` command will be helpful if you don't know which device are available.
6. Run `python test.py` to test the model.

## CAM
Reference: [Grad-CAM](https://arxiv.org/pdf/1610.02391). CAM image is generated as accumumlated weighted activation before last global average pooling (GAP) layer. It is scaled up to 224\*224 to match original image.
```
python test_cam.py
```
CAM images will be generated into $pwd/imgdir, please make sure you've created the target directory before running test_cam.py

Guided back-prop is still an enhancement item.
## TODO
1. More baseline models

## Acknowledgement
I would like to thank Pranav Rajpurkar (Stanford ML group) and Xinyu Weng (北京大學) for sharing their experiences on this task. Also I would like to thank Felix Yu for providing DenseNet-Keras source code.

## Author
Bruce Chou (brucechou1983@gmail.com)

## License
MIT
