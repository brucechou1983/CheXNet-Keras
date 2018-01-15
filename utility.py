import numpy as np
import os
import pandas as pd
import random
import shutil


def split_data(data_entry_file, class_names, train_patient_count, dev_patient_count,
               output_dir, random_state):
    """
    Create dataset split csv files

    """
    e = pd.read_csv(data_entry_file)

    # one hot encode
    for c in class_names:
        e[c] = e["Finding Labels"].apply(lambda labels: 1 if c in labels else 0)

    # shuffle and split
    pid = list(e["Patient ID"].unique())
    random.seed(random_state)
    random.shuffle(pid)
    train = e[e["Patient ID"].isin(pid[:train_patient_count])]
    dev = e[e["Patient ID"].isin(pid[train_patient_count:train_patient_count+dev_patient_count])]
    test = e[e["Patient ID"].isin(pid[train_patient_count+dev_patient_count:])]

    # export csv
    output_fields = ["Image Index", "Patient ID", "Finding Labels"] + class_names
    train[output_fields].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev[output_fields].to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    test[output_fields].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    return


def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts


def create_symlink_for_sample(src_file, label, dataset, dst_name, image_source_dir):
    """
    Create symlink for an image file

    """
    src_path = os.path.join(image_source_dir, src_file)
    dst_dir = os.path.join(dst_name, dataset, label)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, src_file)
    if not os.path.isfile(src_path):
        print(f"** warning: src image file {src_path} can't be found. **")
        return
    os.symlink(src_path, dst_path)


def create_symlink(image_source_dir, output_dir, dir_name):
    """
    Base on dataset splits, create different folders for .flow_from_directory()
    in ImageDataGenerator class

    """
    symlink_dir = os.path.join(output_dir, dir_name)
    # remove previous links
    if os.path.isdir(symlink_dir):
        print(f"delete {symlink_dir}")
        shutil.rmtree(symlink_dir)

    datasets = ("train", "dev", "test")
    for dataset in datasets:
        print(f"** create {dataset} dataset **")
        df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
        df.apply(lambda x: create_symlink_for_sample(
            x["Image Index"],
            x["Finding Labels"],
            dataset=dataset,
            dst_name=symlink_dir,
            image_source_dir=os.path.abspath(image_source_dir),
        ), axis=1)
