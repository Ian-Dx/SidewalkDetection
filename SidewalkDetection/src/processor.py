""" Pre-process images to data and target in the type of Tensor """
import torch
import cv2
import numpy as np
import time
import os

# 19 RGB color pixel, each representing a semantic object
RGB_MAP = np.array(
    [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
     [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
     [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
)

# A token array represents different type of object by adding up the R, G, B channels
TOKEN_MAP = RGB_MAP.sum(-1)

# Assert this way of processing doesn't bring duplicated token
assert len(TOKEN_MAP) == len(set(TOKEN_MAP)), "Duplicated token map!"


def data_reader_single(img_path):
    """
    Read one picture in the data set
    :param img_path: Image path that contains data
    :return: data
    """
    # Read the image
    data = cv2.imread(img_path)
    # BGR -> RGB & [0, 255] -> [0, 1]
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) / 255.0
    # (H, W, C) -> (C, H, W)
    data = data.transpose(2, 1, 0)

    return data


def target_reader_single(img_path):
    """
    Convert a colored picture to a classed target
    :param img_path: Image path that contains targets
    :return: target
    """
    # Read the image
    img = cv2.imread(img_path)
    # Add the RGB channels to generate a token
    img_token = img.sum(-1)
    # State an empty numpy array
    target = np.zeros(img_token.shape)

    # Each pixel stands for an object, representing by a token
    # Background's token is set as 0
    for idx, token in enumerate(TOKEN_MAP):
        target = np.where(img_token == token, idx, target)

    return target


def data_reader(dir_path):
    """
    Read all the images in the given directory
    :param dir_path: Directory path
    :return: Tensor data
    """
    # State an empty array to hold data
    data = []
    dirs = []
    imgs = []

    # List all the directories in the given directory
    for directory in os.listdir(dir_path):
        dirs.append(directory)

    # Keep dirs in order
    dirs.sort()

    for directory in dirs:
        # List all the files in the given directory
        for file in os.listdir(directory):
            # Make sure the file is a png image
            if (os.path.splitext(file)[1] == ".png") & (file.startswith(".") == False):
                # Data add new item
                img_path = dir_path + file
                imgs.append(img_path)
                print(img_path)

    # Keep images in order
    imgs.sort()

    for img in imgs:
        data.append(data_reader_single(img))

    # Convert to tensor
    data = np.array(data)
    data = torch.from_numpy(data).type(torch.FloatTensor)

    return data, imgs


def target_reader(dir_path):
    """
    Read all the images in the given directory
    :param dir_path: Directory path
    :return: Tensor targets
    """
    # State an empty array to hold targets
    target = []
    dirs = []
    imgs = []

    # List all the directories in the given directory
    for directory in os.listdir(dir_path):
        dirs.append(directory)

    # Keep dirs in order
    dirs.sort()

    for directory in dirs:
        # List all the files in the given directory
        for file in os.listdir(directory):
            # Make sure the file is a png image
            if file.endswith("color.png") & (file.startswith(".") == False):
                # Targets add new item
                img_path = dir_path + file
                imgs.append(img_path)

    # Keep images in order
    imgs.sort()

    for img in imgs:
        target.append(target_reader_single(img))

    # Convert to tensor
    target = np.array(target)
    target = torch.from_numpy(target).type(torch.LongTensor)

    return target, imgs


def main():
    """
    Test function
    :return:
    """
    # The average time to read both data and target once is around 0.25s
    time1 = time.time()
    # target_reader("/Volumes/Files/SidewalkDetection/gtFine/train/aachen/")
    data_reader("/Volumes/Files/SidewalkDetection/leftImg8bit/train/aachen/")
    time2 = time.time()
    print(time2-time1)


if __name__ == "__main__":
    main()
