import os
import random

ucm_image_file_dir = "datasets/Images"
ucm_images = "../datasets/ucm_images.txt"  # file path contains the image path

cwd = os.getcwd()[:-6].replace('\\','/') # load error due to the sys path

def make_images_txt(ucm_image_file_dir, ucm_images):
    if (os.path.exists(ucm_images)):
        os.remove(ucm_images)
    # if (os.path.exists(ucm_eval_dir)):
        # os.remove(ucm_eval_dir)
    ucm_train_list = []
    images_class_list = os.listdir(ucm_image_file_dir) # obtain the listdir of the ../data/images

    for images_class in images_class_list:
        images_class_dir = ucm_image_file_dir + "/" + images_class
        images_files_list = os.listdir(images_class_dir)
        # count = 0
        for i in images_files_list:
            # count += 1
            images = cwd + images_class_dir[2:] + "/" + i # real path of each image
            ucm_train_list.append(images + "\n")
            # if count % 10 == 0: # define 1 eval figure in each 10 figures
            #     ucm_eval_list.append(images + "\n")
            # else:
            #     ucm_train_list.append(images + "\n")
    with open(ucm_images, "a") as f:
        for train_image in ucm_train_list:
            f.write(train_image)
    # with open(ucm_eval_dir, "a") as f:
    #     for eval_image in ucm_eval_list:
    #         f.write(eval_image)

make_images_txt(ucm_image_file_dir, ucm_images)
print("over")