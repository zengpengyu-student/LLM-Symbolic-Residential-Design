from ddpm import Diffusion
import cv2
import numpy as np
import torch
from utils.utils import show_pred
import os
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def postprocess_output(x):
    x *= 0.5
    x += 0.5
    x *= 255
    return x

def read_img(path_outer, predict_num):
    for i in range(predict_num):
        # source = cv2.resize(source, (64, 64), interpolation=cv2.INTER_AREA)
        source_outer_contour = cv2.imread(path_outer[i])
        source_outer_contour = cv2.resize(source_outer_contour, (64, 64), interpolation=cv2.INTER_AREA)

        # Do not forget that OpenCV read images in BGR order.
        source_outer_contour = cv2.cvtColor(source_outer_contour, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_outer_contour = source_outer_contour.astype(np.float32) / 255.0
        source_outer_contour = np.expand_dims(source_outer_contour.transpose(2, 0, 1), axis=0)

        if i == 0:
            sources = source_outer_contour
        else:
            sources = np.concatenate((sources, source_outer_contour), axis=0)
    sources = torch.from_numpy(sources).cuda()
    return sources

if __name__ == "__main__":

    path_contour_all = '../test/Coarse_output/'
    list_img = os.listdir(path_contour_all)
    ddpm = Diffusion()
    sample_num = 50 # Number of samples

    num_full_parts = len(list_img) // sample_num
    remainder = len(list_img) % sample_num
    split_list = []
    start_index = 0
    for i in range(num_full_parts):
        end_index = start_index + sample_num
        split_list.append(list_img[start_index:end_index])
        start_index = end_index
    if remainder > 0:
        split_list.append(list_img[start_index:])

    for sub_list in split_list:
        num_list = len(sub_list)
        path_outer_contour = []
        img_name_cache = []
        for img_name in sub_list:
            path_outer_contour.append(f'{path_contour_all}/{img_name}')
            img_name_cache.append(img_name)
        source = read_img(path_outer_contour, num_list)

        img_generation = show_pred(num_list, ddpm.net, "cuda", source=source)

        for j in range(num_list):
            folder_name = f'../test/Fine_output/cache_img/'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            #img_name_new = img_name.split('.')[0]
            save_path_1x1 = f'{folder_name}/{img_name_cache[j]}'
            test_images = postprocess_output(img_generation[j].cpu().data.numpy().transpose(1, 2, 0))
            test_images = np.uint8(test_images)
            #test_images[test_images == 255] = 0
            test_images_origin = test_images.copy()
            #test_images_origin[test_images_origin <= 5] = 255
            # img[img == [0, 255, 255]] = 255
            # img[img == [255, 0, 255]] = 255
            # img[img == [255, 255, 0]] = 255
            #test_images_origin = cv2.resize(test_images_origin, (256, 256), interpolation=cv2.INTER_NEAREST)
            #cv2.imwrite(f'./test/output/0_delete.png', img)
            #test_images = cv2.cvtColor(test_images, cv2.COLOR_BGR2RGB)
            Image.fromarray(test_images_origin).save(f'{save_path_1x1}')
            print(f'i:{img_name_cache[j]}')
