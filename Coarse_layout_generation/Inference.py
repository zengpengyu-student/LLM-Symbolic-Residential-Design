import cv2
import numpy as np
import torch
import os
from models.dit import MFDiT
from meanflow import MeanFlow
from CLIP_load import FrozenOpenCLIPEmbedder

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def img_process(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 255.0) * 2 - 1.0
    return img

def read_img(predict_num, livingroom_name, bedroom_name, kitchen_name, toilet_name, balcony_name, graph_name, bbox_name, name_list):
    sources = []
    prompts = []
    path_source = '../test/model_input'
    for name_scale in name_list:
        for i in range(predict_num):
            prompt = ''# Optional text condition for this example; replace with a case-specific prompt when needed. Empty text is supported.

            livingroom = cv2.imread(f'{path_source}/{name_scale}_{livingroom_name}')
            bedroom = cv2.imread(f'{path_source}/{name_scale}_{bedroom_name}')
            kitchen = cv2.imread(f'{path_source}/{name_scale}_{kitchen_name}')
            toilet = cv2.imread(f'{path_source}/{name_scale}_{toilet_name}')
            balcony = cv2.imread(f'{path_source}/{name_scale}_{balcony_name}')
            graph = cv2.imread(f'{path_source}/{name_scale}_{graph_name}')
            bbox = cv2.imread(f'{path_source}/{bbox_name}')

            if livingroom is None: livingroom = np.zeros((256, 256, 3), dtype=np.uint8)
            if bedroom is None: bedroom = np.zeros((256, 256, 3), dtype=np.uint8)
            if kitchen is None: kitchen = np.zeros((256, 256, 3), dtype=np.uint8)
            if toilet is None: toilet = np.zeros((256, 256, 3), dtype=np.uint8)
            if balcony is None: balcony = np.zeros((256, 256, 3), dtype=np.uint8)
            if graph is None: graph = np.zeros((256, 256, 3), dtype=np.uint8)
            if bbox is None: bbox = np.zeros((256, 256, 3), dtype=np.uint8)

            livingroom = img_process(livingroom)
            bedroom = img_process(bedroom)
            kitchen = img_process(kitchen)
            toilet = img_process(toilet)
            balcony = img_process(balcony)
            graph = img_process(graph)
            bbox = img_process(bbox)
            source = np.concatenate((graph, livingroom, bedroom, kitchen, toilet, balcony, bbox), axis=-1)
            source = source.transpose(2, 0, 1)
            source = torch.from_numpy(source).cuda()

            sources.append(source)
            prompts.append(prompt)
    sources = torch.stack(sources, dim=0)
    prompts_tuple = tuple(prompts)

    return [sources, prompts_tuple]

if __name__ == "__main__":
    name_list = ['small1']

    device = 'cuda'
    img_size = 64
    predict_num = 16
    livingroom_name = 'living_room.png'
    bedroom_name = 'bedroom.png'
    kitchen_name = 'kitchen.png'
    toilet_name = 'bathroom.png'
    balcony_name = 'balcony.png'
    graph_name = 'graph.png'
    bbox_name = 'edge_64.png'

    clip_embedder = FrozenOpenCLIPEmbedder(arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda")
    model = MFDiT(
        input_size=img_size,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        clip_embedder=clip_embedder
    ).to(device)

    checkpoint_path = '../Checkpoint/Coarse_layout_generation.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    print(f'success load: {checkpoint_path} ')

    meanflow = MeanFlow(channels=3, image_size=img_size, num_classes=predict_num)

    c_test = read_img(predict_num, livingroom_name, bedroom_name, kitchen_name, toilet_name, balcony_name, graph_name, bbox_name, name_list)

    model_module = model.module if hasattr(model, 'module') else model
    for j in range(5):
        images = meanflow.sample_each_class(model_module, 1, c_test=c_test)
        for i in range(images.shape[0]):
            image = images[i]
            img_save_path = f"../test/Coarse_output/{j*16 + i}.png"
            meanflow.save_images(image, img_save_path)

