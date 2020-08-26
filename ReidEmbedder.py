import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision

from torchreid.utils import load_pretrained_weights
from reid_utils.default_config import get_default_config
from reid_utils.builder import build_model




class ReidEmbedder():
    def __init__(self):
        weightPath = "/opt/models/reid300.pt"
        configPath = "/opt/models/reid300.yaml"

        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")

        cfg = get_default_config()
        cfg.merge_from_file(configPath)

        self.model = build_model(
            name=cfg.model.name,
            num_classes=1000,
            loss='am_softmax',
            feature_dim=cfg.model.feature_dim,
            fpn_cfg=cfg.model.fpn,
            pooling_type=cfg.model.pooling_type,
            input_size=(256, 128),
            IN_first=cfg.model.IN_first,
        )
        load_pretrained_weights(self.model, weightPath)
        self.model = self.model.cuda()
        self.model.eval()
        self.preprocess_mean = (torch.tensor([0.485, 0.456, 0.406]) * 255.0).cuda()
        self.preprocess_std = (torch.tensor([0.229, 0.224, 0.225]) * 255.0).cuda()

    def preprocess(self, input_images):
        inps = []
        for image in input_images:
            input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (128, 256)).astype(np.float32)
            input_image = torch.as_tensor(input_image, device='cuda:0').unsqueeze_(0)
            inps.append(input_image)
        images = torch.cat(inps, axis=0)
        images = (images - self.preprocess_mean) / self.preprocess_std
        images = images.permute([0, 3, 1, 2])
        return images
        

    def embed(self, frame, bboxes):
        crops = []
        for bbox in bboxes:
            p1 = [int(bbox.x-bbox.w/2), int(bbox.y-bbox.h/2)]
            p2 = [int(bbox.x+bbox.w/2), int(bbox.y+bbox.h/2)]
            if p1[0] < 0: p1[0] = 0
            if p2[0] > frame.shape[1]: p2[0] = frame.shape[1]
            if p1[1] < 0: p1[1] = 0
            if p2[1] > frame.shape[1]: p2[1] = frame.shape[0]
            crops.append(frame[p1[1]: p2[1], p1[0]: p2[0]])
        embeddings = []
        if len(crops) > 0:
            torch_imgs = self.preprocess(crops)
            with torch.no_grad():
                out = self.model(torch_imgs)
            embeddings = out.cpu().numpy() 
        return bboxes, embeddings, frame 





