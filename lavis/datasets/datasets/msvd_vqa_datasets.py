"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.data_utils import load_video
import pdb

class MSVDVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt
        # self._add_instance_ids()
        # pdb.set_trace()

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(0, ann['frame_length']-3, self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32) #改的
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video) #改的
        ##################################################################################################################
        keypoints = torch.tensor(np.load(self.vis_root[:-6]+'motion/'+ann['video']+'.npz', allow_pickle=True)['reconstruction'])
        keypoints_list = np.linspace(0,keypoints.shape[0]-1,81, dtype=int)
        keypoints_start_list = keypoints_list[:-1]
        keypoints_end_list = keypoints_list[1:]
        selected_keypoints_index = []
        for start, end in zip(keypoints_start_list,keypoints_end_list):
            if start == end:
                selected_keypoints_index.append(start)
            else:
                selected_keypoints_index.append(np.random.randint(start, end))
        keypoints = keypoints[selected_keypoints_index]
        ##################################################################################################################

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "keypoints":keypoints,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            # "instance_id": ann["instance_id"],
        }
        
    def __len__(self):
        return len(self.question_id_list)

class MSVDVQAEvalDataset(MSVDVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test')

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-4, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32) #改的

            #ten_path=os.path.join(self.vis_root[:-6]+'tensor', ann['video'],"frame{:06d}.npy".format(frame_index + 1))
            #frame=torch.tensor(np.load(ten_path))
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video) #改的

        ##################################################################################################################
        keypoints = torch.tensor(np.load(self.vis_root[:-6]+'motion/'+ann['video']+'.npz', allow_pickle=True)['reconstruction'])
        keypoints_list = np.linspace(0,keypoints.shape[0]-1,81, dtype=int)
        keypoints_start_list = keypoints_list[:-1]
        keypoints_end_list = keypoints_list[1:]
        selected_keypoints_index = []
        for start, end in zip(keypoints_start_list,keypoints_end_list):
            if start == end:
                selected_keypoints_index.append(start)
            else:
                selected_keypoints_index.append(np.random.randint(start, end))
        keypoints = keypoints[selected_keypoints_index]
        ##################################################################################################################
        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "keypoints":keypoints,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            # "instance_id": ann["instance_id"],
        }
