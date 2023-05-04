import os
import os.path as osp
import yaml, json
import argparse
import logging
from datetime import datetime
import random
from tqdm import tqdm
from typing import Dict, Any
from PIL import Image
import math
from functools import lru_cache

import numpy as np
from collections import defaultdict, Counter

from training_utils import load_config
from scene_synthesis.datasets.threed_front import ThreedFront, CachedRoom, CachedThreedFront
from scene_synthesis.datasets.threed_front_scene import Asset, ModelInfo, Room, ThreedFutureModel
from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder

from num2words import num2words
import nltk
# nltk.download('cmudict')
from nltk.corpus import cmudict
from torchvision.transforms import Compose
"""
Taken from https://stackoverflow.com/questions/20336524/verify-correct-use-of-a-and-an-in-english-texts-python
"""
def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article

def bbox_centers_size_to_vec(box_center, box_size):
    '''
    input: {'min': [1,2,3], 'max': [4,5,6]}
    output: [1,2,3,4,5,6]
    '''
    # print(f'box_center: {box_center}')
    # print(f'box_size: {box_size}')
    box_min = box_center - box_size/2
    box_max = box_center + box_size/2
    # print(f'box_min: {box_min}')
    # print(f'box_max: {box_max}')
    # concatenate two list
    return list(box_min) + list(box_max)

def compute_spatial_relations(box1, box2):
    center1 = np.array([(box1[0] + box1[3]) / 2, (box1[1] + box1[4]) / 2, (box1[2] + box1[5]) / 2])
    center2 = np.array([(box2[0] + box2[3]) / 2, (box2[1] + box2[4]) / 2, (box2[2] + box2[5]) / 2])

    # random relationship
    sx0, sy0, sz0, sx1, sy1, sz1 = box1
    ox0, oy0, oz0, ox1, oy1, oz1 = box2
    d = center1 - center2
    theta = math.atan2(d[2], d[0])  # range -pi to pi

    distance = (d[2]**2 + d[0]**2)**0.5
    
    # "on" relationship
    p = None
    if center1[0] >= box2[0] and center1[0] <= box2[3]:
        if center1[2] >= box2[2] and center1[2] <= box2[5]:
            delta1 = center1[1] - center2[1]
            delta2 = (box1[4] - box1[1] + box2[4] - box2[1]) / 2
            if 0 <(delta1 - delta2) < 0.05:
                p = 'on'
            elif 0.05 < (delta1 - delta2):
                p = 'above'
        return p, distance

    # eliminate relation in vertical axis now
    if abs(d[1]) > 0.5:
        return p, distance

    area_s = (sx1 - sx0) * (sz1 - sz0)
    area_o = (ox1 - ox0) * (oz1 - oz0)
    ix0, ix1 = max(sx0, ox0), min(sx1, ox1)
    iz0, iz1 = max(sz0, oz0), min(sz1, oz1)
    area_i = max(0, ix1 - ix0) * max(0, iz1 - iz0)
    iou = area_i / (area_s + area_o - area_i)
    touching = 0.0001 < iou < 0.5

    if sx0 < ox0 and sx1 > ox1 and sz0 < oz0 and sz1 > oz1:
        p = 'surrounding'
    elif sx0 > ox0 and sx1 < ox1 and sz0 > oz0 and sz1 < oz1:
        p = 'inside'
    # 60 degree intervals along each direction
    elif theta >= 5 * math.pi / 6 or theta <= -5 * math.pi / 6:
        p = 'right touching' if touching else 'left of'
    elif -2 * math.pi / 3 <= theta < -math.pi / 3:
        p = 'behind touching' if touching else 'behind'
    elif -math.pi / 6 <= theta < math.pi / 6:
        p = 'left touching' if touching else 'right of'
    elif math.pi / 3 <= theta < 2 * math.pi / 3:
        p = 'front touching' if touching else 'in front of'

    return p, distance

class CachedTextRoom(object):
    def __init__(
        self,
        scene_id,
        uids,
        jids,
        scene_uid,
        scene_type,
        room_layout,
        floor_plan_vertices,
        floor_plan_faces,
        floor_plan_centroid,
        class_labels,
        translations,
        sizes,
        angles,
        image_path,
        cat_name_seq = None,
        relations = None,
        description = None
    ):
        self.scene_id = scene_id
        self.uids = uids
        self.jids = jids
        self.scene_uid = scene_uid
        self.scene_type = scene_type
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path
        self.cat_name_seq = cat_name_seq
        self.relations = relations
        self.description = description

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]
    

class GetFreqenciedCategory(object):
    def __init__(self, path_to_model_info:str) -> None:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        self.model_info_dict = mf.model_info

    def __call__(self, scene_boxes_sample: CachedTextRoom, **kwds: Any) -> CachedTextRoom:
        # logger.info('sorting ' + str(scene_boxes_sample.scene_uid))
        model_uids_lst = scene_boxes_sample.uids
        model_jids_lst = scene_boxes_sample.jids
        model_translations_lst = scene_boxes_sample.translations
        model_sizes_lst = scene_boxes_sample.sizes
        model_angles_lst = scene_boxes_sample.angles

        model_frequency_dict = dict(Counter(model_jids_lst))
        model_frequency_dict = dict(sorted(model_frequency_dict.items(), key=lambda x: x[1], reverse=True))
        # print(f'model_frequency_dict: {model_frequency_dict}')

        ordered_model_jids_lst = []
        ordered_model_uids_lst = []
        ordered_model_translations_lst = []
        ordered_model_sizes_lst = []
        ordered_model_angles_lst = []
        ordered_category_name_lst = []

        for key, value in model_frequency_dict.items():
            model_jid = key
            # get model_jid index in model_jids_lst
            raw_idx_lst = [idx for idx,jid in enumerate(model_jids_lst) if jid==key]
            assert(value == len(raw_idx_lst))
            for model_idx in raw_idx_lst:
                ordered_model_uids_lst.append(model_uids_lst[model_idx])
                ordered_model_jids_lst.append(model_jids_lst[model_idx])
                ordered_model_translations_lst.append(model_translations_lst[model_idx])
                ordered_model_sizes_lst.append(model_sizes_lst[model_idx])
                ordered_model_angles_lst.append(model_angles_lst[model_idx])
                # print(self.model_info_dict[model_jid].category)
                ordered_category_name_lst.append(self.model_info_dict[model_jid].category)

        scene_boxes_sample.uids = ordered_model_uids_lst
        scene_boxes_sample.jids = ordered_model_jids_lst
        scene_boxes_sample.translations = ordered_model_translations_lst
        scene_boxes_sample.sizes = ordered_model_sizes_lst
        scene_boxes_sample.angles = ordered_model_angles_lst
        scene_boxes_sample.cat_name_seq = ordered_category_name_lst
        logger.info('furniture jids: ' + ' '.join(ordered_model_jids_lst))
        logger.info('furniture catergories: ' + ' '.join(ordered_category_name_lst))
        return scene_boxes_sample

class AddRelationAmongObjects(object):
    def __init__(self, path_to_model_info:str) -> None:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        self.model_info_dict = mf.model_info

    def __call__(self, scene_boxes_sample: CachedTextRoom) -> CachedTextRoom:
        relations_lst = []
        num_objs = len(scene_boxes_sample.jids)

        # print(f'original box sizes: {scene_boxes_sample.sizes}')
        # # choose 3 objects by their sizes
        # ordered_box_size_lst = scene_boxes_sample.sizes
        # ordered_box_size_lst = sorted(ordered_box_size_lst, key=lambda x: np.linalg.norm(np.array(x)), reverse=True)
        # print(f'sorted box sizes: {ordered_box_size_lst}')

        # first_desc_str = 'The bedroom has '
        # # first, random 3 objects' relations
        # first_choice_lst = np.random.randint(len(scene_boxes_sample.jids), size=3)
        # first_chosen_object_jid_lst = []
        # for idx in first_choice_lst:
        #     model_jid = scene_boxes_sample.jids[idx]
        #     # use different kinds objects
        #     if model_jid not in first_chosen_object_jid_lst:
        #         first_chosen_object_jid_lst.append(model_jid)
        #     else:
        #         continue

        #     model_cat_str = self.model_info_dict[model_jid].category
        #     model_style_str = self.model_info_dict[model_jid].style
        #     frequency_dict = Counter(scene_boxes_sample.jids)
        #     model_freq_str = str(frequency_dict[model_jid])
        #     first_desc_str += model_freq_str + ' ' + model_style_str + ' ' + model_cat_str
        # print(f'first_desc_str: {first_desc_str}')

        for idx, this_box_jid in enumerate(scene_boxes_sample.jids):

            choices_lst = [other for other in range(num_objs) if other < idx]
            for other_idx in choices_lst:
                this_box_center = scene_boxes_sample.translations[idx]
                this_box_size = scene_boxes_sample.sizes[idx]
                box1 = bbox_centers_size_to_vec(this_box_center, this_box_size)
                other_box_center = scene_boxes_sample.translations[other_idx]
                other_box_size = scene_boxes_sample.sizes[other_idx]
                box2 = bbox_centers_size_to_vec(other_box_center, other_box_size)

                # print(scene_boxes_sample.jids[idx], f'box1: {box1}')
                # print(scene_boxes_sample.jids[other_idx], f'box2: {box2}')

                relation_str, distance = compute_spatial_relations(box1, box2)
                if relation_str is not None:
                    relation = (idx, relation_str, other_idx, distance)
                    relations_lst.append(relation)
            
        scene_boxes_sample.relations = relations_lst
        return scene_boxes_sample

def clean_obj_name(name):
    return name.replace('_', ' ')

class AddDescriptions(object):
    '''
    Add text descriptions to each scene
    sample['description'] = str is a sentence
    eg: 'The room contains a bed, a table and a chair. The chair is next to the window'
    '''
    def __init__(self):
        self.b_first_3_furniture_unique = False

    def __call__(self, scene_boxes_sample:CachedTextRoom):
        sentences = []
        # clean object names once
        obj_names = list(map(clean_obj_name, scene_boxes_sample.cat_name_seq))
        # objects that can be referred to
        refs = []
        # TODO: handle commas, use "and"
        # TODO: don't repeat, get counts and pluralize
        # describe the first 2 or 3 objects
        # first_n = random.choice([2, 3])
        first_n = 3
        first_n_names = []
        if self.b_first_3_furniture_unique:
            for idx in range(len(obj_names)):
                if obj_names[idx] not in first_n_names:
                    first_n_names.append(obj_names[idx])
                if len(first_n_names) == first_n:
                    break
        else:
            first_n_names = obj_names[:first_n] 

        first_n_counts = Counter(first_n_names)

        s = 'The room has '
        for ndx, name in enumerate(sorted(set(first_n_names), key=first_n_names.index)):
            if ndx == len(set(first_n_names)) - 1 and len(set(first_n_names)) >= 2:
                s += "and "
            if first_n_counts[name] > 1:
                s += f'{num2words(first_n_counts[name])} {name}s '
            else:
                s += f'{get_article(name)} {name} '
            if ndx == len(set(first_n_names)) - 1:
                s += ". "
            if ndx < len(set(first_n_names)) - 2:
                s += ', '
        sentences.append(s)
        refs = set(range(first_n))

        # for each object, the "position" of that object within its class
        # eg: sofa table table sofa
        #   -> 1    1    2      1
        # use this to get "first", "second"

        seen_counts = defaultdict(int)
        in_cls_pos = [0 for _ in obj_names]
        for ndx, name in enumerate(first_n_names):
            seen_counts[name] += 1
            in_cls_pos[ndx] = seen_counts[name]

        for ndx in range(1, len(obj_names)):
            # higher prob of describing the 2nd object
            prob_thresh = 0.1
                
            if random.random() > prob_thresh:
                # possible backward references for this object
                possible_relations = [r for r in scene_boxes_sample.relations \
                                        if (r[0] == ndx) \
                                        and (r[2] in refs) \
                                        and (r[3] < 1.5)]
                if len(possible_relations) == 0:
                    continue
                # now future objects can refer to this object
                refs.add(ndx)

                # if we haven't seen this object already
                if in_cls_pos[ndx] == 0:
                    # update the number of objects of this class which have been seen
                    seen_counts[obj_names[ndx]] += 1
                    # update the in class position of this object = first, second ..
                    in_cls_pos[ndx] = seen_counts[obj_names[ndx]]

                # pick any one
                (n1, rel, n2, dist) = random.choice(possible_relations)
                o1 = obj_names[n1]
                o2 = obj_names[n2]

                # prepend "second", "third" for repeated objects
                if seen_counts[o1] > 1:
                    o1 = f'{num2words(in_cls_pos[n1], ordinal=True)} {o1}'
                if seen_counts[o2] > 1:
                    o2 = f'{num2words(in_cls_pos[n2], ordinal=True)} {o2}'

                # dont relate objects of the same kind
                if o1 == o2:
                    continue

                a1 = get_article(o1)

                if 'touching' in rel:
                    if ndx in (1, 2):
                        s = F'The {o1} is next to the {o2}'
                    else:
                        s = F'There is {a1} {o1} next to the {o2}'
                elif rel in ('left of', 'right of'):
                    if ndx in (1, 2):
                        s = f'The {o1} is to the {rel} the {o2}'
                    else:
                        s = f'There is {a1} {o1} to the {rel} the {o2}'
                elif rel in ('surrounding', 'inside', 'behind', 'in front of', 'on', 'above'):
                    if ndx in (1, 2):
                        s = F'The {o1} is {rel} the {o2}'
                    else:
                        s = F'There is {a1} {o1} {rel} the {o2}'
                s += ' . '
                sentences.append(s)

        # set back into the sample
        scene_boxes_sample.description = sentences
        return scene_boxes_sample


class CachedTextThreedFront(ThreedFront):
    def __init__(self, base_dir, config, scene_ids, transforms=None, model_info_filepath=None):
        self._base_dir = base_dir
        self.config = config
        self.transforms = transforms

        self._parse_train_stats(config["train_stats"])

        self._tags = sorted([
            oi
            for oi in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, oi))  and oi.split("_")[1] in scene_ids
        ])

        self._path_to_rooms = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags
        ])
        rendered_scene = "rendered_scene_256.png"
        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"

        self._path_to_renders = sorted([
            os.path.join(self._base_dir, pi, rendered_scene)
            for pi in self._tags
        ])

        if model_info_filepath is not None:
            # Parse the model info
            mf = ModelInfo.from_file(model_info_filepath)
            self.model_info = mf.model_info

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    @lru_cache(maxsize=32)
    def __getitem__(self, i):
        D = np.load(self._path_to_rooms[i])
        scene_sample = CachedTextRoom(
            scene_id=D["scene_id"],
            uids=D["uids"],
            jids=D["jids"],
            scene_uid=D["scene_uid"],
            scene_type=D["scene_type"],
            room_layout=self._get_room_layout(D["room_layout"]),
            floor_plan_vertices=D["floor_plan_vertices"],
            floor_plan_faces=D["floor_plan_faces"],
            floor_plan_centroid=D["floor_plan_centroid"],
            class_labels=D["class_labels"],
            translations=D["translations"],
            sizes=D["sizes"],
            angles=D["angles"],
            image_path=self._path_to_renders[i]
        )
        if self.transforms:
            scene_sample = self.transforms(scene_sample)
        return scene_sample

    def get_room_params(self, i):
        D = np.load(self._path_to_rooms[i])

        room = self._get_room_layout(D["room_layout"])
        room = np.transpose(room[:, :, None], (2, 0, 1))
        return {
            "room_layout": room,
            "class_labels": D["class_labels"],
            "translations": D["translations"],
            "sizes": D["sizes"],
            "angles": D["angles"]
        }

    def __len__(self):
        return len(self._path_to_rooms)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self), self.n_object_types
        )

    def _parse_train_stats(self, train_stats):
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3]),
            np.array(self._centroids[3:])
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture

def get_root_logger(log_level=logging.INFO, handlers=()):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_filepath", help="Path to config file")
    # args = parser.parse_args()
    # logging.getLogger("trimesh").setLevel(logging.ERROR)
    log_dir = '/home/hkust/fangchuan/codes/text-ATISS/log'
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger = get_root_logger(logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, '{}_gen_text.log').format(time_str))])

    config_filepath = '../config/bedrooms_config.yaml'
    threed_future_model_info_filepath = '/data/dataset/3D_FRONT_FUTURE/3D_Future/3D-FUTURE-model/model_info.json'
    if not osp.exists(config_filepath):
        exit(-1)


    config = load_config(config_filepath)
    config = config["data"]

    print("Processing 3D-FRONT seqs")
    transform_opts = Compose(
        [
            GetFreqenciedCategory(threed_future_model_info_filepath),
            AddRelationAmongObjects(threed_future_model_info_filepath),
            AddDescriptions(),
            # AddStartToken(config),
            # SeqToTensor(),
            # Add_Glove_Embeddings(max_sentences=3),
        ]
    )
    
    split=["train", "val", "test"]
    dataset_type = config["dataset_type"]
    # if "cached" in dataset_type:
    # Make the train/test/validation splits
    splits_builder = CSVSplitsBuilder(config["annotation_file"])
    split_scene_ids = splits_builder.get_splits(split)
    # print(f'split_scene_ids: {split_scene_ids}')

    dataset = CachedTextThreedFront(config["dataset_directory"], 
                                    config=config,
                                    scene_ids=split_scene_ids,
                                    transforms=transform_opts, 
                                    model_info_filepath=threed_future_model_info_filepath)


    for idx, sample in enumerate(tqdm(dataset)):
        logger.info("+ "*10 + str(sample.scene_uid) + " +"*10)
        # print(f'************** {scene_uid} **************')
        desc_str = sample.description
        # print(desc_str)
        logger.info(' '.join(desc_str))
        logger.info('* '*20)