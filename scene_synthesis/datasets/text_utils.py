import math
import random
from typing import Dict, Any
from collections import defaultdict, Counter
import logging

import numpy as np
import nltk
# nltk.download('cmudict')
from nltk.corpus import cmudict
from num2words import num2words

from .threed_front import CachedTextRoom
from .threed_front_scene import ModelInfo


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


class GetFreqenciedCategory(object):
    """Transform dataset into ordered dataset by furniture frequency

    Args:
        object (_type_): _description_
    """
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
        model_cls_labels_lst = scene_boxes_sample.class_labels

        model_frequency_dict = dict(Counter(model_jids_lst))
        model_frequency_dict = dict(sorted(model_frequency_dict.items(), key=lambda x: x[1], reverse=True))
        # print(f'model_frequency_dict: {model_frequency_dict}')

        ordered_model_jids_lst = []
        ordered_model_uids_lst = []
        ordered_model_translations_lst = []
        ordered_model_sizes_lst = []
        ordered_model_angles_lst = []
        ordered_model_cls_labels_lst = []
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
                ordered_model_cls_labels_lst.append(model_cls_labels_lst[model_idx])
                # print(self.model_info_dict[model_jid].category)
                ordered_category_name_lst.append(self.model_info_dict[model_jid].category)

        scene_boxes_sample.uids = ordered_model_uids_lst
        scene_boxes_sample.jids = ordered_model_jids_lst
        scene_boxes_sample.translations = ordered_model_translations_lst
        scene_boxes_sample.sizes = ordered_model_sizes_lst
        scene_boxes_sample.angles = ordered_model_angles_lst
        scene_boxes_sample.class_labels = ordered_model_cls_labels_lst
        scene_boxes_sample.cat_name_seq = ordered_category_name_lst
        # logger.info('furniture jids: ' + ' '.join(ordered_model_jids_lst))
        # logger.info('furniture catergories: ' + ' '.join(ordered_category_name_lst))
        # print('furniture catergories: ' + ' '.join(ordered_category_name_lst))
        return scene_boxes_sample

class AddRelationAmongObjects(object):
    def __init__(self, path_to_model_info:str) -> None:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        self.model_info_dict = mf.model_info

    def __call__(self, scene_boxes_sample: CachedTextRoom) -> CachedTextRoom:
        relations_lst = []
        num_objs = len(scene_boxes_sample.jids)

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

    def  __call__(self, scene_boxes_sample:CachedTextRoom) -> CachedTextRoom:
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
            # TODO: CONSTANT now, should be dynamic adjusted, to get higher prob of describing the 2nd object
            prob_thresh = 0.0
                
            if random.random() < prob_thresh:
                print(f'random.random() < {prob_thresh}: skip this sentence')
            else:
                # possible backward references for this object
                possible_relations = [r for r in scene_boxes_sample.relations \
                                        if (r[0] == ndx) \
                                        and (r[2] in refs) \
                                        and (r[3] < 1.5)]
                # TODO: still problematic, sometimes the possible_relations is empty
                if len(possible_relations) == 0:
                    # print('len possible_relations == 0!!!')
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

        resampled_sentences_lst = []
        if len(sentences) > 3:
            resampled_sentences_lst.append(sentences[0])
            # random selections from the list without repetition
            resampled_sentences_lst += random.sample(sentences[1:], 2)
        else:
            resampled_sentences_lst = sentences
        # set back into the sample
        scene_boxes_sample.description = resampled_sentences_lst
        return scene_boxes_sample

