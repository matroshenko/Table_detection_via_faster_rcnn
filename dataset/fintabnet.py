import os
import json
from collections import defaultdict

import numpy as np

from dataset import DatasetRegistry, DatasetSplit


class MarkupError(Exception):
  pass


class FinTabNet(DatasetSplit):
    def __init__(self, basedir, split):
        assert split in ['train', 'val', 'test']

        raw_file_name_to_boxes_list = self._build_file_name_to_boxes_list(basedir, split)
        self._file_name_to_boxes_list = self._filter_items_with_invalid_boxes(
            raw_file_name_to_boxes_list)

    def training_roidbs(self):
        result = []
        for file_name, bboxes in self._file_name_to_boxes_list.items():
            item = {}
            item['file_name'] = file_name

            N = len(bboxes)
            item["boxes"] = np.asarray(bboxes, dtype=np.float32)
            item["class"] = np.ones((N,), dtype=np.int32)
            item["is_crowd"] = np.zeros((N,), dtype=np.int8)
            result.append(item)

        return result

    def inference_roidbs(self):
        result = []
        for file_path, _ in self._file_name_to_boxes_list.items():
            item = {}
            item['file_name'] = file_path
            company_name, year, file_name = file_path.split('/')[-3:]
            item['image_id'] = company_name + '_' + year + '_' + os.path.splitext(file_name)[0]

        return result

    def _build_file_name_to_boxes_list(self, basedir, split):
        jsonl_file_name = os.path.join(
            basedir, 'FinTableNet_1.0.0_table_{}.jsonl'.format(split))

        file_name_to_tables_list = defaultdict(list)   
        with open(jsonl_file_name, 'r') as f:
            for line in f:
                sample = json.loads(line)
                file_name = os.path.join(
                    self._basedir, 'jpg', os.path.splitext(sample['filename']) + '.jpg')
                bbox = sample['bbox']
                file_name_to_tables_list[file_name].append(bbox)
        return file_name_to_tables_list

    def _filter_items_with_invalid_boxes(self, file_name_to_boxes_list):
        result = {}
        for file_name, bboxes in file_name_to_boxes_list.items():
            try:
                self._check_valid_bboxes(bboxes)
                self._check_no_intersection(bboxes)

                result[file_name] = bboxes
            except MarkupError:
                continue
        return result

    def _check_valid_bboxes(self, bboxes):
        for bbox in bboxes:
            if not self._is_valid_bbox(bbox):
                raise MarkupError

    def _is_valid_bbox(self, bbox):
        left, top, right, bottom = bbox
        return 0 <= left and left < right and 0 <= top and top < bottom

    def _check_no_intersection(self, bboxes):
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                if self._do_intersect(bboxes[i], bboxes[j]):
                    raise MarkupError

    def _do_intersect(self, bbox1, bbox2):
        left1, top1, right1, bottom1 = bbox1
        left2, top2, right2, bottom2 = bbox2
        return (left1 < right2 and left2 < right1
            and top1 < bottom2 and top2 < bottom1)


def register_fintabnet(basedir):
    for split in ["train", "val", "test"]:
        name = "fintabnet_" + split
        DatasetRegistry.register(name, lambda x=split: FinTabNet(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "table"])