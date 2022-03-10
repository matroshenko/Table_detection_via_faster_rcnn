import os
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from dataset import DatasetRegistry, DatasetSplit
from dataset.f_measure_calculator import FMeasureCalculator

__all__ = ['register_fintabnet']


class MarkupError(Exception):
  pass


class FinTabNet(DatasetSplit):
    def __init__(self, basedir, split):
        assert split in ['train', 'val', 'test']

        index = self._build_index(basedir, split)
        self._index = self._filter_index(
            index)

    def training_roidbs(self):
        result = []
        for _, (file_name, bboxes) in self._index.items():
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
        for image_id, (file_path, _) in self._index.items():
            item = {}
            item['file_name'] = file_path
            item['image_id'] = image_id
            result.append(item)

        return result

    def eval_inference_results(self, results, output=None):
        results_index = defaultdict(list)
        for item in results:
            results_index[item['image_id']].append(item['bbox'])

        fscore_calculators = [FMeasureCalculator(0.5), FMeasureCalculator(0.75)]

        for image_id, (_, markup_bboxes) in self._index.items():
            predicted_bboxes = results_index[image_id]
            y_true = np.asarray(markup_bboxes, dtype=np.float32)
            y_pred = np.asarray(predicted_bboxes, dtype=np.float32)
            for calculator in fscore_calculators:
                calculator.update_state(y_true, y_pred)

        return {
            'F1@0.5': fscore_calculators[0].result(),
            'F1@0.75': fscore_calculators[1].result() 
            }

    def _build_index(self, basedir, split):
        jsonl_file_name = os.path.join(
            basedir, 'FinTabNet_1.0.0_table_{}.jsonl'.format(split))

        result = defaultdict(list)   
        with open(jsonl_file_name, 'r') as f:
            for line in f:
                sample = json.loads(line)
                file_name = os.path.join(
                    basedir, 'jpg', os.path.splitext(sample['filename'])[0] + '.jpg')
                image_height = Image.open(file_name).size[1]

                bbox = self._get_bbox(image_height, sample['bbox'])
                result[file_name].append(bbox)
        return result

    def _get_bbox(self, image_height, bbox):
        left = bbox[0]
        top = image_height - bbox[3]
        right = bbox[2]
        bottom = image_height - bbox[1]
        return [left, top, right, bottom]

    def _filter_index(self, index):
        result = {}
        for file_path, bboxes in index.items():
            try:
                self._check_valid_bboxes(bboxes)
                self._check_no_intersection(bboxes)

                result[self._get_image_id(file_path)] = (file_path, bboxes)
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

    def _get_image_id(self, file_path):
        company_name, year, file_name = file_path.split('/')[-3:]
        return company_name + '_' + year + '_' + os.path.splitext(file_name)[0]


def register_fintabnet(basedir):
    for split in ["train", "val", "test"]:
        name = "fintabnet_" + split
        DatasetRegistry.register(name, lambda x=split: FinTabNet(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "table"])