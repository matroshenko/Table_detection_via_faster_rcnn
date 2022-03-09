import os
import json

import numpy as np

from dataset import DatasetRegistry, DatasetSplit


class MarkupError(Exception):
  pass


class Table(object):
  def __init__(self, id, bbox):
    self.id = id
    self.bbox = bbox


class FinTabNet(DatasetSplit):
    def __init__(self, basedir, split):
        assert split in ['train', 'val', 'test']

        self._basedir = basedir
        self._split = split

    def training_roidbs(self):
        jsonl_file_name = os.path.join(
            self._basedir, 'FinTableNet_1.0.0_table_{}.jsonl'.format(self._split))

        file_name_to_tables_list = {}    
        with open(jsonl_file_name, 'r') as f:
            for line in f:
                sample = json.loads(line)
                file_name = os.path.splitext(sample['filename']) + '.jpg'
                table_id = sample['table_id']
                bbox = sample['bbox']
                file_name_to_tables_list[file_name].append(Table(table_id, bbox))

        result = []
        for file_name, tables_list in file_name_to_tables_list.items():
            item = {}
            item['file_name'] = os.path.join(self._basedir, 'jpg', file_name)
            bboxes = [table.bbox for table in tables_list]

            try:
                self._check_valid_bboxes(bboxes)
                self._check_no_intersection(bboxes)
            except MarkupError:
                continue

            N = len(bboxes)
            item["boxes"] = np.asarray(bboxes, dtype=np.float32)
            item["class"] = np.ones((N,), dtype=np.int32)
            item["is_crowd"] = np.zeros((N,), dtype=np.int8)
            result.append(item)

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