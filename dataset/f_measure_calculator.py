import numpy as np

import utils


class FMeasureCalculator(object):
    def __init__(self, iou_threshold):
        assert iou_threshold >= 0.5
        self._threshold = iou_threshold

        self._matched_objects_count = 0
        self._predicted_objects_count = 0
        self._markup_objects_count = 0

    def update_state(self, y_true, y_pred):
        # We assume here, that all markup objects do not intersect.
        num_of_markup_objects = y_true.shape[0]
        np.testing.assert_array_almost_equal(
            utils.np_box_ops.iou(y_true, y_true), 
            np.eye(num_of_markup_objects))

        iou = utils.np_box_ops.iou(y_true, y_pred)
        matching_counts = np.sum(iou > self._threshold, axis=1)
        # If there is more than 1 predicted object with IOU > threshold,
        # all such objects are treated as false positives.
        matched_objects_count = np.sum(matching_counts == 1)
        
        self._matched_objects_count += matched_objects_count
        self._predicted_objects_count += y_pred.shape[0]
        self._markup_objects_count += num_of_markup_objects

    def result(self):
        assert self._matched_objects_count <= self._predicted_objects_count
        assert self._matched_objects_count <= self._markup_objects_count

        if self._matched_objects_count == 0:
            if self._markup_objects_count == 0:
                return 1
            return 0

        precision = self._matched_objects_count / self._predicted_objects_count
        recall = self._matched_objects_count / self._markup_objects_count

        return 2 * precision * recall / (precision + recall)