from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL

def prepare_roidb(imdb):
    # 给原始roidata添加一些说明性的附加属性
    """Enrich the imdb's roidb by adding some derived quantities that
      are useful for training. This function precomputes the maximum
      overlap, taken over ground-truth boxes, between each ROI and
      each ground-truth box. The class with maximum overlap is also
      recorded.
      """

    print(imdb)
    print(imdb.name)

    roidb = imdb.roidb
    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                 for i in range(imdb.num_images)]
    for i in range(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as dense array for argmax
        # gt_overlaps是一个box_num*classes_num的矩阵，应该是每个box在不同类别的得分
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        # 每个box在所有类别的得分最大值，box_num长度
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        # 每个box的得分最高所对应的类，box_num长度
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # 做检查，检查是否为背景max_overlaps == 0意味着背景，否则非背景
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert  all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should noe be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

















