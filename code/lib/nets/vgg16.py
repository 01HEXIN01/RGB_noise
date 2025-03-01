import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.utils.compact_bilinear_polling import compact_bilinear_pooling_layer
import numpy as np
import cv2

import lib.config.config as cfg
from lib.nets.network import Network


# SE模块
def squeeze_excitation_layer(input_x, out_dim, ratio=16, layer_name="SE_block"):
    with tf.name_scope(layer_name):
        # 全局平均池化
        squeeze = tf.reduce_mean(input_x, axis=[1, 2], keepdims=True)

        # 全连接层：通道数压缩
        excitation = slim.fully_connected(squeeze, out_dim // ratio, activation_fn=tf.nn.relu)

        # 全连接层：通道数恢复
        excitation = slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.sigmoid)

        # 通道加权
        scale = input_x * excitation
    return scale


class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        # 通过这个函数建立神经网络，包括特征提取，池化，训练等
        with tf.variable_scope('vgg_16', 'vgg_16'):

            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            q = [4.0, 12.0, 2.0]
            filter1 = [[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]]
            filter2 = [[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]]
            filter3 = [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
            filter1 = np.asarray(filter1, dtype=float) / q[0]
            filter2 = np.asarray(filter2, dtype=float) / q[1]
            filter3 = np.asarray(filter3, dtype=float) / q[2]
            filter1_90 = np.rot90(filter1, k=1)
            filter1_180 = np.rot90(filter1, k=2)
            filter2_90 = np.rot90(filter2, k=1)
            filter2_180 = np.rot90(filter2, k=2)
            filter3_90 = np.rot90(filter3, k=1)
            filter3_180 = np.rot90(filter3, k=2)

            filters = [[filter1, filter1_90, filter1_180], [filter2, filter2_90, filter2_180], [filter3, filter3_90, filter3_180]]
            filters = np.einsum('klij->ijlk', filters)
            filters = filters.flatten()
            initializer_srm = tf.constant_initializer(filters)
            # 构建一个三通道滤波器，这个滤波器用于图像噪音输入,这里取5x5的滤波器

            net = self.build_head(is_training)
            # 创建一个head，用于RGB stream数据的输入

            net2 = self.build_head_forNoise(is_training, initializer, initializer_srm)
            # 创建一个head，用于noise stream数据的输入

            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training,
                                                                                               initializer)
            # 创建rpn

            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)
            # 创建proposals，提供可能的预选框

            cls_score, cls_prob, bbox_pred = self.build_predictions(net, net2, rois, is_training, initializer,
                                                                    initializer_bbox)
            # object分类以及边框预测的回归，创建predictions，生成并输出最终框

            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["rpn_cls_prob"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred
            self._predictions["rois"] = rois

            self._score_summaries.update(self._predictions)
            # 将预测的结果与得分存储在_predictions[]中并进行更新

            return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic, sess, pretrained_model):
        variables_to_restore = []
        noise_variable = {}
        for v in variables:

            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0' \
                    or v.name == 'vgg_16/cbp_fc6/weights:0' or v.name == 'vgg_16/cbp_fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue

            if v.name == 'vgg_16/conv1/conv1_1/weights:0' or v.name == 'vgg_16/conv1n/conv1n_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        # 让faster rcnn的网络继承分类网络的特征提取权重和分类器的权重，
        # 让网络从一个比较好的起点开始被训练，有利于训练结果的快速收敛。
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                # 定义接受权重的变量，不可被训练
                # cbp_fc6_conv = tf.get_variable("cbp_fc6_conv", [7, 7, 512, 4096], trainable=False)
                # cbp_fc7_conv = tf.get_variable("cbp_fc7_conv", [1, 1, 4096, 4096], trainable=False)
                # noise_conv1_rgb = tf.get_variable("noise_conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                # 定义恢复变量的对象
                restorer_fc.restore(sess, pretrained_model)
                # 恢复这些变量

                print(self._variables_to_fix)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix[
                                                                                                  'vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix[
                                                                                                  'vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_head(self, is_training):

        # 用于构建net网络，作为RGB特征的提取层,提取输入图片的RGB特征

        # layer1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        # net = squeeze_excitation_layer(net, out_dim=64, layer_name="SE_block1")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # layer2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        # net = squeeze_excitation_layer(net, out_dim=128, layer_name="SE_block2")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # layer3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
        net = squeeze_excitation_layer(net, out_dim=256, layer_name="SE_block3")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # layer4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = squeeze_excitation_layer(net, out_dim=512, layer_name="SE_block4")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # layer5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')
        # net = squeeze_excitation_layer(net, out_dim=512, layer_name="SE_block5")
        # 512x14x14
        print(net.shape)
        self._act_summaries.append(net)

        self._layers['head'] = net

        # while(1):
        #     # img = cv2.imread(net)
        #     cv2.imshow('RGB', net)
        #     k = cv2.waitKey(5) & 0xFF
        #     if k == 27:
        #         break
        # cv2.destroyAllWindows()

        return net

    def build_head_forNoise(self, is_training, initializer, initializer_srm):

        def truncate_2(x):
            # 将值定义在（-2，2）之间
            neg = ((x + 2) + abs(x + 2)) / 2 - 2
            return -(2 - neg + abs(2 - neg)) / 2 + 2

        # 用于构建SRM网络，提取图片噪声特征

        # layer SRM

        noisenet = slim.conv2d(self._image, 3, [5, 5], trainable=False, weights_initializer=initializer_srm,
                               activation_fn=None, padding='SAME', stride=1, scope='srm')
        net = truncate_2(noisenet)

        # Noise layer1
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope='conv1n')
        # net = squeeze_excitation_layer(net, out_dim=64, layer_name="SE_block1n")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1n')

        # Noise layer2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope='conv2n')
        # net = squeeze_excitation_layer(net, out_dim=128, layer_name="SE_block2n")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2n')

        # Noise layer3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope='conv3n')
        net = squeeze_excitation_layer(net, out_dim=256, layer_name="SE_block3n")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3n')

        # Noise layer4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope='conv4n')
        net = squeeze_excitation_layer(net, out_dim=512, layer_name="SE_block4n")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4n')

        # Noise layer5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope='conv5n')
        # net = squeeze_excitation_layer(net, out_dim=512, layer_name="SE_block5n")
        # 512x14x14

        # 将net加入summaries
        self._act_summaries.append(net)

        # 将net加入head layer
        self._layers['head2'] = net

        return net

    def build_rpn(self, net, is_training, initializer):

        # 构建anchors
        self._anchor_component()
        # 在network中

        # create rpn net
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        # 与512个3x3的滑动窗口进行卷积

        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer, padding='VALID', activation_fn=None,
                                    scope='rpn_cls_score')

        # Change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")

        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer, padding='VALID', activation_fn=None,
                                    scope='rpn_bbox_pred')
        # 与9x4个1x1的filter卷积，得到anchor box的坐标信息，其实是anchor的偏移量

        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):

        # 继承build_rpn的计算结果继续进行计算

        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            # 获取2000个proposals
            rpn_lables = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_lables]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")

        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, net2, rois, is_training, initializer, initializer_bbox):
        # net是RGB网络，net2是噪声网络
        # crop image rois
        # 裁切图片roi
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        # pool5_flat = slim.flatten(pool5, scope='flatten')
        pool5_forNoise = self._crop_pool_layer(net2, rois, "pool5_forNoise")

        # 将两个网络输入双线性池化层
        cbp = compact_bilinear_pooling_layer(pool5, pool5_forNoise, 512)

        cbp_flat = slim.flatten(cbp, scope='cbp_flatten')

        # fc6 = slim.fully_connected(pool5_flat, 4096, scope='bbox_fc6')
        fc6_cbp = slim.fully_connected(cbp_flat, 4096, scope='fc6')
        if is_training:
            # 如果是训练过程的话，加入一个dropout层
            # fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
            fc6_cbp = slim.dropout(fc6_cbp, keep_prob=0.5, is_training=True, scope='cbp_dropout6')

        # fc7 = slim.fully_connected(fc6, 4096, scope='bbox_fc7')
        fc7_cbp = slim.fully_connected(fc6_cbp, 4096, scope='fc7')
        if is_training:
            # fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
            fc7_cbp = slim.dropout(fc7_cbp, keep_prob=0.5, is_training=True, scope='cbp_dropout7')

        # scores and predictions
        cls_score = slim.fully_connected(fc7_cbp, self._num_classes, weights_initializer=initializer,
                                         trainable=is_training, activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        bbox_prediction = slim.fully_connected(fc7_cbp, self._num_classes * 4, weights_initializer=initializer_bbox,
                                               trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction
