import torch
import torchvision
import torchvision.models.detection.mask_rcnn as mask_rcnn
import torchvision.transforms as transforms
from PIL import Image
from ..transforms import transform, scale_factor, scale_size


class MaskRCNN(mask_rcnn.MaskRCNN):
    ''' 
        For MaskRCNN we simply use torchvision implementation with weight transformation 
        from mmdetection.
    '''

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None):

        super().__init__(
            backbone,
            num_classes,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            rpn_batch_size_per_image=rpn_batch_size_per_image,
            rpn_positive_fraction=rpn_batch_size_per_image,
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_bg_iou_thresh,
            box_batch_size_per_image=box_batch_size_per_image,
            box_positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor)

    def detect(self, image):

        assert isinstance(image, str) or isinstance(image, Image.Image)

        image = Image.open(image) if isinstance(image, str) else image

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])

        return self(transform(image).unsqueeze(0))


class MaskRCNNMapper():
    '''
        Maps mmdetection state dict to torchvision model.
    '''

    def __call__(self, state_dict):

        def dotty(*elms):
            return '.'.join(elms)

        def m_to_t(k):

            tokens = k.split('.')
            if 'backbone' in k:
                return dotty('backbone.body', *k.split('.')[1:])
            elif 'neck.lateral_convs' in k:
                # 'neck.lateral_convs.0.conv.weight' > 'backbone.fpn.inner_blocks.0.weight'
                return dotty('backbone.fpn.inner_blocks', tokens[-3], tokens[-1])
            elif 'neck.fpn_convs' in k:
                return dotty('backbone.fpn.layer_blocks', tokens[-3], tokens[-1])
            elif 'rpn_head.rpn_conv' in k:
                return dotty('rpn.head.conv', tokens[-1])
            elif 'rpn_head.rpn_cls' in k:
                return dotty('rpn.head.cls_logits', tokens[-1])
            elif 'rpn_head.rpn_reg' in k:
                return dotty('rpn.head.bbox_pred', tokens[-1])
            elif 'bbox_head.shared_fcs.0' in k:
                return dotty('roi_heads.box_head.fc6', tokens[-1])
            elif 'bbox_head.shared_fcs.1' in k:
                return dotty('roi_heads.box_head.fc7', tokens[-1])
            elif 'bbox_head.fc_cls' in k:
                return dotty('roi_heads.box_predictor.cls_score', tokens[-1])
            elif 'bbox_head.fc_reg' in k:
                return dotty('roi_heads.box_predictor.bbox_pred', tokens[-1])
            elif 'mask_head.convs' in k:
                return dotty('roi_heads.mask_head.mask_fcn' + str(1+int(tokens[-3])), tokens[-1])
            elif 'mask_head.upsample' in k:
                return dotty('roi_heads.mask_predictor.conv5_mask', tokens[-1])
            elif 'mask_head.conv_logits' in k:
                return dotty('roi_heads.mask_predictor.mask_fcn_logits', tokens[-1])
            return k

        return {m_to_t(k): v for k, v in state_dict.items() if 'num_batches' not in k}
