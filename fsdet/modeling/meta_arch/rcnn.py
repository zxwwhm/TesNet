# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn
import sys
import contextlib
import io

from pycocotools.coco import COCO
from fsdet.structures import ImageList
from fsdet.utils.logger import log_first_n
from fsdet.structures import Boxes, Instances, pairwise_iou
from fsdet.utils.events import get_event_storage
from ..matcher import Matcher
from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..proposal_generator.rpnnovel import build_novel_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .rot import *

import json
import random
from torchvision import transforms
import cv2


__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


def gen_sub_box(bbox):
    '''
    :param bbox[x,y,x+w,y+h]
    :return:sub_bbox:[sub_x,sub_y,sub_x+sub_w,sub_y+sub_h]
    '''
    gt_x = bbox[0]
    gt_y = bbox[1]
    gt_w = bbox[2]-bbox[0]
    gt_h = bbox[3]-bbox[1]


    crop_x = random.uniform(0.001, gt_w / 5)
    crop_y = random.uniform(0.001, gt_h / 5)

    crop_w = random.uniform(gt_w / 2, gt_w-1)
    crop_h = random.uniform(gt_h / 2, gt_h-1)

    #before
    crop_w = min(crop_w, gt_w - crop_w - 0.001)
    crop_h = min(crop_h, gt_h - crop_h - 0.001)

    #now
    # crop_w = min(crop_w, gt_w - crop_x - 0.001)
    # crop_h = min(crop_h, gt_h - crop_y - 0.001)


    # area thread
    # thread = random.uniform(0.7,0.99)
    # iou = gt_w*gt_h* thread
    # # sub_w = random.uniform(iou / gt_h, gt_w)
    # sub_w = random.uniform(gt_w*thread-0.0001, gt_w-0.0001)
    # sub_h = iou / sub_w
    # sub_x_max = abs(gt_w - sub_w)
    # sub_y_max = abs(gt_h - sub_h)
    # sub_x = random.uniform(gt_x-0.0001, gt_x + sub_x_max-0.0001)
    # sub_y = random.uniform(gt_y-0.0001, gt_y + sub_y_max-0.0001)
    # # sub_box = [sub_x, sub_y, sub_x, sub_y]
    # sub_box = [sub_x, sub_y, sub_x+sub_w, sub_y+sub_h]

    crop_x = gt_x + crop_x
    crop_y = gt_y + crop_y
    sub_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
    thread = (crop_w * crop_h) / (gt_w * gt_h)
    print(thread)
    return sub_box, thread



@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        # print(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.duplicate=cfg.duplicate
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.novel_rpn=build_novel_generator(cfg,self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.proposal_matcher = Matcher(
            thresholds=[0.5],
            labels=[0, 1],
            allow_low_quality_matches=False,
        )

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            # for p in self.novel_rpn.parameters():
            #     p.requires_grad=False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')
        # for p in "roi_heads.box_predictor.cls_score"
        # print(self.proposal_generator.state_dict().keys(), self.proposal_generator.state_dict()['anchor_generator.cell_anchors.0'])
        # print(self.novel_rpn.state_dict().keys())
        
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator and self.duplicate:
            proposals1, proposal_losses1 = self.proposal_generator(images, features, gt_instances)
            proposals2,proposal_losses2=self.novel_rpn(images,features,gt_instances)
            proposals=[]
            for i in range(len(proposals1)):
                res=[]
                res.append(proposals1[i])
                res.append(proposals2[i])
                proposal=Instances.cat(res)
                proposals.append(proposal)

            proposal_losses={}
            for k in proposal_losses1.keys():
                proposal_losses[k]=proposal_losses1[k]+proposal_losses2[k]
        elif self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        t_N  =  (random.randint(400,500)) // (len(gt_instances))
        #visualization proposals and gt
        aaa = transforms.ToPILImage()(images[0].detach().cpu()).convert('RGB')
        view_image = cv2.cvtColor(np.asarray(aaa),cv2.COLOR_RGB2BGR)
        view_proposals = proposals[0].proposal_boxes.tensor.cpu()
        for i in range(5):
            view_image=cv2.rectangle(view_image, (int(view_proposals[i][0]), int(view_proposals[i][1])), (int(view_proposals[i][2]), int(view_proposals[i][3])), (0, 0, 255), 1)
        cv2.imwrite("/home/jovyan/BY1706159/gitcode/few-shot-object-detection_2/keshihua/test2.jpg", view_image)
        
        
        for instance in range(len(gt_instances)):
            gt_class =gt_instances[instance].gt_classes.detach().cpu().float()
            groundth_box = gt_instances[instance].gt_boxes.tensor.cpu().numpy().tolist()
            new_proposals = []
            new_objectness_logits =[]
            for i in range(t_N):
                sub_box,thread = gen_sub_box(groundth_box[0])
                new_objectness_logits.append(thread)
                new_proposals.append(sub_box)
            new_proposals.append(groundth_box[0])
            new_objectness_logits.append(float(1))
            proposals[instance].remove('proposal_boxes')
            proposals[instance].remove('objectness_logits')
            proposals[instance].set('proposal_boxes',Boxes(torch.tensor(new_proposals).cuda(torch.device("cuda"))))
            proposals[instance].set('objectness_logits',torch.Tensor(new_objectness_logits).cuda(torch.device("cuda")))
            
            
            proposals[instance].set('gt_classes',gt_instances[instance].gt_classes.detach().repeat(t_N+1))
            proposals[instance].set('gt_boxes',Boxes(gt_instances[instance].gt_boxes.tensor.detach().repeat(t_N+1,1)))

            proposals[instance].set('objectness_logits',torch.ones(proposals[instance].proposal_boxes.tensor.shape[0]).cuda(torch.device("cuda")))
            proposals[instance].set('objectness_logits',gt_class.cuda(torch.device("cuda")))
        
        match_quality_matrix = pairwise_iou(
                gt_instances[0].gt_boxes, proposals[0].proposal_boxes
            )
        matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
        positive_num=np.count_nonzero(matched_labels.cpu())
        # base_features_path ="tools/baseperclass_new.pickle"
        # with open(base_features_path, 'rb') as f:
        #     data = pickle.load(f)
        
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        return losses,positive_num

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training


        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # imgid=batched_inputs[0]['image_id']
        if detected_instances is None:
            if self.proposal_generator and self.duplicate:
                proposals1, _ = self.proposal_generator(images, features, None)
                proposals2,_=self.novel_rpn(images,features,None)
                proposals=[]
                for i in range(len(proposals1)):
                    res=[]
                    res.append(proposals1[i])
                    res.append(proposals2[i])
                    proposal=Instances.cat(res)
                    proposals.append(proposal)
                
            elif self.proposal_generator:    
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            
            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results, proposals
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # print("features",features.shape)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
