#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import tempfile
import time
from collections import ChainMap
from loguru import logger
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized


class VOCEvaluator:
    """
    VOC AP Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)
        self.num_atts = 3  # att

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        pos_cnt = []
        pos_tol = []
        neg_cnt = []
        neg_tol = []

        accu = 0.0
        prec = 0.0
        recall = 0.0
        tol = 0.0

        for it in range(self.num_atts):
            pos_cnt.append(0)
            pos_tol.append(0)
            neg_cnt.append(0)
            neg_tol.append(0)
        
        for cur_iter, (imgs, target, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
              
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.update(self.convert_to_voc_format(outputs, info_imgs, ids))
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.att
            for i in range(target.shape[0]):  #batch, 50, 8
                # print(outputs[i].shape)   (pre_num,8)
                batch_num = 0
                one_batch_target = target[i,:,:]
                sum_target = torch.sum(one_batch_target, dim=1)
                for j in range(50):
                    if sum_target[j]!=0:
                        batch_num += 1
                
                output = outputs[i]
                # att_pre = output[:, 7]
                tol = tol + batch_num
                # att_bbox = output[:, 0:4]
                match_id = 0
                match = 10000
                for att_id in range(self.num_atts):
                    for num_id in range(batch_num):
                        if target[i, num_id, 5+att_id]==1:
                            pos_tol[att_id] = pos_tol[att_id]+1

                            for bb in range(output.shape[0]):
                                abs_match = torch.sum(torch.abs(output[bb,0:4]-target[i,num_id,0:4].cuda()))
                                if abs_match<match:
                                    match_id = bb
                                    match = abs_match
                            
                            # if output[match_id, 7] == att_id and (match_id< num_id or match_id==num_id):
                            if output[match_id,7]==att_id:
                                pos_cnt[att_id] = pos_cnt[att_id] + 1
                            match = 10000

                        if target[i, num_id, 5+att_id]==0:
                            neg_tol[att_id] = neg_tol[att_id] + 1

                            for bb in range(output.shape[0]):
                                abs_match = torch.sum(torch.abs(output[bb, 0:4] - target[i, num_id, 0:4].cuda()))
                                # print(abs_match)
                                if abs_match < match:
                                    match_id = bb
                                    match = abs_match
                            if output[match_id, 7] != att_id:
                                neg_cnt[att_id] = neg_cnt[att_id] + 1  
                            match =  10000  
                match=10000
                for num_id in range(batch_num):
                    tp = 0
                    fn = 0
                    fp = 0
                    for att_id in range(self.num_atts):
                        for bb in range(output.shape[0]):
                            abs_match = torch.sum(torch.abs(output[bb, 0:4] - target[i, num_id, 0:4].cuda()))
                            if abs_match < match:
                                match_id = bb
                                match = abs_match
                        # print("att_id  {}".format(att_id))
                        # print("output[match_id, 7]    {}".format(output[match_id, 7]))
                        # print("target[i, num_id, 5 + att_id]  {}".format(target[i, num_id, 5 + att_id]))

                        if output[match_id, 7]==att_id and target[i, num_id, 5 + att_id] == 1:
                            tp = tp + 1
                        elif output[match_id, 7]==att_id and target[i, num_id, 5 + att_id] == 0:
                            fp = fp +1
                        elif output[match_id, 7] != att_id and target[i, num_id, 5 + att_id] == 1:
                            fn = fn + 1
                        match = 10000
                     
                    #print('{}:{}:{}'.format(tp,fp,fn))
                    
                    if tp + fn + fp != 0:
                        accu = accu +  1.0 * tp / (tp + fn + fp)
                    if tp + fp != 0:
                        prec = prec + 1.0 * tp / (tp + fp)
                    if tp + fn != 0:
                        recall = recall + 1.0 * tp / (tp + fn)

        print('=' * 100)
        print('\t Attr       \tp_true/n_true        \tp_tol/n_tol          \tp_pred/n_pred        \tcur_mA')
        mA = 0.0
        for it in range(self.num_atts):
            cur_mA = ((1.0*pos_cnt[it]/pos_tol[it]) + (1.0*neg_cnt[it]/neg_tol[it])) / 2.0
            mA = mA + cur_mA
            print('\t#{:2}: {:4}\{:4}\t {:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(
                    it, pos_cnt[it], neg_cnt[it], pos_tol[it], neg_tol[it], (pos_cnt[it]+neg_tol[it]-neg_cnt[it]), (neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
        mA = mA / self.num_atts
        print('\t' + 'mA:        '+str(mA))

        
        accu = accu / tol
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall)
        print('\t' + 'Accuracy:  '+str(accu))
        print('\t' + 'Precision: '+str(prec))
        print('\t' + 'Recall:    '+str(recall))
        print('\t' + 'F1_Score:  '+str(f1))
        print('=' * 100)


            
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        all_boxes = [
            [[] for _ in range(self.num_images)] for _ in range(self.num_classes)
        ]
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images)
            )
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.dataloader.dataset.evaluate_detections(
                all_boxes, tempdir
            )
            return mAP50, mAP70, info
