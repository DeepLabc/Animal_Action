import argparse
import os
import time
from loguru import logger
import cv2
import torch
from torch import tensor,float32,cat
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, VOC_CLASSES, ATTRIBUTE
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

from Track.Tracker import Detection, Tracker

from ActionsEstLoader import TSSTG

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '0'

def is_in_box(xy, box):
    if box[0] < xy[0] < box[2] and box[1] < xy[1] < box[3]:
        return True
    else:
        return False

def kpt2bbox(kpt, ex=20):
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='', type=str, help="ckpt for eval")
    parser.add_argument("-pose", "--pose_checkpoint", default='', type=str, help="pose ckpt path")
    parser.add_argument('-pose_config', default='./pose/hrnet_w32_ap10k_256_256.py', help='Config file for pose')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=4,
        help='Link thickness for visualization')

    parser.add_argument(
        '--de', default='cuda:0', help='Device used for inference')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        att_names=ATTRIBUTE,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.att_names = att_names    #     
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
       

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_thresh=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        atts = output[:, 7:10]  # bbox,3   
           
        # HRNet
        ret_bbox = []
        np_bbox = bboxes.numpy()
        np_bbox = np_bbox.astype(int)
        np_score = scores.numpy()
        for i in range(len(np_bbox)):
            animal = {}
            info = np.concatenate((np_bbox[i], [np_score[i]]), axis=0)
            animal["bbox"] = info
            ret_bbox.append(animal)
        # SORT
        sort_det = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            cla = cls[i]
            box_conf = output[:,4][i]
            cls_conf = output[:,5][i]
            score = round(float(scores[i]), 4)
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            sort_det.append([x0, y0, x1, y1, box_conf, cls_conf, cla])

        dets = np.array(sort_det)
        dets = torch.from_numpy(dets).cuda()
        
        return ret_bbox,dets, img
        
        # return vis_res

def image_demo(predictor, vis_folder, path, current_time, save_result, args):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.de.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        ret_bbox, dets, att_img = predictor.visual(outputs[0], img_info, predictor.confthre)  #  list  detection bbox
        # test a single image, with a list of bboxes.
        if len(ret_bbox)==0:
            continue
        frame = cv2.imread(image_name)
        pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                frame,
                ret_bbox,
                bbox_thr=0.3,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)  
            # output_layer_names
            # print(pose_results)

            # show the results
        vis_img = vis_pose_result(
                pose_model,
                frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=0.2,
                radius=args.radius,
                thickness=args.thickness,
                show=False)

        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )

 
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, vis_img)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    # tracker init
    max_age = 10
    tracker = Tracker(max_age=max_age, n_init=3)

    # action recognition base on skeleton
    action_pre = TSSTG()

    # animal pose predict:  refer to AP-10k
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.de)
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    while True:
        ret_val, frame = cap.read()
        if ret_val:

            outputs, img_info = predictor.inference(frame)
            if outputs[0]==None:
                if args.save_result:
                    vid_writer.write(frame)
                    continue
            hrnet_bbox, dets, img = predictor.visual(outputs[0], img_info, predictor.confthre)  #  list  detection bbox
        
            detected = dets
            tracker.predict()
                
            for track in tracker.tracks:
                det = tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=float32).cuda()  # 
                detected = cat([detected, det], dim=0) if detected is not None else det

            detections = []
            if detected is not None:
                t1 = time.time()
                poses, returned_outputs = inference_top_down_pose_model(
                        pose_model,
                        frame,
                        hrnet_bbox,
                        bbox_thr=0.1,
                        format='xyxy',
                        dataset=dataset,
                        dataset_info=dataset_info,
                        return_heatmap=False,
                        outputs=None)

                # print('phose inference time:{}'.format(time.time()-t1))
                detections = [Detection(kpt2bbox(ps['keypoints'][:, :2]),
                                  ps['keypoints'],
                                  ps['keypoints'][:, 2].mean()) for ps in poses]

                
            tracker.update(detections)

            # predict each tracker 
            for i, track in enumerate(tracker.tracks):
                j=i
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                # center = track.get_center().astype(int)
                clr = (255, 0, 0)
                
                action='pedding'
                    
                if len(track.keypoints_list) == 10:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_pre.predict(pts, frame.shape[:2])
                    action_name = action_pre.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                   
                    frame = cv2.putText(frame, action, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, clr, 2)    #print action class
          
            vis_img = vis_pose_result(
                        pose_model,
                        frame,
                        poses,
                        dataset=dataset,
                        dataset_info=dataset_info,
                        kpt_score_thr=0.1,
                        radius=args.radius,
                        thickness=args.thickness,
                        show=False)

            if args.save_result:
                    vid_writer.write(vis_img)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, VOC_CLASSES, ATTRIBUTE, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
