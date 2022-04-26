# Animal Action Recognition Via Skeleton
This is a repo for animal behavior recognition based on skeleton. First, use YOLOX to complete the animal detection, then cut out the area of the corresponding animal and input it to HRNet to complete the skeleton recognition, and finally pack and input the multi frame skeleton information into ST-GCN to complete the animal recognition.

Action classes including: Stand, Walk, Run, Lay, Eat

![img](test2.gif)

## Configure
1、install anaconda(Optional)


2、yolox config, see details at: https://github.com/Megvii-BaseDetection/YOLOX


## Download pretrain weight(Baidu netdisk)

target detetion: 链接：https://pan.baidu.com/s/13xstEeDE0Eh9CsikGETYLQ (gvn7)

skeleton detetion: https://pan.baidu.com/s/10CWV0Qrbn69XXRjygnSdbA (nwtr)

action predict: https://pan.baidu.com/s/188_0wvfiKCQKOCYshb-58g (5ai5)

## Use
We use yolox_s model to complete animal detection

Modifying the weight path of st-gcn in script ./ActionEstLoader.py

```
python tools/top_down_video_det.py -n yolox-s --path <video path> -c <yolox weight> -pose <skeleton detection weight> --save_result --device gpu
```

train your model(use skeleton):
```
cd Actionsrecognition
python train.py
```
## Reference
Target Detection: https://github.com/Megvii-BaseDetection/YOLOX

Skeleton Detection: https://github.com/AlexTheBad/AP-10K

Action Recognition: https://github.com/yysijie/st-gcn

Human Action: https://github.com/GajuuzZ/Human-Falling-Detect-Tracks
