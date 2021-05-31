import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import core.utils

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


def detect(save_img=False):

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # TODO ======================= DeepSORT ==================================
    # initialize deep sort
    model_filename = "weights/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )

    # initialize tracker
    tracker = Tracker(metric, max_age=60, max_iou_distance=0.7, n_init=1)
    # TODO ===================================================================

    # TODO Get variables for object detection
    source, weights, view_img, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
    )
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://"))
    )

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:

        # * path -> path of img/video
        # * im0s -> image read from path (could be image (or) frame of a video)
        # * img -> im0s is padded and other changes are made, resulting in img
        # * self.cap -> video capture object

        # convert numpy array to tensor, then convert to gpu/cpu representation
        img = torch.from_numpy(img).to(device)
        # convert to half precision on gpu
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # convert to grayscale ?
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # img.dimension() returns dimension of tensor
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        preds = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t2 = time_synchronized()

        # print(preds)
        # * prediction output -> N x 6 tensor
        # * tensor = (min_x, min_y, max_x, max_y, confidence, class)
        # * sample
        """ 
        [
            tensor(
                [
                    [ 2.87142e+02,  3.37627e+00,  3.54700e+02,  1.01038e+02,  8.83265e-01,  1],
                    [ 2.23392e+02,  1.93529e+02,  3.10253e+02,  3.00007e+02,  8.42104e-01,  1],
                    [ 2.26288e+02,  6.68579e-01,  4.01710e+02,  4.09652e+02,  7.93190e-01,  0],
                    [ 1.45278e+02,  2.99023e+01,  1.97888e+02,  8.51547e+01,  6.85377e-01,  1],
                    [ 1.92674e+02,  1.84224e+02,  3.78924e+02,  4.79458e+02,  6.09018e-01,  0],
                    [ 1.19260e+02,  1.70975e+01,  2.76960e+02,  3.21991e+02,  5.86185e-01,  0],
                    [ 2.04914e-01, -9.77791e-01,  1.15449e+02,  2.00803e+02,  3.66411e-01,  0]
                ]
            )
        ]
        """

        # TODO ====================== rescale coords, convert to xywh format ============================

        # gain = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # for i in range(len(bboxes)):
        #     box = bboxes[i]
        #     box = scale_coords(img.shape[2:], box, im0s.shape).round()
        #     xywh = (xyxy2xywh(torch.tensor(*box).view(1, 4)) / gain).view(-1).tolist()  # normalized xywh
        #     print(xywh)

        # TODO ==========================================================================================

        # Apply Classifier
        if classify:
            preds = apply_classifier(preds, modelc, img, im0s)

        class_names = []
        bboxes = []
        scores = []
        classes = []

        # Process detections
        for i, det in enumerate(preds):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            detections_str = ""

            if len(det):
                # Rescale coords (xyxy) from img1_shape to img0_shape
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    # type(xyxy) => List[torch.Tensor]
                    norm_xyxy = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                    print(f"xyxy: {xyxy}")
                    print(norm_xyxy)

                    # convert to xywh format
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    # add detections of head alone, names = ['person', 'head']
                    if cls.item() == 1:
                        # bboxes.append(norm_xyxy)
                        bboxes.append(xywh)
                        scores.append(conf.item())
                        classes.append(cls.item())
                        class_names.append(names[int(cls.item())])

                # * the bboxes were of the format (x_center, y_center, width, height)
                # * deep sort needs (x_topleft, y_topleft, width, height)
                # * convert coords
                for bbox in bboxes:
                    bbox[0] -= int(bbox[2] / 2)
                    bbox[1] -= int(bbox[3] / 2)

                # TODO ======================== Print results =====================================
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    detections_str = f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                # TODO ===========================================================================

            # TODO ===================== Drawing all detections ===================================
            # if len(det):
            #     # Write results
            #     for *xyxy, conf, cls in reversed(det):

            #         # Write to file
            #         if save_txt:
            #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #             line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
            #             with open(txt_path + '.txt', 'a') as f:
            #                 f.write(('%g ' * len(line)).rstrip() % line + '\n')

            #         # Add bbox to image
            #         if save_img or view_img:
            #             # get confidence value from tensor
            #             conf = conf.item()
            #             label = f'{names[int(cls)]} {conf:.2f}'
            #             if opt.heads or opt.person:
            #                 if 'head' in label and opt.heads:
            #                     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            #                 if 'person' in label and opt.person:
            #                     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            #             else:
            #                 plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            # TODO ==================================================================================

        # TODO ================================ Tracker ====================================
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        classes = np.array(classes)
        class_names = np.array(class_names)

        # encode yolo detections and feed to tracker
        features = encoder(im0, bboxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(
                bboxes, scores, class_names, features
            )
        ]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # non maxima supression again ?
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores
        )

        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            label = f"{class_name}: {track.track_id}"
            # draw bounding box
            plot_one_box(x=bbox, img=im0, color=color, label=label, line_thickness=2)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = "mp4v"  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                        )
                    vid_writer.write(im0)

        # Print time (inference + NMS)
        print(f"{detections_str}Inference + NMS done. ({t2 - t1:.3f}s)")
        fps = 1.0 / (t2 - t1)
        print(f"FPS: {fps}")

    # * Text to confirm that the image/video has been saved
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    # * Time taken to process the img/video
    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="data/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--person", action="store_true", help="displays only person")
    parser.add_argument("--heads", action="store_true", help="displays only head")
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
