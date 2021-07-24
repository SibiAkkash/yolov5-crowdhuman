# Yolov5 + DeepSORT head tracking

### Yolov5 model trained on crowd human using yolov5(m) architecture
Download Link:  [YOLOv5m-crowd-human](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) 


<br/>

# Output


## Entry video

![back-gif](./data/gifs/2015_05_09_07_56_07_back.gif)

<br />

## Exit video

![front-gif](./data/gifs/2016_04_07_14_29_25_front.gif)

<br />
  
  
# Test

```bash
python .\detect.py \
--weights .\weights\crowdhuman_yolov5m.pt  \
--source <path-to-input-video/image> \
--view-img \
--heads \
--conf-thres 0.5 \
--save
```


# Command line flags

* --weights: path to weights file
  * default = "yolov5s.pt"
* --source: path to input image/video, 0 for webcam
* --results-loc: location to store results text file
  * default = "runs/detect"
* --img-size: 
* --conf-thres: object confidence threshold
  * default = 0.25
* --iou-thres: IOU threshold for non-maxima suppression
  * default = 0.45
* --device: identifier for CUDA device
  * GPU: 0, 1, 2 or 3 for 
  * CPU: cpu
* --view-img: View tracking results live
* --save: save resulting video
* --colab: Run inference on Google Colab    
  * uses cv2_imshow() which works in colab
* --person: display and uses person detections only
* --heads: displays and uses head detections only
