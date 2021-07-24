## Yolov5 + DeepSORT head tracking

### Yolov5 model trained on crowd human using yolov5(m) architecture
Download Link:  [YOLOv5m-crowd-human](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) 


<br/>

## Output
<br/>

### Exit video

![back-gif](./data/gifs/2015_05_09_07_56_07_back.gif)

<br />

### Exit video

![front-gif](./data/gifs/2016_04_07_14_29_25_front.gif)


<br />

## Test

```bash
$ python detect.py --weights crowdhuman_yolov5m.pt --source _test/ --view-img

```
  
  
## Test (Only Person Class)

```bash
python3 detect.py --weights crowdhuman_yolov5m.pt --source _test/ --view-img  --person
```

  
## Test (Only Heads)

```bash
python .\detect.py --weights .\weights\crowdhuman_yolov5m.pt --source <path-to-input-video> --view-img --heads --conf-thres 0.5
```
