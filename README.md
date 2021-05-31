## Yolov5 + DeepSORT head tracking

### Yolov5 model trained on crowd human using yolov5(m) architecture
Download Link:  [YOLOv5m-crowd-human](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) 


<br/>

**Output (Crowd Human Model)**

![image](https://drive.google.com/uc?export=view&id=1ZOhDBRXj-Ra0vPL7iG6lrxCWAFhJTAti)

<br/>



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
