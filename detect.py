# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
import config
from pathlib import Path
from dataclasses import dataclass

import win32con, win32gui, win32api
import pyglet
#from detect import start
from pyglet.window import Window
from pyglet.graphics import Batch
import threading
import queue
import pyautogui
import math
import win32api, win32con



import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

@dataclass
class Vector:
    x: int
    y: int

    def add(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
@dataclass   
class RGB:
    r: int
    g: int
    b: int

    def add(self, other):
        return RGB(self.x + other.x, self.y + other.y)

batch = Batch()


def display_get_width():
    display = pyglet.canvas.get_display()
    screen = display.get_default_screen()
    return screen.width

def display_get_height():
    display = pyglet.canvas.get_display()
    screen = display.get_default_screen()
    return screen.height

ObjectCount = 0
ObjectCountCache = 0
Objects = {}

class CacheObject:
    def __init__(self, obj_type, **kwargs):
        self.type = obj_type
        self.kwargs = kwargs

def update_object(obj):
    global ObjectCountCache
    Objects[ObjectCountCache] = obj
    ObjectCountCache += 1

def draw_line(pos1 : Vector, pos2 : Vector, color : RGB, width : int):
    global ObjectCountCache
    obj = CacheObject('Line', pos1=pos1, pos2=pos2, color=color, width=width)
    update_object(obj)

def reset():
    global ObjectCountCache
    ObjectCountCache = 0

def c_update():
    global ObjectCount
    ObjectCount = ObjectCountCache

def get_object_by_index(index):
    return Objects.get(index, None)

def draw_box(x, y, w, h, color=(255, 0, 0), width=1):
     # Calculate the four corners based on x, y, w, and h
    topleft = Vector(x, y)
    topright = Vector(x + w, y)
    bottomleft = Vector(x, y + h)
    bottomright = Vector(x + w, y + h)
    
    # Draw the four lines for the box
    draw_line(topleft, topright, color, width)  # Top
    draw_line(topleft, bottomleft, color, width)  # Left
    draw_line(topright, bottomright, color, width)  # Right
    draw_line(bottomleft, bottomright, color, width)  # Bottom
    
def draw_text(screen_pos, text, color=(255, 255, 255)):
    global ObjectCountCache
    obj = CacheObject('Text', screen_pos=screen_pos, text=text, color=color)
    update_object(obj)

def is_in_fov(x, y, radius):
    
    center_x = display_get_width() / 2
    center_y = display_get_height() / 2

    dx = x - center_x
    dy = y- center_y
    distance_from_center = math.sqrt(dx**2 + dy**2)

    if distance_from_center <= radius:
        return True
    return False

window = Window(display_get_width(), display_get_height(), style=Window.WINDOW_STYLE_OVERLAY, caption="Karlito's Aiming Software")
detection_queue = queue.Queue()
# Correct `_hwnd`.
win32gui.SetLayeredWindowAttributes(
    window._hwnd, win32api.RGB(255, 255, 255), 0, win32con.LWA_COLORKEY
)
# Correct `_view_hwnd`.
win32gui.SetWindowLong(
    window._view_hwnd,
    win32con.GWL_EXSTYLE,
    win32gui.GetWindowLong(window._view_hwnd, win32con.GWL_EXSTYLE)
    | win32con.WS_EX_LAYERED,
)
win32gui.SetLayeredWindowAttributes(
    window._view_hwnd, win32api.RGB(255, 255, 255), 0, win32con.LWA_COLORKEY
)
pyglet.gl.glClearColor(255, 255, 255, 255.0)
#draw_line(Vector(250,250),Vector(500,600),RGB(250,250,255))

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "screen",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=[0],  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=True,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    
):
    #drawing.draw_line(drawing.Vector(250,250),drawing.Vector(500,800),drawing.RGB(250,250,255))
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir : Path = Path("runs/detect")
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1  
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                reset()
                for *xyxy, conf, cls in reversed(det):
                    #print(xyxy)
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                            
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy

                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )
                        f_x : float = coords[0]
                        f_y : float = coords[1]
                        x : int = int(f_x * display_get_width())
                        y : int = int((1 - f_y) * (display_get_height() - 1))

                        if y > 900 and config.game == "cs2" : continue

                        #draw_box((100, 100), (200, 200))
                        draw_line((display_get_width() / 2, 0), (x, y), color=(255, 0, 0), width=1)

                        if is_in_fov(x, y, config.fov) :
                            pyautogui.click()
                            win32api.mouse_event(win32con.MOUSEEVENTF_ABSOLUTE, 100, 100, 0, 0)
                            
                    
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                c_update()
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    #parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    #parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    #parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
 
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

def start():
    #draw_line(Vector(250,250),Vector(500,700),RGB(250,250,255))
    opt = parse_opt()
    main(opt)

thread = threading.Thread(target=start, daemon=True)
thread.start()

@window.event
def on_draw():
    window.clear()

    for i in range(ObjectCount):
        obj = get_object_by_index(i)
        if obj:
            if obj.type == 'Line':
                pos1 = obj.kwargs['pos1']
                pos2 = obj.kwargs['pos2']
                color = obj.kwargs['color']
                width = obj.kwargs['width']
                pyglet.shapes.Line(pos1[0], pos1[1], pos2[0], pos2[1], width=width, color=color, batch=batch).draw()

            elif obj.type == 'Text':
                screen_pos = obj.kwargs['screen_pos']
                text = obj.kwargs['text']
                label = pyglet.text.Label(text, x=screen_pos[0], y=screen_pos[1], color=(*obj.kwargs['color'], 255), batch=batch)
                label.draw()

    

    batch.draw()

    #reset()
    #c_update()

def update(dt):
    window.dispatch_event('on_draw')

pyglet.clock.schedule_interval(update, 1/30.0)
arc = pyglet.shapes.Arc(display_get_width() / 2, display_get_height() / 2, config.fov, color=(200, 200, 200), batch = batch)
pyglet.app.run()

