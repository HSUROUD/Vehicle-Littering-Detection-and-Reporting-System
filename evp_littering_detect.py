import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import math
import easyocr
import pandas as pd
import os

car_classes_filter = [2,3]
trash_classes_filter = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15]

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
trash_deque = []
car_deque = []

cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
trash_deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
car_deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
'''
def load_model(device, model_path, model_type, optimize=True, height=256, square=True):
    if optimize:
        model = torch.jit.load(model_path, map_location=device).to(device)
    else:
        model = torch.hub.load('intel-isl/MiDaS', model_type).to(device).eval()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform
    net_w, net_h = 384, 384
    if square:
        net_w, net_h = max(net_w, net_h), max(net_w, net_h)
    else:
        net_w, net_h = net_w, net_h
    return model, transform, net_w, net_h
'''
grayscale = False
side = False
first_execution = True
# 加载 EVP 模型
#%cd C:/Users/User/Desktop/Object-Detection-Web-App-Using-YOLOv7-and-Flask-main/EVP-main
from transformers import AutoModel
from PIL import Image
import torchvision.transforms as transforms
import sys
from transformers import AutoModel
from PIL import Image
sys.path.append('C:/Users/User/Desktop/Object-Detection-Web-App-Using-YOLOv7-and-Flask-main/EVP-main/stable-diffusion')
sys.path.append('C:/Users/User/Desktop/Object-Detection-Web-App-Using-YOLOv7-and-Flask-main/EVP-main/timing-transformers')
sys.path.append('C:/Users/User/Desktop/Object-Detection-Web-App-Using-YOLOv7-and-Flask-main/EVP-main/clip')
sys.path.append('C:/Users/User/Desktop/Object-Detection-Web-App-Using-YOLOv7-and-Flask-main/EVP-main/depth')

evp_model = AutoModel.from_pretrained("MykolaL/evp_depth", trust_remote_code=True).to('cpu')
print('evp success')
##########################################################################################midas
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)



##########################################################################################
# 这里添加了您的其他函数定义，例如：
# letterbox, xyxy_to_xywh, compute_color_for_labels, load_classes 等
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 

def calculate_box_center(box):
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y
'''
def is_point_inside_box(point, box):
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def calculate_box_distance(center1, center2):
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance
'''
def box_inside(box1, box2):
    """
    Check if box2 is inside box1
    box1 and box2 are in the format [x1, y1, x2, y2]
    """
    x1_inside = box2[0] >= box1[0]  # 檢查 box2 的左邊界是否在 box1 的左邊界內部
    y1_inside = box2[1] >= box1[1]  # 檢查 box2 的上邊界是否在 box1 的上邊界內部
    x2_inside = box2[2] <= box1[2]  # 檢查 box2 的右邊界是否在 box1 的右邊界內部
    y2_inside = box2[3] <= box1[3]  # 檢查 box2 的下邊界是否在 box1 的下邊界內部 
    print('x1_inside',x1_inside)
    print('y1_inside',y1_inside)
    print('x2_inside',x2_inside)
    print('y2_inside',y2_inside)

    return x1_inside and y1_inside and x2_inside and y2_inside

def violation_detect(trashes,cars,depth):
    violations = []
    trash_list = []
    matched_cars = set()
    for trash in trashes:
        store = None
        count = None
        #print('trash',trash)
        trash_x_min, trash_y_min, trash_x_max, trash_y_max, trash_identity, trash_object_id = trash
        trash_list.append(trash_object_id)
        print('trash_object_id',trash_object_id)
        #if trash_object_id in trash_classes_filter:
        trash_box = trash_x_min, trash_y_min, trash_x_max, trash_y_max
        trash_area = abs((trash_x_max-trash_x_min)*(trash_y_min-trash_y_max))
        trash_center_x , trash_center_y = calculate_box_center(trash_box)
        # 将中心坐标转换为整数
        trash_center_x_int = int(round(trash_center_x))
        trash_center_y_int = int(round(trash_center_y))
        print("Depth map shape:", depth.shape)
        print("Depth map dtype:", depth.dtype)
        print("Trash center:", trash_center_y_int, trash_center_x_int)
        

        # 使用整数索引提取深度值
        trash_depth = depth[trash_center_y_int, trash_center_x_int].item()
        print("Trash depth:", trash_depth)

        for car in cars:
            car_tuple = tuple(car)
            if car_tuple in matched_cars:  # 如果该车辆已匹配过，则跳过
                continue
            #print('car',car)
            car_x_min, car_y_min, car_x_max, car_y_max, car_identity, car_object_id = car
            #print('car_object_id',car_object_id)
            #if car_object_id in car_classes_filter:
            car_box = car_x_min, car_y_min, car_x_max, car_y_max
            car_area = abs((car_x_max-car_x_min)*(car_y_min-car_y_max))
            car_center_x, car_center_y = calculate_box_center(car_box)
            # 将中心坐标转换为整数
            car_center_x_int = int(round(car_center_x))
            car_center_y_int = int(round(car_center_y))

            # 使用整数索引提取深度值
            car_depth = depth[car_center_y_int, car_center_x_int].item()
            
            if car_area >= trash_area*4: 
                if trash_depth == car_depth:
                    print("Matched trash and car:")
                    print("Trash center:", trash_center_y_int, trash_center_x_int)
                    print("Car center:", car_center_y_int, car_center_x_int)
                    print("Trash depth:", trash_depth)
                    print("Car depth:", car_depth)
                    violations.append(car)
                    print('car append:',car)
                    matched_cars.add(car_tuple)  # 将匹配过的车辆添加到集合中
                    store = None
                    break
                else:
                    depth_difference = trash_depth - car_depth
                    print('depth_difference:',depth_difference )
                    if count is None or count > depth_difference:
                        count = depth_difference
                        store = car
        if store is not None:
            print("Trash depth:", trash_depth)
            print("Car depth:", car_depth)
            print('car append:',store)
            violations.append(store)
    return violations,trash_list

def add_violation_record(csv_file, car_plate, trash_type, frame):
    df = pd.read_csv(csv_file)  # 读取CSV文件为DataFrame
    print('car_plate',car_plate)
    print('trash_type',trash_type)
    for i in range(len(car_plate)):
        print('add to csv:',car_plate[i],trash_type[i],frame)
        new_record = pd.DataFrame({'car plate': [car_plate[i]], 'trash type': [trash_type[i]], 'frame': [frame]})  # 创建新记录
        df = pd.concat([df, new_record], ignore_index=True)  # 将新记录添加到DataFrame
    df.to_csv(csv_file, index=False)  # 将DataFrame保存回CSV文件

def calculate_overlap_ratio(box1, box2):
    # 計算交集部分的左上角和右下角座標
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    # 計算交集部分的寬度和高度
    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)

    # 計算交集部分的面積
    intersection_area = intersection_width * intersection_height

    # 計算兩個矩形的聯集部分的面積
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area

    # 計算重疊比例
    overlap_ratio = intersection_area / union_area if union_area > 0 else 0

    return overlap_ratio

def detect(opt):
    # 设置视频源
    
    video_path = opt.source
    video_filename = Path(video_path).stem
    video = cv2.VideoCapture(video_path)
    
    new_folder_path = Path(opt.project) / video_filename
    new_folder_path.mkdir(parents=True, exist_ok=True)

    csv_file = os.path.join(opt.project, video_filename+'/'+video_filename + '.csv')
    print('csv path:',csv_file)


    if not os.path.isfile(csv_file):
        df = pd.DataFrame(columns=['car plate', 'trash type', 'frame'])  # 定义DataFrame
        df.to_csv(csv_file, index=False)  
        print('csv file created')
    
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialzing object for writing video output
    output_video_path = new_folder_path / (video_filename + '.mp4')
    depth_video_path = new_folder_path / ('depth' + '.mp4')
    depth_video_path1 = new_folder_path / ('depth1' + '.mp4')
    output = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    depth_video_output = cv2.VideoWriter(str(depth_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), False)
    depth_video_output1 = cv2.VideoWriter(str(depth_video_path1), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    torch.cuda.empty_cache()
    # Initializing model and setting it for inference
    vid_path, vid_writer = None, None

    
        
    with torch.no_grad():
        source, weights, weights2, weights3, view_img, save_txt, imgsz, trace = opt.source, opt.trash_weights, opt.car_weights, opt.plate_weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
        #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        
        set_logging()
        device = select_device(opt.device)
        print('device:',device)
        half = device.type != 'cpu'
        trash_model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(trash_model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            trash_model.half()

        names = trash_model.module.names if hasattr(trash_model, 'module') else trash_model.names

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            trash_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(trash_model.parameters())))

        classes1 = None
        trash_classes = load_classes('data/trash.names')
        if trash_classes:
            classes1 = []
            #print(coco_classes)
            for class_name1 in trash_classes:
                classes1.append(trash_classes.index(class_name1))
        #print(opt['classes1'])
        #print('trash',classes1)

        car_model = attempt_load(weights2, map_location=device)
        stride2 = int(car_model.stride.max())
        imgsz2 = check_img_size(imgsz, s=stride)
        if half:
            car_model.half()

        names2 = car_model.module.names if hasattr(car_model, 'module') else car_model.names
        colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in names2]

        if device.type != 'cpu':
            car_model(torch.zeros(1, 3, imgsz2, imgsz2).to(device).type_as(next(car_model.parameters())))

        classes2 = None
        coco_classes = load_classes('data/coco.names')
        if coco_classes:
            classes2 = []
            #print(coco_classes)
            for class_name2 in coco_classes:
                classes2.append(coco_classes.index(class_name2))

        plate_model = attempt_load(weights3, map_location=device)
        #plate_model = attempt_load(weights3, map_location=torch.device('cuda'))
        #plate_model = attempt_load(weights3, map_location=device).to(torch.float32)

        stride3 = int(plate_model.stride.max())
        imgsz3 = check_img_size(imgsz, s=stride)
        if half:
            plate_model.half()

        names3 = plate_model.module.names if hasattr(plate_model, 'module') else plate_model.names
        colors3 = [[random.randint(0, 255) for _ in range(3)] for _ in names3]

        if device.type != 'cpu':
            plate_model(torch.zeros(1, 3, imgsz3, imgsz3).to(device).type_as(next(plate_model.parameters())))

        classes3 = [0]
        # print('classes1',classes1)
        # print('classes2',classes2)
        # print('classes3',classes3)
        #print(opt['classes1'],opt['classes2'])

    for j in range(nframes):

            ret, img0 = video.read()
            img_copy = img0.copy()

            switch = False

            trash_store = []
            car_store = []
            if ret:
                if j ==0:
                    start_time = time.time()
                #midas_frame = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                #midas_tensor = transform(midas_frame).to(device)
                #深度辨識影片區
                # original_image_rgb = np.flip(img0, 2)  # in [0, 255] (flip required to get RGB)
                # image = transform({"image": original_image_rgb/255})["image"]

                # prediction = process(midas_device, midas, midas_model_type, image, (net_w, net_h),
                #                          original_image_rgb.shape[1::-1], midas_optimize, True)
                
                # original_image_bgr = np.flip(original_image_rgb, 2) if side else None
                # depth_map1 = create_side_by_side(original_image_bgr, prediction, grayscale)
                # depth_map = cv2.cvtColor(depth_map1, cv2.COLOR_BGR2GRAY)
                if j ==0:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print("execution time:", execution_time, "s")


                img = letterbox(img0, imgsz, stride=stride)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # print("Input data type:", img.dtype)
                # for name, param in plate_model.named_parameters():
                #     print(f" plate Parameter '{name}' type:", param.dtype)
                # for name, param in car_model.named_parameters():
                #     print(f" car Parameter '{name}' type:", param.dtype)
                # for name, param in trash_model.named_parameters():
                #     print(f" trash Parameter '{name}' type:", param.dtype)

                # Inference
                t1 = time_synchronized()
                pred1 = trash_model(img, augment= False)[0]
                pred1 = non_max_suppression(pred1 , opt.conf_thres, opt.iou_thres, classes=classes1, agnostic= False)
                pred2 = car_model(img, augment=False)[0]
                pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=classes2, agnostic=False)
                pred_start= time.time()
                pred3 = plate_model(img, augment=False)[0]
                pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, classes=classes3, agnostic=False)
                pred_end= time.time()
                #depth_map = midas(midas_tensor)
                #depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1), size=(h,w), mode="bicubic", align_corners=False).squeeze()
                t2 = time_synchronized()
                #pred, LP_detected_img = detect(car_model, source_img, device, imgsz=640)
                #模型一辨識
                for i, det in enumerate(pred1):
                    #print('det1 :', det)
                    trash_s = ''
                    trash_s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            trash_s += f"{n} {classes1[int(c)]}{'s' * (n > 1)}, "  # add to string
                        trash_xywh_bboxs = []
                        trash_confs = []
                        trash_oids = []
                        #print('trash_s',trash_s)
                        for *xyxy, conf, cls in reversed(det):
                            #print('Original cls:', cls)
                            #print('Converted cls:', int(cls))
                            #if int(cls) in trash_classes_filter:
                            #print('Trash cls', int(cls))
                            trash_x_c, trash_y_c, trash_bbox_w, trash_bbox_h = xyxy_to_xywh(*xyxy)
                            trash_xywh_obj = [trash_x_c, trash_y_c, trash_bbox_w, trash_bbox_h]
                            trash_xywh_bboxs.append(trash_xywh_obj)
                            trash_confs.append([conf.item()])
                            trash_oids.append(int(cls))
                        #print('trash_xywh_bboxs',trash_xywh_bboxs)
                        #print('trash_confs',trash_confs)
                        #print('trash_oids')

                        trash_xywhs = torch.Tensor(trash_xywh_bboxs)
                        trash_confss = torch.Tensor(trash_confs)

                        trash_outputs = trash_deepsort.update(trash_xywhs, trash_confss, trash_oids, img0)

                        if len(trash_outputs) > 0:
                            for trash_output in trash_outputs:
                                trash_x_min, trash_y_min, trash_x_max, trash_y_max, trash_identity, trash_object_id = trash_output
                                #for key in list(trash_deque):
                                    #if key not in identity:
                                        #trash_deque.pop(key)
                                #id = int(identity[i]) if identity is not None else 0
                                #print('trash_object_id before append',trash_object_id)
                                if trash_identity not in trash_deque and trash_object_id in trash_classes_filter:  
                                    switch = True
                                    trash_deque.append(trash_identity)
                                    trash_store.append(trash_output)
                                    color = compute_color_for_labels(trash_object_id)
                                    #print(trash_object_id)
                                    trash_obj_name = trash_classes[trash_object_id]
                                    #if 0 <= trash_object_id < len(opt['classes1']):
                                        #trash_obj_name = opt['classes1'][trash_object_id]
                                        #trash_label = f'{trash_identity}:{trash_obj_name}'
                                        #trash_bbox = [trash_x_min, trash_y_min, trash_x_max, trash_y_max]
                                        #plot_one_box(trash_bbox, img0, label=trash_label, color=color, line_thickness=3)
                                    #else:
                                        #print(f"Invalid class index: {trash_object_id}")

                                    trash_label = f'{trash_identity}:{trash_obj_name}'
                                    trash_bbox = [trash_x_min, trash_y_min, trash_x_max, trash_y_max]
                                    plot_one_box(trash_bbox, img0, label=trash_label, color=color, line_thickness=3)

                    #模型二辨識
                for i, det in enumerate(pred2):
                    #print('det2 :', det)
                    car_s = ''
                    car_s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            car_s += f"{n} {classes2[int(c)]}{'s' * (n > 1)}, "  # add to string
                        car_xywh_bboxs = []
                        car_confs = []
                        car_oids = []
                        #print('car_s',car_s)
                        for *xyxy, conf, cls in reversed(det):
                            #if int(cls) in car_classes_filter:
                            #print('car cls', cls)
                            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                            car_xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                            car_xywh_bboxs.append(car_xywh_obj)
                            car_confs.append([conf.item()])
                            car_oids.append(int(cls))

                        car_xywhs = torch.Tensor(car_xywh_bboxs)
                        car_confss = torch.Tensor(car_confs)

                        car_outputs = car_deepsort.update(car_xywhs, car_confss, car_oids, img0)

                        if len(car_outputs) > 0:
                            for car_output in car_outputs:
                                car_x_min, car_y_min, car_x_max, car_y_max, car_identity, car_object_id = car_output
                                #for key in list(car_deque):
                                    #if key not in identity:
                                        #car_deque.pop(key)
                                #id = int(identity[i]) if identity is not None else 0
                                if car_identity not in car_deque:  
                                    car_deque.append(car_identity)
                                #print('car_object_id before append',car_object_id)
                                if car_object_id in car_classes_filter:
                                    car_store.append(car_output)
                                    color = compute_color_for_labels(car_object_id)
                                    car_obj_name = coco_classes[car_object_id]
                                    car_label = f'{car_identity}:{car_obj_name}'
                                    car_bbox = [car_x_min, car_y_min, car_x_max, car_y_max]
                                #plot_one_box(car_bbox, img0, label=car_label, color=color, line_thickness=3)
                if switch:
                    frame_pil = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(frame_pil)
                    transform = transforms.ToTensor()
                    image = transform(im).unsqueeze(0).to('cpu')
                    depth = evp_model(image)
                    depth_np = depth.squeeze()#彩色深度圖
                    depth_min = depth_np.min()
                    depth_max = depth_np.max()
                    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255
                    depth_gray = depth_normalized.astype('uint8')
                    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_HOT)
                    depth_color_image = Image.fromarray(depth_color)
                    depth_gray_image = Image.fromarray(depth_gray)

                    violation, trash_lists = violation_detect(trash_store, car_store , depth_gray)
                #print(violation)
                    if violation:
                        trash_list=[]
                        plate_list=[]
                        number_added = False
                        for i in trash_lists:
                            trash_list.append(trash_classes[i])

                        for i, det in enumerate(pred3):
                            pred2_start = time.time()
                        #print('det2 :', det)
                            plate_s = ''
                            plate_s += '%gx%g ' % img.shape[2:]  # print string
                            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                            if len(det):
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    plate_s += f"{n} {names3[int(c)]}{'s' * (n > 1)}, "  # use names2 for the second model
                                    pred2_end = time.time()

                                for vehicel in violation:
                                    print('vehicel',vehicel)
                                    count = 1
                                    x_min, y_min, x_max, y_max, identity, object_id = vehicel
                                    color = (0, 0, 255)  # 紅色的 BGR 顏色碼
                                    obj_name = coco_classes[object_id]
                                    bbox = [x_min, y_min, x_max, y_max]
                                    label = f'{identity}:{obj_name} violation:littering'
                                    for *xyxy, conf, cls in reversed(det):
                                        #p_x_c, p_y_c, p_bbox_w, p_bbox_h = xyxy_to_tlwh(xyxy)
                                        #print('p_x_c, p_y_c, p_bbox_w, p_bbox_h',p_x_c, p_y_c, p_bbox_w, p_bbox_h)
                                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                        plate_box = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
                                        box_coordinates = [int(round(tensor_item.item())) for tensor_item in plate_box]
                                        print('plate_box:',box_coordinates)
                                        plate_start = time.time()
                                        #plot_one_box(box_coordinates, img0, label=str(count), color=color, line_thickness=3)
                                        #plot_one_box(bbox, img0, label=str(count), color=color, line_thickness=3)
                                        '''
                                        cv2.imshow('violation', img0)
                                        # 等待按鍵輸入，0 表示無限等待
                                        cv2.waitKey(0)
                                        cv2.destroyAllWindows()
                                        '''

                                        if box_inside(bbox,box_coordinates ) or calculate_overlap_ratio(bbox,box_coordinates)>=0.8:
                                            plate_start = time.time()
                                            cropped_image = img0[y1:y2, x1:x2]  # 裁剪原始圖像的特定區域
                                            reader = easyocr.Reader(['en'])
                                            result = reader.readtext(cropped_image)
                                            plot_one_box(box_coordinates, img0, label='plate', color=color, line_thickness=3)
                                            #label = f'{names2[int(cls)]} {conf:.2f}'  # use names2 for the second model
                                            try:
                                                label = result[0][1] 
                                                plate_list.append(result[0][1])
                                                print('append number')
                                                number_added = True
                                            except:
                                                #label = f'{names2[int(cls)]} {conf:.2f}'  # use names2 for the second model
                                                label = f'{identity}:{obj_name} violation:littering'
                                                plate_list.append('null')
                                                number_added = True
                                                print('append null 1')
                                            plate_end = time.time()
                                            print('plate detect time:',pred2_end+pred_end-pred2_start-pred_start)
                                            print('easyocr time:',plate_end-plate_start)
                                    if number_added == False:
                                        label = f'{identity}:{obj_name} violation:littering'
                                        plate_list.append('null')
                                        print('append null 2')
                                        #print('label:',label )
                                        #plot_one_box(xyxy, img0, label=label, color=colors2[int(cls)], line_thickness=3)
                                    #label = f'{identity}:{obj_name} violation:littering'
                                    plot_one_box(bbox, img0, label=label, color=color, line_thickness=3)
                                    count+=1
                    add_violation_record(csv_file, plate_list, trash_list, j+1)
                    '''
                    try:
                        add_violation_record(csv_file, plate_list, trash_list, j+1)
                    except:
                        pass
                    '''
                    origin_frame_filename = new_folder_path / f"origin_{j+1}.jpg"
                    frame_filename = new_folder_path / f"frame_{j+1}.jpg"
                    depth_output = new_folder_path / f"depth_frame_{j+1}.jpg"
                    depth_output1 = new_folder_path / f"depth1_frame_{j+1}.jpg"
                    cv2.imwrite(str(origin_frame_filename), img_copy)
                    cv2.imwrite(str(frame_filename), img0)
                    depth_gray_image.save(depth_output)
                    depth_color_image.save(depth_output1)
                    #cv2.imwrite(str(depth_output1), depth_rgb)
                    print(f"violation frame saved in folder: {new_folder_path}")


                    # 等待按鍵輸入，0 表示無限等待
                    #cv2.waitKey(1)
                    #cv2.destroyAllWindows()

                #depth_output = depth_map.cpu().numpy()
                #depth_output = (depth_output - depth_output.min()) / (depth_output.max() - depth_output.min()) * 255
                #depth_output = depth_output.astype(np.uint8)

                #print(f"{j+1}/{nframes} frames processed")
                
                output.write(img0)
                #depth_video_output.write(depth_map)
                #depth_video_output1.write(depth_map1)






    # 释放资源
    output.release()
    #depth_video_output.release()
    #depth_video_output1.release()
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--car_weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--trash_weights', nargs='+', type=str, default='trash.pt', help='model.pt path(s)')
    parser.add_argument('--plate_weights', nargs='+', type=str, default='plate.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='static', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # 添加其他您需要的参数
    opt = parser.parse_args()

    # 检查需求并执行检测
    with torch.no_grad():
        detect(opt)
