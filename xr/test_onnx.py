import sys

from torch._C import device
sys.path.append('/mnt/sdb/dpai3/project/YOLOP')
from pathlib import Path
import shutil
import os
import cv2
import time
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression
import torch.backends.cudnn as cudnn
from lib.dataset import LoadImages, LoadStreams
from lib.core.function import AverageMeter
from tqdm import tqdm
import torchvision.transforms as transforms
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.core.postprocess import morphological_process, connect_lane
from lib.utils import plot_one_box,show_seg_result


normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


label_map = {
    0:"person",
    1:"rider",
    2:"car",
    3:"bus",
    4:"truck",
    5:"bike",
    6:"motor",
    7:"tl_green",
    8:"tl_red",
    9:"tl_yellow",
    10:"tl_none",
    11:"traffic sign",
    12:"train"
}

def data_process(img):
    img_bgr = img
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)
    return img, canvas, r, dw, dh, new_unpad_w, new_unpad_h


def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop(weight="yolop-640-640.onnx",
                img_path="./inference/images/7dd9ef45-f197db95.jpg"):

    ort.set_default_logger_severity(4)
    # onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(weight,providers=['CPUExecutionProvider'])
    print(f"Load {weight} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    save_det_path = f"/mnt/sdb/dpai3/project/YOLOP/inference/output/detect_onnx.jpg"
    # save_da_path = f"D:\StudyFiles\project\Graduation_design\YOLOP\inference\output\da_onnx.jpg"
    save_ll_path = f"/mnt/sdb/dpai3/project/YOLOP/inference/output/ll_onnx.jpg"
    save_merge_path = f"/mnt/sdb/dpai3/project/YOLOP/inference/output/output_onnx.jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
    det_out, ll_seg_out = ort_session.run(
        ['det_out', 'lane_line_seg'],
        input_feed={"images": img}
    )

    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    print(f"detect {boxes.shape[0]} bounding boxes.")

    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    cv2.imwrite(save_det_path, img_det)

    # select da & ll segment area.
    # da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    # da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
    # print(da_seg_mask.shape)
    print(ll_seg_mask.shape)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    # color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    # merge: resize to original size
    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    # da: resize to original size
    # da_seg_mask = da_seg_mask * 255
    # da_seg_mask = da_seg_mask.astype(np.uint8)
    # da_seg_mask = cv2.resize(da_seg_mask, (width, height),
    #                          interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_merge_path, img_merge)
    # cv2.imwrite(save_da_path, da_seg_mask)
    cv2.imwrite(save_ll_path, ll_seg_mask)

    print("detect done.")


def detect(opt):
    if not os.path.exists(opt.save_dir):  # output dir
        # shutil.rmtree(opt.save_dir)  # delete dir
        os.makedirs(opt.save_dir)  # make new dir
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    ort.set_default_logger_severity(4)
    # onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(opt.weight)
    print(f"Load {opt.weight} done!")
    print(type(ort_session))
    # set gpu device

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()
    # ort_session = ort_session.set_providers(['CUDAExecutionProvider'])

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)
    print("num outputs: ", len(outputs_info))

    vid_path, vid_writer = None, None
    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size
    # Run inference
    t0 = time.time()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    device = torch.device('cuda:0')
    for index, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        time_img = transform(img).to(device)
        img_bgr = img_det
        height, width, _ = img_bgr.shape
        # convert to RGB
        img_rgb = img_bgr[:, :, ::-1].copy()
        # resize & normalize
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)
        # Inference
        t1 = time_synchronized()
        # inference: (1,n,6) (1,2,640,640)
        det_out, ll_seg_out = ort_session.run(
            ['det_out', 'lane_line_seg'],
            input_feed={"images": img}
        )
        t2 = time_synchronized()
        inf_time.update(t2-t1,time_img.size(0))
        det_out = torch.from_numpy(det_out).float()
        t3 = time_synchronized()
        boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
        t4 = time_synchronized()
        nms_time.update(t4-t3,time_img.size(0))
        boxes = boxes.cpu().numpy().astype(np.float32)

        if boxes.shape[0] == 0:
            print("no bounding boxes detected.")
            
        # scale coords to original size.
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r

        print(f"detect {boxes.shape[0]} bounding boxes.")

        # select da & ll segment area.
        # da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        # da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
        # print(da_seg_mask.shape)
        print(ll_seg_mask.shape)

        color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
        # color_area[da_seg_mask == 1] = [0, 255, 0]
        color_area[ll_seg_mask == 1] = [255, 0, 0]
        color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        color_mask = np.mean(color_seg, 2)
        img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        # merge: resize to original size
        img_merge[color_mask != 0] = \
            img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img_merge = img_merge.astype(np.uint8)
        img_merge = cv2.resize(img_merge, (width, height),
                            interpolation=cv2.INTER_LINEAR)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
            # img_merge = cv2.putText(img_merge, label_map[i], (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        # da: resize to original size
        # da_seg_mask = da_seg_mask * 255
        # da_seg_mask = da_seg_mask.astype(np.uint8)
        # da_seg_mask = cv2.resize(da_seg_mask, (width, height),
        #                          interpolation=cv2.INTER_LINEAR)

        # ll: resize to original size
        ll_seg_mask = ll_seg_mask * 255
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        
        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_merge)

        elif dataset.mode == 'video':
            vid_path = opt.source
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_merge.shape
                print('fps:', fps)
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_merge)
        
        else:
            cv2.imshow('image', img_merge)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="/mnt/sdb/dpai3/project/YOLOP/weights/yolop-640-640-det-ll-120.onnx")
    parser.add_argument('--img', type=str, default="/mnt/sdb/dpai3/project/YOLOP/inference/images/adb4871d-4d063244.jpg")
    parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--source', type=str, default='../inference/images', help='source')
    parser.add_argument('--save_dir', type=str, default='../inference/onnx_output', help='directory to save results')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    args = parser.parse_args()

    # infer_yolop(weight=args.weight, img_path=args.img)
    detect(args)
    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img test.jpg
    """