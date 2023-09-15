from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import fitz
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
import torch
from yolov5_dh.utils.augmentations import letterbox

from yolov5_dh.models.common import DetectMultiBackend
from yolov5_dh.utils.general import (check_img_size, non_max_suppression, scale_coords)
from yolov5_dh.utils.torch_utils import select_device, smart_inference_mode

import streamlit as st

MODEL_PATH = f'{str(Path(__file__).parent)}/../../dln_dh.pt'

def load_model(model_path, device='cpu', head=1, imgsz=(1024, 1024), bs = 1, dnn=False, data=None, half=False):
    device = select_device(device)
    model = DetectMultiBackend(model_path, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names[head], model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    return model, stride, names, imgsz

MODEL, STRIDE, NAMES, IMAGE_SIZE = load_model(MODEL_PATH, device='cpu', head=1, imgsz=(1024, 1024))


DPI = 72
DEFAULT_DPI = 72
MATRIX = fitz.Matrix(DPI / DEFAULT_DPI, DPI / DEFAULT_DPI)  # sets zoom factor for 300 dpi

@dataclass()
class Word:
    bbox: Tuple[float, float, float, float]
    word: str


@dataclass()
class AssignmentText:
    page: int
    dlo_box: Tuple[float, float, float, float]
    dlo_label: str
    words: List[Word]
    dlo_score: float = 1.0


@smart_inference_mode()
def infer_image(model, im0, stride, names, imgsz, head, page_rect):
    im0 = np.array(im0)
    size = check_img_size(im0.shape, s=stride)
    im = letterbox(im0, size, stride=stride, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False, head=head)
    pred = non_max_suppression(pred, 0.25, 0.2, None, False, max_det=1000)
    detections = []
    for det in pred:
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        page_rect_gain = torch.tensor([page_rect.height, page_rect.width])[[1, 0, 1, 0]]

        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        
        for *xyxy, conf, category in reversed(det):
            xyxy = ((torch.tensor(xyxy).view(1, 4) / gn) * page_rect_gain).view(-1).tolist()
            line = (names[int(category)], *xyxy, float(conf))  # label format
            detections.append(line)
    return detections

@st.cache_data()
def extract_layout_objects(pdf_path: str, page_range: List[int]):
    dlo_by_page = {}
    with fitz.open(pdf_path) as doc:
        for page_num in page_range:
            page = doc[page_num]
            pix = page.get_pixmap(matrix=MATRIX)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            preds = infer_image(MODEL, img, STRIDE, NAMES, IMAGE_SIZE, head=1, page_rect=page.rect)
            dlo_by_page[page_num] = preds
        word_box_assignment, dlo_by_page = assign_text_boxes(doc, dlo_by_page)
    return word_box_assignment, dlo_by_page


def assign_text_boxes(doc: fitz.Document, yolo_boxes: Dict[int, List[Tuple[str, float, float, float, float, float]]]):
    matches_by_page = {}
    yolo_by_page = {}
    for page_number, boxes in yolo_boxes.items():
        page = doc[page_number]
        words = page.get_text('words')
        word_boxes = np.array([word[:4] for word in words])
        yolo_b = np.array([box[1:5] for box in boxes])
        ioa = bboxes_ioa(yolo_b, word_boxes)

        matches = ioa.argmax(dim=0)
        scores = ioa.max(dim=0).values

        matched_boxes = defaultdict(list)
        for i, (match, score) in enumerate(zip(matches, scores)):
            match = int(match)
            if float(score) > 0.5:
               matched_boxes[match].append((words[i], i))
        assigments = [
            AssignmentText(
                page_number,
                boxes[yolo_box][1:5],
                boxes[yolo_box][0],
                [Word(word[:4], word[4]) for word, _ in wrds],
                boxes[yolo_box][5]
            )  
            for yolo_box, wrds in matched_boxes.items()
        ]
        matches_by_page[page_number] = assigments
        yolo_boxes_sorted = [yolo_b[idx] for idx in matched_boxes.keys()]
        yolo_by_page[page_number] = yolo_boxes_sorted
    return matches_by_page, yolo_by_page


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def bboxes_ioa(yolo_boxes: np.array, word_boxes: np.array, eps=1e-7) -> torch.Tensor:
    yolo_boxes = torch.tensor(yolo_boxes.astype(np.float64))
    word_boxes = torch.tensor(word_boxes.astype(np.float64))

    (a1, a2), (b1, b2) = yolo_boxes[:, None].chunk(2, 2), word_boxes.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoA = inter / (area2)
    return inter / (torch.zeros_like(box_area(yolo_boxes.T)[:, None]) + box_area(word_boxes.T) + eps)


if __name__ == '__main__':
    preds = extract_layout_objects(
        '/home/ilia_kiselev/Downloads/Content_samples_for_automation/LO_Physics_Mazur2/Mazur1_Principles_Ch2.pdf', 
        [5])
    print(preds)