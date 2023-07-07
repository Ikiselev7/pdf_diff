from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import fitz
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from rapidfuzz import fuzz,process
from scipy.optimize import linear_sum_assignment
import numpy as np
from itertools import chain
from uuid import uuid4

from document_layot_analysis import AssignmentText, Word
from diff_compute import tokenize_text, get_diff
import streamlit as st

@dataclass()
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass()
class GraphicalDiffOp:
    id: str
    type: str


@dataclass()
class InsertMixin:
    new_page: int
    new_offset: int
    new_length: int
    b_box_new: List[BBox]


@dataclass()
class DeleteMixin:
    old_page:int
    old_offset: int
    old_length: int
    b_box_old: List[BBox]


@dataclass()
class GraphicalInsertOp(GraphicalDiffOp, InsertMixin):
    type:str = 'insert'


@dataclass()
class GraphicalDeleteOp(GraphicalDiffOp, DeleteMixin):
    type:str = 'delete'


@dataclass()
class GraphicalReplaceOp(GraphicalDiffOp, InsertMixin, DeleteMixin):
    type:str = 'replace'


@dataclass()
class Dlo:
    box: Tuple[float, float, float, float]
    label: str
    score: float


@dataclass()
class DloDiff:
    diff: List[GraphicalDiffOp]
    id: str


@dataclass()
class DloInsertMixin:
    new_dlo: Dlo
    new_page: int


@dataclass()
class DloDeleteMixin:
    old_dlo: Dlo
    old_page: int


@dataclass()
class DloReplaceOp(DloDiff, DloInsertMixin, DloDeleteMixin):
    type: str = 'replace'


@dataclass()
class DloInsertOp(DloDiff, DloInsertMixin):
    type: str = 'insert'


@dataclass()
class DloDeleteOp(DloDiff, DloDeleteMixin):
    type: str = 'delete'


GRAFICAL_DIFF_OPS = {op.type: op for op in (GraphicalDeleteOp, GraphicalInsertOp, GraphicalReplaceOp)}

def split_by_lines(words: List[Word], line_threshold: int=5) -> List[List[Word]]:
    word = next(iter(words))
    line = [word]
    lines = []
    for nword in words[1:]:
        cx1, cy1, cx2, cy2 = word.bbox
        x1, y1, x2, y2 = nword.bbox
        if abs(y1 - cy1) <= line_threshold:
            line.append(nword)
        else:
            lines.append(line)
            line = [nword]
        word = nword
    lines.append(line)
        
    return lines


def get_word_map(words: List[Word]):
    word_rect = []
    joined = ''
    for word in words:
        s_pos = len(joined) + 1
        joined = " ".join([joined, word.word])
        e_pos = len(joined)
        word_rect.append(((s_pos, e_pos), word, word.bbox))
    return word_rect, joined


def match_closest(old_blocks, new_blocks, score_thr=50.):
    old_blocks = np.array(old_blocks)
    new_blocks = np.array(new_blocks)

    scores = process.cdist(old_blocks, new_blocks, scorer=fuzz.ratio)
    
    cost_matrix = np.max(scores) - scores
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    best_matches = [(i, j, scores[i, j]) for i, j in zip(row_indices, col_indices) if scores[i, j] >= score_thr]
    
    deletions = [i for i in range(len(old_blocks)) if i not in [k for k, _, _ in best_matches]]
    insertions = [i for i in range(len(new_blocks)) if i not in [j for _, j, _ in best_matches]]
        
    return best_matches, deletions, insertions, cost_matrix
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

def merge_boxes(boxes, line_threshold=5):
    box = next(iter(boxes))
    line = [box]
    lines = []
    for bbox in boxes[1:]:
        cx1, cy1, cx2, cy2 = box
        x1, y1, x2, y2 = bbox
        if abs(y1 - cy1) <= line_threshold:
            line.append(bbox)
        else:
            lines.append(line)
            line = [bbox]
        box = bbox
    lines.append(line)
    
    merged = []
    for line in lines:
        x1 = min([x1 for x1, _, _, _ in line])
        y1 = min([y1 for _, y1, _, _ in line])
        x2 = max([x2 for _, _, x2, _ in line])
        y2 = max([y2 for _, _, _, y2 in line])
        merged.append((x1, y1, x2, y2))
    return merged


def get_single_diff(o_text, n_text, o_words_map, n_words_map, old_page, new_page):
    diff = []
    changes = get_diff(tokenize_text(o_text), tokenize_text(n_text))
    for diff_op in changes:
        change, s_o, e_o, s_n, e_n = diff_op.type, diff_op.source_offset, diff_op.source_offset + diff_op.source_length, diff_op.target_offset, diff_op.target_offset + diff_op.target_length
        d = {'id': str(uuid4())}
        if change in ('replace', 'delete'):
            d.update({
                "old_page": old_page,
                "old_offset": s_o,
                "old_length": e_o - s_o,
            })
            for (s_pos, e_pos), word, (x0, y0, x1, y1) in o_words_map:
                if min(e_o, e_pos) - max(s_pos, s_o) > 0:
                    if 'b_box_old' in d:
                        d['b_box_old'].append((x0, y0, x1, y1))
                    else:
                        d['b_box_old'] = [(x0, y0, x1, y1)]
            if 'b_box_old' in d:
                d['b_box_old'] = merge_boxes(d['b_box_old'])
        
        if change in ('replace', 'insert'):
            d.update({
                "new_page": new_page,
                "new_offset": s_n,
                "new_length": e_n - s_n,
            })
            for (s_pos, e_pos), word, (x0, y0, x1, y1) in n_words_map:
                if  min(e_n, e_pos) - max(s_pos, s_n) > 0:
                    if 'b_box_new' in d:
                        d['b_box_new'].append((x0, y0, x1, y1))
                    else:
                        d['b_box_new'] = [(x0, y0, x1, y1)]
            if 'b_box_new' in d:
                d['b_box_new'] = merge_boxes(d['b_box_new'])
        op = GRAFICAL_DIFF_OPS[change](**d)
        diff.append(op)
    return diff


@st.cache_data()
def layout_diff(old_dlo: Dict[int, List[AssignmentText]], new_dlo: Dict[int, List[AssignmentText]]):
    diff = []
    old_dlo = list(chain(*[dlos for _, dlos in old_dlo.items()]))
    new_dlo = list(chain(*[dlos for _, dlos in new_dlo.items()]))

    oword_maps, old_texts = list(zip(*[get_word_map(dlo.words) for dlo in old_dlo]))
    nword_maps, new_texts = list(zip(*[get_word_map(dlo.words) for dlo in new_dlo]))

    matches, deletions, insertions, cost_matrix = match_closest(old_texts, new_texts)
    for o_index, n_index, score in matches:
        odlo = Dlo(
            box=old_dlo[o_index].dlo_box,
            label=old_dlo[o_index].dlo_label,
            score=old_dlo[o_index].dlo_score
        )
        ndlo = Dlo(
            box=new_dlo[n_index].dlo_box,
            label=new_dlo[n_index].dlo_label,
            score=new_dlo[n_index].dlo_score
        )
                    
        single_diff = []
        if score < 100.0:
            single_diff = get_single_diff(
                old_texts[o_index],
                new_texts[n_index],
                oword_maps[o_index],
                nword_maps[n_index],
                old_dlo[o_index].page,
                new_dlo[n_index].page
            )
        
        diff.append(DloReplaceOp(
            id = str(uuid4()),
            old_dlo=odlo,
            new_dlo=ndlo,
            old_page=old_dlo[o_index].page,
            new_page=new_dlo[n_index].page,
            diff=single_diff
        ))
    for deletion in deletions:
        diff.append(DloDeleteOp(
            id = str(uuid4()),
            old_dlo=Dlo(
                    box=old_dlo[deletion].dlo_box,
                    label=old_dlo[deletion].dlo_label,
                    score=old_dlo[deletion].dlo_score
                ),
            old_page=old_dlo[deletion].page,
            diff=[GraphicalDeleteOp(
                id = str(uuid4()),
                old_page=old_dlo[deletion].page,
                old_offset=0,
                old_length=len(" ".join([word.word for word in old_dlo[deletion].words])),
                b_box_old=merge_boxes([m[2] for m in oword_maps[deletion]])
            )]
        ))
    for insertion in insertions:
        diff.append(DloInsertOp(
            id = str(uuid4()),
            new_dlo=Dlo(
                    box=new_dlo[insertion].dlo_box,
                    label=new_dlo[insertion].dlo_label,
                    score=new_dlo[insertion].dlo_score
                ),
            new_page=new_dlo[insertion].page,
            diff = [GraphicalInsertOp(
                id = str(uuid4()),
                new_page=new_dlo[insertion].page,
                new_offset=0,
                new_length=len(" ".join([word.word for word in new_dlo[insertion].words])),
                b_box_new=merge_boxes([m[2] for m in nword_maps[insertion]])
            )]
        ))
        
    return diff, cost_matrix
