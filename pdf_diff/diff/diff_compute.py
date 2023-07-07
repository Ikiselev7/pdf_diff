import abc
from dataclasses import dataclass
from itertools import chain
from typing import List
from diff_match_patch import diff_match_patch
import difflib
import spacy
import json


NLP = spacy.load("en_core_web_sm")


@dataclass()
class SpaWord:
    text: str
    start: int
    end: int


@dataclass()
class SpaSentence:
    text: str
    start: int
    end: int
    words: List[SpaWord]


@dataclass()
class DiffOp:
    source_offset: int
    source_length: int
    target_offset: int
    target_length: int


@dataclass()
class InsertOp(DiffOp):
    type: str = 'insert'                                                                                                    


@dataclass()
class DeleteOp(DiffOp):
    type: str = 'delete'


@dataclass()
class ReplaceOp(DiffOp):
    type: str = 'replace'    


DIFF_OPS_CLS = {op.type: op for op in (InsertOp, DeleteOp, ReplaceOp)}


def tokenize_text(text):
    doc = NLP(text)
    
    sentences = []
    for sent in doc.sents:
        sentences.append(
            SpaSentence(
            sent.text, 
            sent.start_char, 
            sent.end_char, 
            [SpaWord(token.text, token.idx, token.idx + len(token.text)) for token in sent]
        ))

    return sentences


def get_word_level_diff(
        old_sent: List[SpaSentence],
        new_sent: List[SpaSentence],
    ):
    old_words = list(chain(*(sent.words for sent in old_sent)))
    new_words = list(chain(*(sent.words for sent in new_sent)))

    seqmatcher = difflib.SequenceMatcher(
        a=[o.text for o in old_words],
        b=[n.text for n in new_words]
    )

    ops = []

    for opcode, s_o, e_o, s_n, e_n in seqmatcher.get_opcodes():
        if opcode == 'equal':
            continue
        opcls = DIFF_OPS_CLS[opcode]
        old_part = old_words[s_o: e_o]
        if old_part:
            ostart = old_part[0].start
            oend = old_part[-1].end
        else:
            ostart = old_words[-1].end
            oend = old_words[-1].end
        
        new_part = new_words[s_n: e_n]
        if new_part:
            nstart = new_part[0].start
            nend = new_part[-1].end
        else:
            nstart = new_words[-1].end
            nend = new_words[-1].end

        op = opcls(
            source_offset=ostart,
            source_length=oend - ostart,
            target_offset=nstart,
            target_length=nend - nstart,            
        )
        ops.append(op)

    return ops    


def get_diff(old: List[SpaSentence], new: List[SpaSentence]):
    seqmatcher = difflib.SequenceMatcher(
        a=[o.text for o in old], 
        b=[n.text for n in new]
    )
    diff = []
    for opcode, s_o, e_o, s_n, e_n in seqmatcher.get_opcodes():
        if opcode == 'equal':
            continue
        elif opcode == 'replace':
            diff.extend(get_word_level_diff(old[s_o: e_o], new[s_n: e_n]))
        else:
            diff.append(
                DIFF_OPS_CLS[opcode](
                    source_offset=old[s_o: e_o][0].start if old[s_o: e_o] else 0,
                    source_length=old[s_o: e_o][-1].end - old[s_o: e_o][0].start if old[s_o: e_o] else 0,
                    target_offset=new[s_n: e_n][0].start if new[s_n: e_n] else 0,
                    target_length=new[s_n: e_n][-1].end - new[s_n: e_n][0].start if new[s_n: e_n] else 0,
                )
            )
    return diff


def _get_changes2(old, new):
    dmp = diff_match_patch()
    diff = dmp.diff_main(old, new)
    dmp.diff_cleanupSemantic(diff)

    offset = 0
    changes = []
    i = 0
    while i < len(diff):
        op, data = diff[i]
        length = len(data)
        if op != 0: # not equal
            if op == -1 and i+1 < len(diff) and diff[i+1][0] == 1: # replace case
                next_op, next_data = diff[i+1]
                next_length = len(next_data)
                changes.append({
                    'operation': 'replace',
                    'offset': offset,
                    'length': length,
                    'replace_length': next_length,
                    'data': data,
                    'replace_data': next_data
                })
                i += 1
                offset += next_length
            else:
                changes.append({
                    'operation': 'delete' if op == -1 else 'insert',
                    'offset': offset,
                    'length': length,
                    'data': data
                })

        if op != -1: # not delete
            offset += length

        i += 1

    return changes


def _get_changes(old, new):
    d = difflib.Differ()
    diff = list(d.compare(old, new))
    
    changes = []
    offset = 0
    i = 0
    while i < len(diff):
        line = diff[i]
        op = line[0]
        data = line[2:]
        length = len(data)
        if op != ' ':
            if op == '-' and i+1 < len(diff) and diff[i+1][0] == '+':
                next_line = diff[i+1]
                next_data = next_line[2:]
                next_length = len(next_data)
                changes.append({
                    'operation': 'replace',
                    'offset': offset,
                    'length': length,
                    'replace_length': next_length,
                    'data': data,
                    'replace_data': next_data
                })
                offset += next_length
                i += 1
            else:
                changes.append({
                    'operation': 'delete' if op == '-' else 'insert',
                    'offset': offset,
                    'length': length,
                    'data': data
                })
        if op != '-':
            offset += length
        i += 1
    return changes




# old = "In the end it doesn't even matter. I try so hard and got so far. Hello world, is it good day? How are you? I am fine, thank you."
# new = 'I try so hard and got so far. Haha. Goodbye world, it was good day! How are you? Blah blah blah.'
# # changes = get_changes(old, new)
# # print(changes)
# # print(get_changes(old, new))

# print(get_diff(tokenize_text(old), tokenize_text(new)))

