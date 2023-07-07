import os

from dataclasses import asdict

from document_layot_analysis import extract_layout_objects
from pdf_layout_diff import layout_diff
import streamlit as st
from pdf_ui_coponent import pdf_diff_ui
import base64
import json


def read_local_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()



# def main():
#     old_pdf_path = '/home/ilia_kiselev/Downloads/Content_samples_for_automation/Correlation_Physics_Mazur_1-to-2/Mazur1_Practice_Ch2.pdf'
#     new_pdf_path = '/home/ilia_kiselev/Downloads/Content_samples_for_automation/Correlation_Physics_Mazur_1-to-2/Mazur2_Ch2.pdf'
#     old_page_range = range(2, 3) #range(10, 17)
#     new_page_range = range(25, 26) #range(25, 33)

#     old_dlo, odlo_by_page = extract_layout_objects(old_pdf_path, old_page_range)
#     new_dlo, ndlo_by_page = extract_layout_objects(new_pdf_path, new_page_range)

#     diff, cost_matrix = layout_diff(old_dlo, new_dlo)
#     old = base64.b64encode(read_local_file(old_pdf_path)).decode("utf-8")
#     new = base64.b64encode(read_local_file(new_pdf_path)).decode("utf-8")

#     with open("diff_out.json", 'w') as f:
#         f.write(json.dumps([asdict(d) for d in diff]))


# if __name__ == '__main__':
#     main()


st.set_page_config(layout="wide")

old_pdf_path = '/home/ilia_kiselev/Downloads/Content_samples_for_automation/Correlation_Physics_Mazur_1-to-2/Mazur1_Practice_Ch2.pdf'
new_pdf_path = '/home/ilia_kiselev/Downloads/Content_samples_for_automation/Correlation_Physics_Mazur_1-to-2/Mazur2_Ch2.pdf'

old_page_range = range(2, 18) #range(10, 17)
new_page_range = range(25, 33) #range(25, 33)

old_dlo, odlo_by_page = extract_layout_objects(old_pdf_path, old_page_range)
new_dlo, ndlo_by_page = extract_layout_objects(new_pdf_path, new_page_range)

diff, cost_matrix = layout_diff(old_dlo, new_dlo)


if old_pdf_path is not None and new_pdf_path is not None:
    selected = pdf_diff_ui(
        old_version={'data': base64.b64encode(read_local_file(old_pdf_path)).decode("utf-8")},
        new_version={'data': base64.b64encode(read_local_file(new_pdf_path)).decode("utf-8")},
        diff=[asdict(d) for d in diff],
        key="differ",
    )
