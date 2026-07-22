import os

os.environ["MINERU_DEVICE_MODE"] = "cpu"
os.environ["MINERU_MODEL_SOURCE"] = "local"

from mineru.cli.common import do_parse

pdf_path = r"D:\forAI\project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf"
output_dir = r"D:\forAI\project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"

print("Starting MinerU pipeline direct parse...")
print(f"Input: {pdf_path}")
print(f"Output: {output_dir}")

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

file_name = os.path.splitext(os.path.basename(pdf_path))[0]

do_parse(
    output_dir=output_dir,
    pdf_file_names=[file_name],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=["ch"],
    backend="pipeline",
    parse_method="auto",
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_model_output=True,
    f_dump_orig_pdf=True,
    f_dump_content_list=True,
    f_draw_layout_bbox=True,
    f_draw_span_bbox=True,
)

print("Done!")
