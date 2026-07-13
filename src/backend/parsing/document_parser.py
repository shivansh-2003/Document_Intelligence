# cell 3 — PDF: hi_res + table structure + image payloads
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from utils.parsing_utils import RawElement ,to_raw
from pathlib import Path


class DocumentPartitioner:
   

    def partition(self, path: str) -> list[RawElement]:
        suffix = Path(path).suffix.lower()
        fn = {
            ".pdf":  self.pdf,
            ".docx": self.docx,
            ".pptx": self.pptx,
        }.get(suffix)
        if fn is None:
            raise ValueError(f"unsupported: {suffix}")
        return fn(path)

    def pdf(self, path: str) -> list[RawElement]:
        els = partition_pdf(
            filename=path,
            strategy="hi_res",                           # required for Table/Image detection
            infer_table_structure=True,                  # -> metadata.text_as_html
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,         # -> metadata.image_base64
        )
        return to_raw(els, path)

    def docx(self, path: str) -> list[RawElement]:
        return to_raw(partition_docx(filename=path, infer_table_structure=True), path)

    def pptx(self, path: str) -> list[RawElement]:
        return to_raw(partition_pptx(filename=path, infer_table_structure=True), path)