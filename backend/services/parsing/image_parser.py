"""
image_parser.py
Multi-modal image extraction pipeline for RAG systems.
Extracts rich summaries and text from any image type (diagrams, charts,
graphs, trees, screenshots, scanned pages).

INTEGRATION CONTRACT:
- When an image contains tabular content, this module ONLY extracts the raw
  2D grid and passes it to table_parser.py via ExtractedImageTable.
  NO table summarization, NO markdown generation, NO quality scoring duplication.
- For all non-table images, produces exhaustive VLM descriptions + OCR text.

NEW BACKENDS:
    1. DoclingSmolVLM  – local HuggingFace SmolVLM-256M for fast picture
                         description (no API key, runs on CPU/GPU).
    2. OllamaQwenVLM   – Ollama wrapper for qwen3.6 (or any Qwen-VL tag)
                         for local vision+text inference.

Dependencies:
    pip install pillow pytesseract openai numpy
    pip install transformers torch accelerate  # For DoclingSmolVLM
    pip install easyocr                          # Optional OCR fallback
"""

import base64
import io
import json
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# OCR Backends
# ------------------------------------------------------------------
try:
    import pytesseract
    from pytesseract import Output

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# ------------------------------------------------------------------
# Table Parser Integration
# ------------------------------------------------------------------
from table_parser import ExtractedImageTable, NormalizedTable, TableParser


# =====================================================================
#  DATA MODELS
# =====================================================================

class ImageType(Enum):
    TABLE = "table"
    CHART = "chart"          # bar, line, pie, scatter, heatmap
    DIAGRAM = "diagram"      # flowchart, architecture, UML, mind map
    GRAPH = "graph"          # network graph, tree, DAG
    DOCUMENT = "document"    # scanned page, receipt, form
    GENERAL = "general"      # photo, screenshot, illustration


@dataclass
class ParsedImage:
    """
    Standardized output from the image parser.
    For tables: `table` field is populated; `summary` contains only a
    lightweight provenance note (the heavy lifting lives in table_parser.py).
    For non-tables: `summary` contains the exhaustive VLM description and
    `extracted_text` contains raw OCR output.
    """
    source: str
    image_type: ImageType
    summary: str
    extracted_text: str = ""
    structured_content: Optional[Dict[str, Any]] = None   # e.g. chart data points
    table: Optional[NormalizedTable] = None             # Routed from table_parser
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0


# =====================================================================
#  VLM BACKEND ABSTRACTION
# =====================================================================

class VLMBackend(ABC):
    """Pluggable vision-language model interface."""

    @abstractmethod
    def describe(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> str:
        """Send image + text prompt to VLM; return text response."""
        ...

    def _pil_to_base64(self, image: Image.Image, fmt: str = "PNG") -> str:
        buffered = io.BytesIO()
        image.save(buffered, format=fmt)
        return base64.b64encode(buffered.getvalue()).decode()


# ------------------------------------------------------------------
#  1. OpenAI GPT-4o / GPT-4-Turbo
# ------------------------------------------------------------------
class OpenAIVLM(VLMBackend):
    """OpenAI GPT-4o / GPT-4-Turbo vision backend."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install `openai` to use OpenAIVLM") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def describe(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> str:
        img_b64 = self._pil_to_base64(image, fmt="PNG")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content


# ------------------------------------------------------------------
#  2. Ollama (llava, llama3.2-vision, moondream, qwen3.6, etc.)
# ------------------------------------------------------------------
class OllamaVLM(VLMBackend):
    """
    Local Ollama backend. Works with any vision-capable model pulled via Ollama.
    Recommended vision tags: llava, llama3.2-vision, moondream, qwen2.5-vl.
    Note: qwen3.6 is primarily a text model; use a vision-tagged variant
    (e.g. qwen2.5-vl if available) for image understanding.
    """

    def __init__(
        self,
        model: str = "llava",
        base_url: str = "http://localhost:11434",
    ):
        try:
            import ollama
        except ImportError as exc:
            raise ImportError("Install `ollama` to use OllamaVLM") from exc
        self.client = ollama
        self.model = model
        self.base_url = base_url

    def describe(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> str:
        img_b64 = self._pil_to_base64(image, fmt="PNG")
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            options={"num_predict": max_tokens, "temperature": 0.1},
        )
        return response["message"]["content"]


# ------------------------------------------------------------------
#  3. Docling SmolVLM (local, lightweight, no API key)
# ------------------------------------------------------------------
class DoclingSmolVLM(VLMBackend):
    """
    HuggingFace SmolVLM-256M-Instruct (or 500M variant) via transformers.
    Optimized for fast picture description in document pipelines.
    Runs entirely locally; ideal for batch PDF page annotation.

    Model: HuggingFaceTB/SmolVLM-256M-Instruct
    Install: pip install transformers torch accelerate pillow
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        device: Optional[str] = None,   # 'cuda', 'cpu', or None for auto
        dtype: Optional[Any] = None,    # torch.float16 or torch.bfloat16
    ):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError as exc:
            raise ImportError(
                "Install `transformers torch accelerate` for DoclingSmolVLM"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def describe(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> str:
        import torch

        # SmolVLM expects a specific chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # The processor returns pixel_values for the image; ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Re-process image pixels separately because apply_chat_template
        # may not attach pixel_values in all transformers versions
        pixel_inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = pixel_inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,   # Greedy for reproducibility
            )
        # Decode only the newly generated tokens
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return generated_text.strip()


# =====================================================================
#  PROMPT LIBRARY
# =====================================================================

class Prompts:
    """
    Carefully engineered prompts for zero data-loss extraction.
    Each prompt demands exhaustive, structured output.
    """

    TABLE_DETECTION = (
        "You are a document analysis engine. Look at the image and decide if the "
        "PRIMARY content is a structured data table (rows and columns with cells). "
        "Reply with exactly one word: TABLE or NOT_TABLE. No explanation."
    )

    TABLE_GRID_EXTRACTION = (
        "You are a precision table OCR engine. The image contains a data table. "
        "Extract the ENTIRE table as a JSON 2D array (list of rows). "
        "Rules:\n"
        "1. Each inner array is one table row, left-to-right.\n"
        "2. Preserve ALL text exactly as shown. Do not abbreviate.\n"
        "3. Use empty string \"\" for blank cells.\n"
        "4. Include header rows if present.\n"
        "5. Output ONLY the JSON array. No markdown fences, no commentary.\n\n"
        "Example output format:\n"
        "[[\"Header1\", \"Header2\"], [\"Row1Col1\", \"Row1Col2\"], [\"\", \"\"]]"
    )

    CHART_SUMMARY = (
        "You are a data extraction assistant. The image is a chart or graph. "
        "Produce an exhaustive, lossless description for a RAG retrieval system. "
        "Include:\n"
        "1. Chart type (bar, line, pie, scatter, area, heatmap, etc.).\n"
        "2. Exact title and all axis labels (x-axis, y-axis, units).\n"
        "3. Legend: every series name, color, and marker style.\n"
        "4. All visible data values, percentages, or approximate numeric readings.\n"
        "5. Notable trends, peaks, troughs, outliers, or anomalies.\n"
        "6. Any annotations, data labels, or callouts.\n"
        "7. Date ranges or categories shown.\n"
        "Do not omit any visible data point. Be quantitative."
    )

    DIAGRAM_SUMMARY = (
        "You are a technical documentation assistant. The image is a diagram, "
        "flowchart, architecture drawing, or process map. "
        "Produce an exhaustive, lossless description for RAG retrieval. "
        "Include:\n"
        "1. Overall layout and structure (top-down, left-right, circular, etc.).\n"
        "2. Every node, box, or component with its EXACT label text.\n"
        "3. Every connection, arrow, or edge with its label or cardinality.\n"
        "4. Decision points, conditions, and branch labels.\n"
        "5. Color coding or visual groupings and their meaning.\n"
        "6. Any icons, symbols, or notation standards used (UML, BPMN, etc.).\n"
        "Preserve all text verbatim. Describe topology completely."
    )

    GRAPH_SUMMARY = (
        "You are a network analysis assistant. The image is a graph, tree, "
        "hierarchy, or network visualization. "
        "Produce an exhaustive description for RAG retrieval. "
        "Include:\n"
        "1. Graph type (directed, undirected, tree, DAG, bipartite).\n"
        "2. Every node/vertex with exact label and any attributes.\n"
        "3. Every edge/connection with labels, weights, or directions.\n"
        "4. Root nodes, leaf nodes, cycles, or clusters if visible.\n"
        "5. Layout algorithm hints (force-directed, hierarchical, radial).\n"
        "6. Any color or size encodings and their meaning.\n"
        "Preserve all text exactly."
    )

    DOCUMENT_SUMMARY = (
        "You are a document intelligence assistant. The image is a scanned page, "
        "form, receipt, or document snippet. "
        "Produce an exhaustive, lossless description for RAG retrieval. "
        "Include:\n"
        "1. Document type (invoice, form, receipt, letter, report page, etc.).\n"
        "2. All visible text in reading order, preserving layout sections.\n"
        "3. Form fields, labels, and filled values.\n"
        "4. Tables: if a small table is present, transcribe it; otherwise note its presence.\n"
        "5. Signatures, stamps, logos, or handwriting.\n"
        "6. Any redactions or highlighted areas.\n"
        "Preserve all text exactly. Do not summarize away details."
    )

    GENERAL_SUMMARY = (
        "You are a visual understanding assistant. Describe this image in "
        "exhaustive detail for a multi-modal RAG system. "
        "Include:\n"
        "1. All visible text (exactly as shown, preserve spelling).\n"
        "2. Visual elements, objects, colors, textures, and spatial layout.\n"
        "3. Relationships between elements (arrows, grouping, overlap).\n"
        "4. Any numbers, dates, names, or identifiers visible.\n"
        "5. Context or purpose if inferable from content.\n"
        "6. UI elements if it's a screenshot (buttons, menus, panels).\n"
        "Do not omit any visible detail. Every element should be mentioned."
    )


# =====================================================================
#  IMAGE PREPROCESSING
# =====================================================================

class ImagePreprocessor:
    """Resize, enhance, and normalize images for VLM / OCR consumption."""

    MAX_VLM_SIZE = (2048, 2048)   # OpenAI high-res limit
    OCR_DPI = 300

    @classmethod
    def prepare_for_vlm(cls, image: Image.Image) -> Image.Image:
        """Resize if too large; convert to RGB."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.width > cls.MAX_VLM_SIZE[0] or image.height > cls.MAX_VLM_SIZE[1]:
            image.thumbnail(cls.MAX_VLM_SIZE, Image.Resampling.LANCZOS)
        return image

    @classmethod
    def prepare_for_ocr(cls, image: Image.Image) -> Image.Image:
        """Grayscale, upscale slightly, denoise for OCR."""
        img = image.convert("L")  # grayscale
        # Mild upscale if too small
        if img.width < 800 or img.height < 800:
            ratio = max(800 / img.width, 800 / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img


# =====================================================================
#  CORE PARSER
# =====================================================================

class ImageParser:
    """
    Multi-modal image parser for RAG pipelines.

    Responsibilities:
      1. Load image from path, URL, or PIL Image.
      2. Run OCR to extract raw text.
      3. Detect if image is a table.
         -> YES: extract 2D grid via VLM, build ExtractedImageTable,
            delegate normalization to TableParser.normalize_from_image_table().
         -> NO:  run VLM with image-type-specific prompt for exhaustive summary.
      4. Return ParsedImage with unified metadata.

    Usage:
        # OpenAI cloud VLM
        parser = ImageParser(vlm=OpenAIVLM(api_key="sk-..."))

        # Local Ollama with qwen3.6 (or any vision model)
        parser = ImageParser(vlm=OllamaVLM(model="qwen3.6"))

        # Local Docling SmolVLM (no API key, 256M params)
        parser = ImageParser(vlm=DoclingSmolVLM())

        result = parser.parse("dashboard_screenshot.png")
        if result.table:
            print(result.table.markdown)   # From table_parser.py
        else:
            print(result.summary)            # Rich VLM description
    """

    def __init__(
        self,
        vlm: Optional[VLMBackend] = None,
        use_ocr: bool = True,
        ocr_engine: str = "auto",   # 'tesseract', 'easyocr', 'auto'
        table_parser: Optional[TableParser] = None,
        min_table_quality: float = 0.3,
    ):
        self.vlm = vlm
        self.use_ocr = use_ocr
        self.ocr_engine = ocr_engine
        self.table_parser = table_parser or TableParser(min_quality=min_table_quality)
        self.min_table_quality = min_table_quality

        # Lazy-init EasyOCR reader
        self._easyocr_reader: Optional[Any] = None

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def parse(
        self,
        source: Union[str, Path, Image.Image],
        source_type: str = "image",
        page_or_slide: Optional[int] = None,
        surrounding_context: Optional[str] = None,
    ) -> ParsedImage:
        """
        Parse a single image and return a ParsedImage object.

        Args:
            source: File path, URL, or PIL Image.
            source_type: Provenance tag (e.g. 'pdf_image', 'web_image').
            page_or_slide: Page/slide number for metadata.
            surrounding_context: Text around the image in the parent doc.
        """
        image = self._load_image(source)
        source_str = str(source) if not isinstance(source, Image.Image) else "PIL_Image"

        # --- Step 1: OCR text extraction (always run, cheap signal) ---
        ocr_text = self._run_ocr(image) if self.use_ocr else ""

        # --- Step 2: Table detection ---
        is_table = self._detect_table(image, ocr_text)

        # --- Step 3: Branch ---
        if is_table:
            return self._handle_table(
                image=image,
                source=source_str,
                source_type=source_type,
                page_or_slide=page_or_slide,
                surrounding_context=surrounding_context,
                ocr_text=ocr_text,
            )

        return self._handle_non_table(
            image=image,
            source=source_str,
            source_type=source_type,
            page_or_slide=page_or_slide,
            surrounding_context=surrounding_context,
            ocr_text=ocr_text,
        )

    def parse_batch(
        self,
        sources: List[Union[str, Path, Image.Image]],
        source_type: str = "image",
    ) -> List[ParsedImage]:
        """Parse multiple images; returns list of ParsedImage."""
        results: List[ParsedImage] = []
        for src in sources:
            try:
                results.append(self.parse(src, source_type=source_type))
            except Exception as e:
                warnings.warn(f"Failed to parse image {src}: {e}")
                results.append(
                    ParsedImage(
                        source=str(src),
                        image_type=ImageType.GENERAL,
                        summary="",
                        quality_score=0.0,
                    )
                )
        return results

    # -----------------------------------------------------------------
    #  Table Branch (delegates to table_parser.py)
    # -----------------------------------------------------------------

    def _handle_table(
        self,
        image: Image.Image,
        source: str,
        source_type: str,
        page_or_slide: Optional[int],
        surrounding_context: Optional[str],
        ocr_text: str,
    ) -> ParsedImage:
        """
        TABLE PATH: Extract raw 2D grid via VLM, wrap in ExtractedImageTable,
        and pass to TableParser for normalization. We do NOT summarize here.
        """
        grid = self._extract_table_grid(image)

        # Build the contract object expected by table_parser.py
        img_table = ExtractedImageTable(
            grid=grid,
            source=source,
            source_type=source_type,
            page_or_slide=page_or_slide,
            title_or_caption=None,  # Could be enriched via surrounding_context heuristics
            surrounding_context=surrounding_context,
            extraction_method="vlm_ocr",
            confidence=1.0,
        )

        # Delegate 100% of normalization to table_parser.py
        normalized: NormalizedTable = self.table_parser.normalize_from_image_table(
            data=img_table, table_index=0
        )

        # Minimal provenance summary for the image parser consumer
        provenance = (
            f"[Image contains a table extracted and normalized by table_parser.py. "
            f"Quality score: {normalized.quality_score}. "
            f"Headers: {normalized.headers}.]"
        )

        return ParsedImage(
            source=source,
            image_type=ImageType.TABLE,
            summary=provenance,
            extracted_text=ocr_text,
            table=normalized,
            metadata={
                "extraction_method": "vlm_grid → table_parser",
                "vlm_confidence": img_table.confidence,
                "surrounding_context": surrounding_context,
            },
            quality_score=normalized.quality_score,
        )

    # -----------------------------------------------------------------
    #  Non-Table Branch (VLM + OCR)
    # -----------------------------------------------------------------

    def _handle_non_table(
        self,
        image: Image.Image,
        source: str,
        source_type: str,
        page_or_slide: Optional[int],
        surrounding_context: Optional[str],
        ocr_text: str,
    ) -> ParsedImage:
        """
        NON-TABLE PATH: Classify image type, run VLM with specific prompt,
        optionally structure chart data, merge with OCR text.
        """
        # Classify for best prompt selection
        img_type = self._classify_non_table(image, ocr_text)

        # Select prompt
        prompt_map = {
            ImageType.CHART: Prompts.CHART_SUMMARY,
            ImageType.DIAGRAM: Prompts.DIAGRAM_SUMMARY,
            ImageType.GRAPH: Prompts.GRAPH_SUMMARY,
            ImageType.DOCUMENT: Prompts.DOCUMENT_SUMMARY,
            ImageType.GENERAL: Prompts.GENERAL_SUMMARY,
        }
        prompt = prompt_map.get(img_type, Prompts.GENERAL_SUMMARY)

        # VLM description
        vlm_summary = ""
        if self.vlm is not None:
            try:
                vlm_img = ImagePreprocessor.prepare_for_vlm(image)
                vlm_summary = self.vlm.describe(vlm_img, prompt, max_tokens=4096)
            except Exception as e:
                warnings.warn(f"VLM description failed: {e}")

        # Merge OCR text into summary (ensures no text loss)
        full_summary = self._merge_summary_and_ocr(vlm_summary, ocr_text, img_type)

        # Optional: structured extraction for charts
        structured: Optional[Dict[str, Any]] = None
        if img_type == ImageType.CHART and self.vlm is not None:
            structured = self._extract_chart_structure(image)

        # Quality heuristic
        quality = self._compute_non_table_quality(vlm_summary, ocr_text)

        return ParsedImage(
            source=source,
            image_type=img_type,
            summary=full_summary,
            extracted_text=ocr_text,
            structured_content=structured,
            metadata={
                "page_or_slide": page_or_slide,
                "surrounding_context": surrounding_context,
                "vlm_prompt_used": prompt.split("\n")[0],
            },
            quality_score=quality,
        )

    # -----------------------------------------------------------------
    #  VLM Helpers
    # -----------------------------------------------------------------

    def _detect_table(self, image: Image.Image, ocr_text: str) -> bool:
        """
        Hybrid table detection:
        1. Heuristic: OCR text has many pipe-like or tabular patterns.
        2. VLM classifier if available.
        """
        # Heuristic A: OCR text looks like a table
        lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
        pipe_like = sum(1 for l in lines if l.count("|") >= 2 or l.count("\t") >= 2)
        if lines and pipe_like / len(lines) > 0.3:
            return True

        # Heuristic B: many short numeric/short-text lines of similar length
        if len(lines) >= 3:
            lengths = [len(l) for l in lines]
            if np.std(lengths) < np.mean(lengths) * 0.3:
                return True

        # VLM classifier (most accurate)
        if self.vlm is not None:
            try:
                small_img = ImagePreprocessor.prepare_for_vlm(image)
                resp = self.vlm.describe(
                    small_img, Prompts.TABLE_DETECTION, max_tokens=10
                )
                return "TABLE" in resp.upper()
            except Exception as e:
                warnings.warn(f"VLM table detection failed: {e}")

        return False

    def _extract_table_grid(self, image: Image.Image) -> List[List[str]]:
        """
        Ask VLM to return a JSON 2D array representing the table.
        Parse and validate the grid.
        """
        if self.vlm is None:
            raise RuntimeError("VLM required for table grid extraction but none configured.")

        vlm_img = ImagePreprocessor.prepare_for_vlm(image)
        raw = self.vlm.describe(vlm_img, Prompts.TABLE_GRID_EXTRACTION, max_tokens=4096)

        # Sanitize: remove markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        # Parse JSON
        try:
            grid = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract first JSON array
            match = re.search(r"\[\s*\[.*?\]\s*\]", raw, re.DOTALL)
            if match:
                try:
                    grid = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"VLM returned unparseable table grid: {raw[:200]}") from exc
            else:
                raise ValueError(f"VLM returned unparseable table grid: {raw[:200]}")

        # Validate shape
        if not isinstance(grid, list) or not all(isinstance(r, list) for r in grid):
            raise ValueError("VLM table grid is not a list of lists.")

        # Normalize row lengths (pad short rows)
        if grid:
            max_len = max(len(r) for r in grid)
            grid = [r + [""] * (max_len - len(r)) for r in grid]

        return grid

    def _classify_non_table(self, image: Image.Image, ocr_text: str) -> ImageType:
        """
        Lightweight classification to pick the best VLM prompt.
        Uses VLM if available, otherwise heuristics.
        """
        if self.vlm is not None:
            classify_prompt = (
                "Classify this image into exactly one category: "
                "CHART, DIAGRAM, GRAPH, DOCUMENT, or GENERAL. "
                "Reply with only the category word."
            )
            try:
                vlm_img = ImagePreprocessor.prepare_for_vlm(image)
                resp = self.vlm.describe(vlm_img, classify_prompt, max_tokens=20)
                cat = resp.strip().upper()
                for it in ImageType:
                    if it.name in cat:
                        return it
            except Exception as e:
                warnings.warn(f"VLM classification failed: {e}")

        # Heuristic fallback using OCR text density
        words = ocr_text.split()
        if not words:
            return ImageType.GENERAL
        avg_word_len = sum(len(w) for w in words) / len(words)

        if avg_word_len > 6 and len(words) > 50:
            return ImageType.DOCUMENT
        if any(k in ocr_text.lower() for k in ["axis", "legend", "plot", "data", "%", "total"]):
            return ImageType.CHART
        if any(k in ocr_text.lower() for k in ["flow", "process", "step", "decision", "start", "end"]):
            return ImageType.DIAGRAM

        return ImageType.GENERAL

    def _extract_chart_structure(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Optional: ask VLM to return chart data as structured JSON.
        """
        if self.vlm is None:
            return None

        prompt = (
            "This image is a chart. Extract its data into strict JSON with keys: "
            "chart_type, title, x_axis_label, y_axis_label, series (list of "
            "{name, data_points}), and annotations. Output ONLY JSON. No prose."
        )
        try:
            vlm_img = ImagePreprocessor.prepare_for_vlm(image)
            raw = self.vlm.describe(vlm_img, prompt, max_tokens=4096)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            return json.loads(raw)
        except Exception:
            return None

    # -----------------------------------------------------------------
    #  OCR Engine
    # -----------------------------------------------------------------

    def _run_ocr(self, image: Image.Image) -> str:
        """Run OCR and return concatenated text."""
        img = ImagePreprocessor.prepare_for_ocr(image)

        engines = []
        if self.ocr_engine == "auto":
            engines = ["tesseract", "easyocr"]
        else:
            engines = [self.ocr_engine]

        texts: List[str] = []
        for eng in engines:
            try:
                if eng == "tesseract" and HAS_TESSERACT:
                    texts.append(self._ocr_tesseract(img))
                    break  # tesseract success
                elif eng == "easyocr" and HAS_EASYOCR:
                    texts.append(self._ocr_easyocr(img))
                    break  # easyocr success
            except Exception as e:
                warnings.warn(f"OCR engine {eng} failed: {e}")
                continue

        return "\n".join(texts)

    def _ocr_tesseract(self, image: Image.Image) -> str:
        config = r"--oem 3 --psm 6"
        return pytesseract.image_to_string(image, config=config)

    def _ocr_easyocr(self, image: Image.Image) -> str:
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False)
        np_img = np.array(image)
        results = self._easyocr_reader.readtext(np_img, detail=0)
        return "\n".join(results)

    # -----------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _load_image(source: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.copy()
        source_str = str(source)
        if source_str.startswith(("http://", "https://")):
            import requests
            resp = requests.get(source_str, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path)

    @staticmethod
    def _merge_summary_and_ocr(
        vlm_summary: str, ocr_text: str, img_type: ImageType
    ) -> str:
        """
        Ensure OCR text is present in final summary so no text is lost.
        If VLM already included the text, don't duplicate aggressively.
        """
        if not ocr_text.strip():
            return vlm_summary

        # Simple containment check
        ocr_snippets = [s.strip() for s in ocr_text.splitlines() if len(s.strip()) > 4]
        missing_ratio = sum(
            1 for s in ocr_snippets if s not in vlm_summary
        ) / max(len(ocr_snippets), 1)

        if missing_ratio > 0.3:
            return (
                f"{vlm_summary}\n\n"
                f"--- Extracted OCR Text ---\n{ocr_text}"
            )
        return vlm_summary

    @staticmethod
    def _compute_non_table_quality(vlm_summary: str, ocr_text: str) -> float:
        """Heuristic quality score based on output richness."""
        if not vlm_summary and not ocr_text:
            return 0.0
        score = 0.5
        if len(vlm_summary) > 200:
            score += 0.3
        if len(ocr_text) > 50:
            score += 0.2
        return round(min(score, 1.0), 2)


# =====================================================================
#  HIGH-LEVEL RAG INTEGRATION
# =====================================================================

def parse_images_for_rag(
    sources: List[Union[str, Path, Image.Image]],
    vlm: Optional[VLMBackend] = None,
    source_type: str = "image",
    table_parser: Optional[TableParser] = None,
    enrich_with_llm: bool = False,
    llm_client=None,
) -> List[Dict[str, Any]]:
    """
    One-shot batch parser producing chunk-ready dicts for a RAG pipeline.

    Returns list of dicts with keys:
        - text          : full text for embedding
        - summary       : VLM summary (or table provenance note)
        - markdown      : table markdown if image was a table
        - extracted_text: raw OCR
        - metadata      : image_type, quality_score, source, etc.
        - type          : "image"
    """
    parser = ImageParser(vlm=vlm, table_parser=table_parser)
    parsed = parser.parse_batch(sources, source_type=source_type)

    chunks: List[Dict[str, Any]] = []
    for p in parsed:
        parts = []
        if p.summary:
            parts.append(p.summary)
        if p.extracted_text and p.image_type != ImageType.TABLE:
            parts.append(f"OCR Text:\n{p.extracted_text}")
        if p.structured_content:
            parts.append(f"Structured Data:\n{json.dumps(p.structured_content, indent=2)}")

        full_text = "\n\n".join(parts)

        # Optional LLM re-summarization / compression
        if enrich_with_llm and llm_client is not None and p.image_type != ImageType.TABLE:
            prompt = (
                "Given this image description and OCR text, write a dense 3-sentence "
                "summary that preserves all facts, names, numbers, and relationships. "
                "Then append the original description for completeness.\n\n"
                f"{full_text[:6000]}\n\nDense Summary:"
            )
            try:
                if hasattr(llm_client, "invoke"):
                    resp = llm_client.invoke(prompt)
                    desc = resp.content if hasattr(resp, "content") else str(resp)
                else:
                    resp = llm_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    desc = resp.choices[0].message.content
                full_text = f"{desc}\n\n{full_text}"
            except Exception as e:
                warnings.warn(f"LLM enrichment failed for {p.source}: {e}")

        chunk: Dict[str, Any] = {
            "text": full_text,
            "summary": p.summary,
            "extracted_text": p.extracted_text,
            "image_type": p.image_type.value,
            "source": p.source,
            "source_type": source_type,
            "quality_score": p.quality_score,
            "type": "image",
        }

        # If it was a table, embed the table_parser output
        if p.table:
            chunk["markdown"] = p.table.markdown
            chunk["table_headers"] = p.table.headers
            chunk["table_footnotes"] = p.table.footnotes
            chunk["table_quality"] = p.table.quality_score
            # For embedding, prepend table markdown to the text
            table_text = f"Table from image:\n{p.table.markdown}"
            if p.table.footnotes:
                table_text += f"\nFootnotes: {', '.join(p.table.footnotes)}"
            chunk["text"] = table_text + "\n\n" + chunk["text"]

        if p.structured_content:
            chunk["structured_content"] = p.structured_content

        chunks.append(chunk)

    return chunks