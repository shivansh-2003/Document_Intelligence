from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
except ImportError as exc:
    raise ImportError(
        "Install crawl4ai first:\n"
        "    pip install crawl4ai\n"
        "    crawl4ai-setup"
    ) from exc


# ---------------------------------------------------------------------------
# Data container  (same shape as ParsedDocument in the other parsers)
# ---------------------------------------------------------------------------

@dataclass
class ParsedWebDocument:
    """Holds the three content buckets extracted from a web page."""

    texts:  list[str]  = field(default_factory=list)
    tables: list[str]  = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class WebParser:
    def __init__(
        self,
        extract_images: bool = True,
        use_pruning_filter: bool = True,
        word_count_threshold: int = 10,
        verbose: bool = False,
    ) -> None:
        self.extract_images       = extract_images
        self.use_pruning_filter   = use_pruning_filter
        self.word_count_threshold = word_count_threshold
        self.verbose              = verbose
        self._result: Optional[ParsedWebDocument] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse_async(self, url: str) -> ParsedWebDocument:
        """Crawl *url* and return a ParsedWebDocument. Crawl4AI is async-native."""

        config = self._build_config()

        async with AsyncWebCrawler(verbose=self.verbose) as crawler:
            crawl_result = await crawler.arun(url=url, config=config)

        if not crawl_result.success:
            raise RuntimeError(
                f"Crawl failed for {url!r}: {crawl_result.error_message}"
            )

        self._result = self._partition(crawl_result)
        return self._result

    def parse(self, url: str) -> ParsedWebDocument:
        """Synchronous convenience wrapper around parse_async()."""
        return asyncio.run(self.parse_async(url))

    @property
    def result(self) -> Optional[ParsedWebDocument]:
        """Last ParsedWebDocument produced, or None if parse() hasn't been called."""
        return self._result

    # ------------------------------------------------------------------
    # CrawlerRunConfig factory
    # ------------------------------------------------------------------

    def _build_config(self) -> CrawlerRunConfig:
        """
        Build a CrawlerRunConfig from instance settings.

        Key choices (per the Crawl4AI docs):
        - DefaultMarkdownGenerator converts cleaned HTML → Markdown.
        - PruningContentFilter (optional) removes nav/footer/ad noise and
          populates result.markdown.fit_markdown.
        - word_count_threshold drops very short text blocks at the HTML level.
        - CacheMode.BYPASS always fetches fresh content.
        - excluded_tags removes navigation and footer clutter from the HTML
          before markdown conversion, which improves table detection.
        """

        content_filter = (
            PruningContentFilter(
                threshold=0.5,
                threshold_type="fixed",
                min_word_threshold=self.word_count_threshold,
            )
            if self.use_pruning_filter
            else None
        )

        md_generator = DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={
                "ignore_links": False,   # keep links for context
                "body_width": 0,         # no hard line-wrap
                "escape_html": False,
            },
        )

        return CrawlerRunConfig(
            # --- Markdown / text ---
            markdown_generator=md_generator,
            word_count_threshold=self.word_count_threshold,

            # --- HTML cleanup ---
            excluded_tags=["nav", "footer", "script", "style", "aside"],
            remove_overlay_elements=True,

            # --- Images ---
            # result.media["images"] is populated automatically by the crawler.
            # Set exclude_all_images=True only when extract_images is False.
            exclude_all_images=not self.extract_images,

            # --- Cache ---
            # BYPASS: always fetch fresh, still writes to cache for later reuse.
            cache_mode=CacheMode.BYPASS,

            # --- Misc ---
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # Content partitioning
    # ------------------------------------------------------------------

    def _partition(self, crawl_result) -> ParsedWebDocument:
        """
        Split the CrawlResult into texts, tables, and images.

        Sources used (all from the official CrawlResult schema):
        - result.markdown.fit_markdown  → filtered text (if pruning is on)
        - result.markdown.raw_markdown  → fallback unfiltered text
        - result.media["images"]        → image list with src, alt, score …
        - The markdown itself           → tables encoded as | … | blocks
        """

        doc = ParsedWebDocument()

        # 1. Pick the best available markdown string
        md_obj  = crawl_result.markdown          # MarkdownGenerationResult
        markdown = ""
        if md_obj:
            # fit_markdown exists only when a content filter was used
            markdown = md_obj.fit_markdown or md_obj.raw_markdown or ""

        # 2. Tables – extract markdown pipe-table blocks first, before stripping them
        doc.tables = self._extract_tables(markdown)

        # 3. Images – Crawl4AI populates result.media["images"] natively;
        #    each entry already has src, alt, desc, score, width, height, type.
        if self.extract_images:
            doc.images = self._extract_images(
                crawl_result.media.get("images", []),
                base_url=crawl_result.url,
            )

        # 4. Texts – remaining markdown after removing table blocks and image refs
        doc.texts = self._extract_texts(markdown, doc.tables)

        return doc

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def _extract_tables(self, markdown: str) -> list[str]:
        """
        Pull out markdown pipe-table blocks.

        A valid markdown table requires:
        - All lines begin with '|'.
        - At least one separator row (| --- | style).
        - At least two lines total (header + separator).
        """

        tables: list[str] = []
        lines  = markdown.splitlines()
        i      = 0

        while i < len(lines):
            if lines[i].strip().startswith("|"):
                block: list[str] = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block.append(lines[i])
                    i += 1

                # Require a proper separator row (| --- | or | :--- | etc.)
                has_separator = any(
                    re.match(r"^\|[\s\-:|]+\|", ln.strip())
                    for ln in block
                )
                if len(block) >= 2 and has_separator:
                    tables.append("\n".join(block))
            else:
                i += 1

        return tables

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def _extract_images(
        self, raw_images: list[dict], base_url: str
    ) -> list[dict]:
        """
        Normalise images from result.media["images"].

        Crawl4AI provides: src, alt, desc, score, width, height, type.
        We resolve relative URLs using the crawled page's base URL.
        """

        seen:   set[str]  = set()
        images: list[dict] = []

        for img in raw_images:
            src = img.get("src", "").strip()
            if not src:
                continue

            # Resolve relative paths (e.g. /images/logo.png → https://…/images/logo.png)
            src = urljoin(base_url, src)

            if src in seen:
                continue
            seen.add(src)

            images.append({
                "src":    src,
                "alt":    img.get("alt", ""),
                "desc":   img.get("desc", ""),
                "score":  img.get("score"),
                "width":  img.get("width"),
                "height": img.get("height"),
                "type":   img.get("type", "image"),
            })

        return images

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_texts(
        self, markdown: str, tables: list[str]
    ) -> list[str]:
        """
        Return meaningful text blocks after removing table blocks and image refs.
        """

        text = markdown

        # Remove every extracted table block so it doesn't bleed into texts
        for table in tables:
            text = text.replace(table, "\n\n")

        # Strip markdown image references  ![alt](url)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

        # Collapse runs of blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        blocks: list[str] = []
        for block in text.split("\n\n"):
            block = block.strip()
            # Keep blocks that have real content; skip horizontal rules and blanks
            if block and not re.fullmatch(r"[-*_]{3,}", block):
                word_count = len(block.split())
                if word_count >= self.word_count_threshold:
                    blocks.append(block)

        return blocks