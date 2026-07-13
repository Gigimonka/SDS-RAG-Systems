"""Convert Help&Manual XML projects to a Russian Wiki.js content repository.\n\nThis version preserves internal Help&Manual bookmarks and cross-page fragments.

The source files are deliberately treated as read-only.  A few historical
topics contain malformed attributes (for example ``href anchor="..."``).
Those files are repaired in memory only and are listed in the conversion
report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT = ROOT / "wikijs_export"

PROJECTS = [
    {
        "name": "Common_Project26-2",
        "title": "ИнфоКлиника",
        "path": "infoclinica",
        "asset_path": "infoclinica",
        "topics": ROOT / "Common_Project26-2" / "Topics",
        "toc": ROOT / "Common_Project26-2" / "Maps" / "table_of_contents.xml",
        "images": ROOT / "Common_Project26-2" / "Images",
        "baggage": ROOT / "Common_Project26-2" / "Baggage",
    },
]

# Help&Manual occasionally writes an attribute name without a value.  Keep
# the repair narrow so normal text containing '=' is never changed.
MISSING_ATTRIBUTE_VALUE = re.compile(
    r"\s(?P<name>href|styleclass)\s+(?=[A-Za-z_:][\w:.-]*\s*=)"
)
BARE_URL = re.compile(r"(?<![`])\b(?:https?|ssh|ftp)://[^\s<>\[\]{}\"'«»]+")
LINK_CAPTION_TOKEN = re.compile(r"<%LINK_CAPTION%>", re.IGNORECASE)
TEMPLATE_VARIABLE = re.compile(
    r"<%(?P<name>[^%<>]+)%>?",
    flags=re.UNICODE,
)
DEFAULT_BRANDNAME = 'МИС "Инфоклиника"/"Инфодент"'


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(node: ET.Element | None, tag: str) -> list[ET.Element]:
    if node is None:
        return []
    return [child for child in node if local_name(child.tag) == tag]


def child(node: ET.Element | None, tag: str) -> ET.Element | None:
    matches = children(node, tag)
    return matches[0] if matches else None


def slugify(value: str, fallback: str = "item", max_len: int = 64) -> str:
    # Build compact IDs for anchors.  File and directory names use path_name()
    # below, which preserves spaces from the source title.
    # Preserve the spelling and capitalization from <caption>/<title>.
    # Ordering is handled by sidebar_position, so names do not need a numeric
    # prefix such as "01-".
    value = value.strip()
    value = re.sub(r"[^\w.-]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    value = value or fallback
    value = value[:max_len].rstrip("-._")
    return value or fallback


def normalize_space(value: str | None, *, strip: bool = False) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value.replace("\xa0", " "))
    return value.strip() if strip else value


def path_name(value: str, fallback: str = "item", max_len: int = 120) -> str:
    """Make a readable Wiki.js filesystem name while preserving case."""
    value = normalize_space(value, strip=True)
    # Quotes are typography in a caption, not separators in a path.  Remove
    # them instead of turning them into the visible `-...-` seen in names like
    # `Модуль "SMS информирование"`.
    value = re.sub(r'["“”«»]', "", value)
    value = re.sub(r'[<>:/\\|?*\x00-\x1f]', "-", value)
    # Wiki.js treats dots in storage filenames as extension separators, so a
    # page such as ``Портал Инфоклиника.RU`` does not resolve at its URL.
    # Keep the title readable while using a path-safe visual separator.
    value = re.sub(r"\s*\.\s*", " - ", value)
    # Webpack treats `!` in a module path as a loader separator.  Help&Manual
    # uses it as a visible prefix in a few topic titles (for example
    # `! Правила именования переменных`), so it must not remain in a generated
    # Markdown filename.  The title/front matter still preserves the prefix.
    value = value.replace("!", "")
    value = re.sub(r" {2,}", " ", value).strip()
    value = value[:max_len].rstrip(" .")
    return value or fallback



def wiki_path_segment(value: str, fallback: str = "item", max_len: int = 100) -> str:
    """Build a safe and stable Wiki.js path segment from a Help&Manual href."""
    value = normalize_space(value, strip=True).lower()
    # Wiki.js derives the page path from the Markdown filename. Dots inside a
    # filename are ambiguous for its storage helper, so normalize them to '-'.
    value = value.replace(".", "-")
    value = re.sub(r"[^\w-]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    value = value or fallback
    value = value[:max_len].rstrip("-_")
    return value or fallback


def wiki_page_url(project_path: str, doc_path: Path) -> str:
    """Return the public Wiki.js URL for a generated Markdown page."""
    relative = (Path(project_path) / doc_path.with_suffix("")).as_posix()
    return "/" + quote(relative, safe="/._-~")


def wiki_frontmatter(
    *,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    modified: str | None = None,
) -> str:
    """Build metadata in the format used by Wiki.js storage exports."""
    timestamp = modified or datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    tag_value = ", ".join(tags or [])
    fields = [
        ("title", json.dumps(title, ensure_ascii=False)),
        ("description", json.dumps(description, ensure_ascii=False)),
        ("published", "true"),
        ("date", timestamp),
        ("tags", json.dumps(tag_value, ensure_ascii=False)),
        ("editor", "markdown"),
        ("dateCreated", timestamp),
    ]
    return "---\n" + "\n".join(f"{key}: {value}" for key, value in fields) + "\n---"

def resolve_template_variables(
    value: str,
    variables: dict[str, str] | None = None,
) -> str:
    """Resolve Help&Manual variables before escaping Markdown text.

    Help&Manual stores variables such as ``<%BRANDNAME%>`` as ordinary text
    in the XML.  If they are escaped together with the rest of the text, the
    generated Markdown contains implementation details (``&lt;%...%&gt;``)
    instead of useful documentation text.

    If a variable is not configured, its name is kept as ordinary text. This
    also handles Help&Manual conditional labels such as
    ``<%Клинико-диагностическая лаборатория%>``.
    """
    if not variables:
        return value

    return TEMPLATE_VARIABLE.sub(
        lambda match: variables.get(
            match.group("name").strip(),
            match.group("name").strip(),
        ),
        value,
    )


def escape_text(
    value: str | None,
    variables: dict[str, str] | None = None,
) -> str:
    """Escape source text so Markdown punctuation remains text, not syntax."""
    value = normalize_space(value)
    if not value:
        return ""

    value = resolve_template_variables(value, variables)

    # `%`-variables are not HTML tags. Protect them while escaping only when
    # the caller intentionally did not provide a replacement dictionary.
    variables_in_text: list[str] = []

    def protect_variable(match: re.Match[str]) -> str:
        variables_in_text.append(match.group(0))
        return f"\x00{len(variables_in_text) - 1}\x00"

    value = TEMPLATE_VARIABLE.sub(protect_variable, value)
    value = html.escape(value, quote=False)
    # MDX treats `{...}` as a JavaScript expression. Help topics use braces
    # as ordinary characters in message examples, formulas and scripts.
    value = value.replace("{", "&#123;").replace("}", "&#125;")
    for symbol in ("\\", "*", "_", "`", "[", "]"):
        value = value.replace(symbol, "\\" + symbol)
    # MDX/Wiki.js auto-detects bare URLs.  Documentation also contains
    # placeholders such as http://address:port, which are not valid URLs and
    # make the MDX URL resolver fail. Keep all bare URLs as visible code.
    def protect_url(match: re.Match[str]) -> str:
        token = match.group(0)
        trailing = ""
        while token and token[-1] in ".,;:!?)]":
            trailing = token[-1] + trailing
            token = token[:-1]
        return f"`{token}`{trailing}"

    value = BARE_URL.sub(protect_url, value)
    for index, variable in enumerate(variables_in_text):
        value = value.replace(f"\x00{index}\x00", variable)
    return value


def plain_text(node: ET.Element | None) -> str:
    return normalize_space("".join(node.itertext()), strip=True) if node is not None else ""


def topic_id_from_href(href: str | None) -> str:
    if not href:
        return ""
    href = href.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
    name = Path(href).name
    # A dot is a valid part of Help&Manual topic IDs (for example
    # ``Kassa_ffd-1.05``); only strip an actual XML/HTML extension.
    for extension in (".xml", ".htm", ".html"):
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return name


def anchor_slug(value: str | None) -> str:
    """Normalize Help&Manual anchor IDs identically for links and targets."""
    if not value:
        return ""
    value = value.strip().lstrip("#")
    return slugify(value, "section", 80)


def anchor_from_node(node: ET.Element) -> str:
    """Read an anchor ID from either the modern id or legacy name attribute."""
    return anchor_slug(node.get("id") or node.get("name"))


def split_internal_href(href: str) -> tuple[str, str]:
    """Split an internal Help&Manual href into topic reference and fragment."""
    href = (href or "").strip().replace("\\", "/")
    if "#" in href:
        topic_href, fragment = href.split("#", 1)
    else:
        topic_href, fragment = href, ""
    return topic_href.strip(), fragment.strip()


def parse_xml(path: Path) -> tuple[ET.Element | None, bool, str | None]:
    """Parse XML, retrying after a non-destructive repair of known bad attrs."""
    source = path.read_text(encoding="utf-8-sig")
    try:
        return ET.fromstring(source), False, None
    except ET.ParseError as first_error:
        repaired = MISSING_ATTRIBUTE_VALUE.sub(r' \g<name>="" ', source)
        if repaired == source:
            return None, False, str(first_error)
        try:
            return ET.fromstring(repaired), True, str(first_error)
        except ET.ParseError as second_error:
            return None, True, f"{first_error}; after repair: {second_error}"


def topic_title(path: Path) -> str:
    """Read a topic title for naming a generated Markdown file."""
    root, _, _ = parse_xml(path)
    if root is None:
        return path.stem
    return plain_text(child(root, "title")) or path.stem


def caption_of(ref: ET.Element) -> str:
    caption = child(ref, "caption")
    return plain_text(caption) or topic_id_from_href(ref.get("href"))



def collect_topicrefs(project: dict, stats: dict) -> tuple[list[dict], set[str]]:
    """Read Help&Manual TOC and map it to Wiki.js page paths."""
    root, recovered, error = parse_xml(project["toc"])
    if recovered:
        stats["recovered_xml"].append(str(project["toc"].relative_to(ROOT)))
    if root is None:
        raise RuntimeError(f"Не удалось разобрать TOC {project['toc']}: {error}")

    used_paths: set[Path] = set()
    records: list[dict] = []
    referenced: set[str] = set()

    def walk(node: ET.Element, parent_path: Path, depth: int, index: int) -> None:
        href = topic_id_from_href(node.get("href"))
        if not href:
            # Some TOC containers can have no page of their own. Do not lose
            # their descendants; keep walking at the current path.
            for child_index, item in enumerate(children(node, "topicref"), start=1):
                walk(item, parent_path, depth, child_index)
            return

        title = caption_of(node)
        referenced.add(href.lower())

        segment = wiki_path_segment(href, f"topic-{len(records) + 1}")
        doc_path = unique_path(parent_path / f"{segment}.md", used_paths)
        nested = children(node, "topicref")

        records.append(
            {
                "href": href,
                "title": title,
                "doc_path": doc_path,
                "sidebar_position": index,
                "is_category": bool(nested),
                "source_kind": "toc",
                "modified": node.get("modified"),
                "depth": depth,
            }
        )

        child_parent = doc_path.with_suffix("")
        for child_index, item in enumerate(nested, start=1):
            walk(item, child_parent, depth + 1, child_index)

    for root_index, item in enumerate(children(root, "topicref"), start=1):
        walk(item, Path(), 0, root_index)

    return records, referenced

def index_topics(topics_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in topics_dir.rglob("*.xml"):
        if "__history" in path.parts:
            continue
        result[path.stem.lower()] = path
    return result


def build_image_index(images_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact: dict[str, str] = {}
    by_name: defaultdict[str, list[str]] = defaultdict(list)
    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(images_dir).as_posix()
        exact[relative.lower()] = relative
        by_name[path.name.lower()].append(relative)
    return exact, dict(by_name)


def build_baggage_index(baggage_dir: Path | None) -> dict[str, str]:
    if baggage_dir is None or not baggage_dir.exists():
        return {}
    return {
        path.name.lower(): path.relative_to(baggage_dir).as_posix()
        for path in baggage_dir.rglob("*")
        if path.is_file()
    }


def format_span(
    value: str | None,
    style: str = "",
    variables: dict[str, str] | None = None,
) -> str:
    value = normalize_space(value)
    if not value or not value.strip():
        return ""
    leading = re.match(r"^\s*", value).group(0)
    trailing = re.search(r"\s*$", value).group(0)
    core_end = len(value) - len(trailing) if trailing else len(value)
    core = value[len(leading):core_end]
    rendered = escape_text(core, variables)
    style = style.lower()
    bold = "font-weight:bold" in style
    italic = "font-style:italic" in style
    if bold and italic:
        rendered = f"***{rendered}***"
    elif bold:
        rendered = f"**{rendered}**"
    elif italic:
        rendered = f"*{rendered}*"
    return leading + rendered + trailing



def image_url(ctx: dict, source: str) -> str | None:
    source = source.replace("\\", "/").lstrip("./")
    exact = ctx["image_exact"].get(source.lower())
    if exact is None:
        candidates = ctx["image_by_name"].get(Path(source).name.lower(), [])
        exact = candidates[0] if len(candidates) == 1 else None
    if exact is None:
        ctx["stats"]["missing_images"].add(source)
        return None
    return "/assets/help/{}/{}".format(
        ctx["asset_path"], quote(exact, safe="/%()_.,~-"),
    )

def render_inline(node: ET.Element, ctx: dict, *, allow_links: bool = True) -> str:
    pieces: list[str] = []
    if node.text:
        pieces.append(
            format_span(node.text, node.get("style", ""), ctx.get("variables"))
        )

    for item in list(node):
        tag = local_name(item.tag)
        if tag in {"text", "var"}:
            raw = item.text or "".join(item.itertext())
            pieces.append(
                format_span(raw, item.get("style", ""), ctx.get("variables"))
            )
        elif tag == "link" and allow_links:
            label = (
                render_inline(item, ctx, allow_links=False).strip()
                or topic_id_from_href(item.get("href"))
                or "Ссылка"
            )
            raw_href = (item.get("href", "") or "").strip()
            link_type = (item.get("type", "") or "").lower()

            # Help&Manual stores an internal anchor in two possible ways:
            #   <link href="topic" anchor="section">
            #   <link href="topic#section">
            # It can also use href="#section" for a jump inside the same page.
            internal_href, href_fragment = split_internal_href(raw_href)
            explicit_anchor = (
                item.get("anchor")
                or item.get("fragment")
                or item.get("bookmark")
                or ""
            )
            anchor = anchor_slug(explicit_anchor or href_fragment)

            is_external = (
                link_type == "weblink"
                or urlparse(raw_href).scheme in {"http", "https", "mailto", "ftp"}
            )

            if is_external:
                href = raw_href

                # Help&Manual often stores the real web address as the link
                # caption and leaves href as <%LINK_CAPTION%>.
                visible = plain_text(item)
                if LINK_CAPTION_TOKEN.search(href):
                    href = LINK_CAPTION_TOKEN.sub(
                        lambda _match: visible.strip(),
                        href,
                    )
                    if any(marker in href for marker in ("<", ">")):
                        pieces.append(label)
                        continue

                href = href.strip()
                if not urlparse(href).scheme:
                    baggage = ctx["baggage_by_name"].get(
                        Path(href).name.lower()
                    )
                    if baggage:
                        destination = "/files/help/{}/{}".format(
                            ctx["asset_path"],
                            quote(baggage, safe="/%()_.,~-"),
                        )
                        pieces.append(
                            f"[{label}]({destination})"
                            if destination
                            else label
                        )
                        continue

                    if re.match(
                        r"^(?:www\.|[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/|$)",
                        href,
                    ):
                        href = "https://" + href

                destination = quote(href, safe=":/?&=#@+;,%")
                pieces.append(
                    f"[{label}]({destination})"
                    if destination
                    else label
                )

            elif internal_href:
                topic_id = topic_id_from_href(internal_href)
                target = ctx["links"].get(topic_id.lower())

                if target:
                    destination = wiki_page_url(
                        ctx["project_path"],
                        target,
                    )
                    if anchor:
                        destination += f"#{anchor}"
                    pieces.append(f"[{label}]({destination})")
                else:
                    ctx["stats"]["broken_topic_links"].add(topic_id)
                    pieces.append(label)

            elif anchor:
                # href="#section" or an empty href plus anchor="section":
                # jump to a target on the current Wiki.js page.
                pieces.append(f"[{label}](#{anchor})")

            else:
                pieces.append(label)
        elif tag == "link":
            pieces.append(render_inline(item, ctx, allow_links=False))
        elif tag == "image":
            src = item.get("src", "")
            url = image_url(ctx, src) if src else None
            caption = plain_text(child(item, "caption"))
            alt_source = caption or Path(src).stem or "Изображение"
            alt = escape_text(alt_source)
            if url is None:
                pieces.append(f"*[{alt}: изображение не найдено]*")
            else:
                width = item.get("width", "")
                if width.isdigit():
                    alt_attr = html.escape(alt_source, quote=True)
                    pieces.append(f'<img src="{url}" alt="{alt_attr}" width="{width}" />')
                else:
                    pieces.append(f"![{alt}]({url})")
        elif tag == "anchor":
            anchor = anchor_from_node(item)
            if anchor:
                pieces.append(f'<a id="{anchor}" name="{anchor}"></a>')
        elif tag == "br":
            pieces.append("<br />")
        elif tag in {"list", "table"}:
            # Block elements are handled by render_blocks.
            pass
        else:
            pieces.append(render_inline(item, ctx, allow_links=allow_links))
        if item.tail:
            pieces.append(format_span(item.tail, variables=ctx.get("variables")))
    return "".join(pieces)


def table_to_md(table: ET.Element, ctx: dict) -> str:
    rows: list[list[str]] = []
    for row in children(table, "tr"):
        cells: list[str] = []
        for cell_node in list(row):
            if local_name(cell_node.tag) not in {"td", "th"}:
                continue
            value = render_blocks(list(cell_node), ctx, in_table=True)
            value = value.replace("|", "\\|").replace("\n", "<br />").strip()
            cells.append(value or " ")
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [" "] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def heading_level(style: str) -> int | None:
    normalized = style.lower().replace("_", "")
    match = re.search(r"heading(\d+)", normalized)
    return int(match.group(1)) if match else None


def comparable_heading_text(value: str) -> str:
    """Normalize rendered heading text for title de-duplication."""
    value = re.sub(r"<a\b[^>]*></a>", "", value, flags=re.IGNORECASE)
    value = html.unescape(value)
    value = re.sub(r"\\([\\`*_[\]{}()#+.!|>-])", r"\1", value)
    return normalize_space(value, strip=True).casefold()


def is_page_title_heading(value: str, title: str) -> bool:
    return comparable_heading_text(value) == comparable_heading_text(
        escape_text(title)
    )


def render_blocks(nodes: list[ET.Element], ctx: dict, in_table: bool = False) -> str:
    blocks: list[str] = []
    for node in nodes:
        tag = local_name(node.tag)
        if tag in {"header", "section", "body"}:
            nested = render_blocks(list(node), ctx, in_table=in_table)
            if nested:
                blocks.append(nested)
        elif tag == "para":
            nested_table = child(node, "table")
            if nested_table is not None:
                blocks.append(table_to_md(nested_table, ctx))
                continue
            text = render_inline(node, ctx).strip()
            if not text:
                continue
            level = heading_level(node.get("styleclass", ""))
            if level:
                # Keep Help&Manual anchors as raw HTML. Wiki.js renders
                # CommonMark and preserves these targets for old cross-links.
                anchor_ids = []
                for item in list(node):
                    if local_name(item.tag) != "anchor":
                        continue
                    anchor_id = anchor_from_node(item)
                    if anchor_id and anchor_id not in anchor_ids:
                        anchor_ids.append(anchor_id)

                for anchor_id in anchor_ids:
                    for rendered_anchor in (
                        f'<a id="{anchor_id}" name="{anchor_id}"></a>',
                        f'<a id="{anchor_id}"></a>',
                    ):
                        text = text.replace(
                            rendered_anchor,
                            "",
                            1,
                        ).strip()

                anchors = "\n".join(
                    f'<a id="{anchor_id}" name="{anchor_id}"></a>'
                    for anchor_id in anchor_ids
                )
                # The generated page already contains the document title as
                # H1.  Help&Manual usually repeats the same title as the
                # first Heading1 in the body; retaining it creates the
                # visible `# Title` / `## Title` pair in Wiki.js.
                if (
                    level == 1
                    and not ctx.get("page_heading_skipped")
                    and ctx.get("page_title")
                    and is_page_title_heading(text, ctx["page_title"])
                ):
                    ctx["page_heading_skipped"] = True
                    if anchors:
                        blocks.append(anchors)
                    continue
                heading = f"{'#' * min(level + 1, 6)} {text}"
                blocks.append(f"{anchors}\n{heading}" if anchors else heading)
            else:
                blocks.append(text)
        elif tag == "list":
            value = list_to_md(node, ctx)
            if value:
                blocks.append(value)
        elif tag == "table":
            value = table_to_md(node, ctx)
            if value:
                blocks.append(value)
        else:
            text = render_inline(node, ctx).strip()
            if text:
                blocks.append(text)
    separator = "<br />" if in_table else "\n\n"
    return separator.join(block for block in blocks if block)


def list_to_md(node: ET.Element, ctx: dict, level: int = 0) -> str:
    ordered = (node.get("type", "") or "").lower() == "ol"
    lines: list[str] = []
    counter = 1
    for item in list(node):
        tag = local_name(item.tag)
        if tag == "li":
            marker = f"{counter}." if ordered else "-"
            text_parts: list[ET.Element] = []
            nested_lists: list[ET.Element] = []
            for item_child in list(item):
                if local_name(item_child.tag) == "list":
                    nested_lists.append(item_child)
                else:
                    text_parts.append(item_child)
            temp = ET.Element("inline")
            temp.text = item.text
            for item_child in text_parts:
                temp.append(item_child)
            text = render_inline(temp, ctx).strip()
            lines.append(f"{'  ' * level}{marker} {text}".rstrip())
            counter += 1
            for nested in nested_lists:
                lines.append(list_to_md(nested, ctx, level + 1))
        elif tag == "list":
            lines.append(list_to_md(item, ctx, level + 1))
    return "\n".join(line for line in lines if line)



def convert_topic(source: Path, record: dict, ctx: dict) -> str:
    root, recovered, error = parse_xml(source)
    if recovered:
        ctx["stats"]["recovered_xml"].append(str(source.relative_to(ROOT)))
    if root is None:
        ctx["stats"]["parse_errors"].append(
            {"file": str(source.relative_to(ROOT)), "error": error}
        )
        title = record["title"] or source.stem
        fm = wiki_frontmatter(
            title=title,
            description="Ошибка конвертации исходного XML",
            tags=[ctx["project_title"], "Help&Manual"],
            modified=record.get("modified"),
        )
        return f"{fm}\n\n# {escape_text(title)}\n\n> Не удалось разобрать исходный XML: {error}\n"

    title = plain_text(child(root, "title")) or record["title"] or source.stem
    body = child(root, "body")
    ctx["page_title"] = title
    ctx["page_heading_skipped"] = False
    content = render_blocks(list(body) if body is not None else [], ctx).strip()
    content = re.sub(r"\n{3,}", "\n\n", content)

    description = f"Раздел справочной системы «{ctx['project_title']}»"
    tags = [ctx["project_title"], "Help&Manual"]
    if record.get("source_kind") == "unlisted":
        tags.append("Не включено в оглавление")

    fm = wiki_frontmatter(
        title=title,
        description=description,
        tags=tags,
        modified=record.get("modified"),
    )
    body_text = f"# {escape_text(title, ctx.get('variables'))}"
    if content:
        body_text += f"\n\n{content}"
    return f"{fm}\n\n{body_text}\n"


def copy_images(project: dict, export: Path) -> int:
    source_dir = project["images"]
    target_dir = export / "assets" / "help" / project["asset_path"]
    if not source_dir.exists():
        return 0
    count = 0
    for path in source_dir.rglob("*"):
        if path.is_file():
            target = target_dir / path.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    return count


def copy_baggage(project: dict, export: Path) -> int:
    source_dir = project.get("baggage")
    if source_dir is None or not source_dir.exists():
        return 0
    target_dir = export / "files" / "help" / project["asset_path"]
    count = 0
    for path in source_dir.rglob("*"):
        if path.is_file():
            target = target_dir / path.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    return count

def unique_path(path: Path, used: set[Path]) -> Path:
    if path not in used:
        used.add(path)
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1



def write_landing_pages(
    export: Path,
    project: dict,
    records: list[dict],
    *,
    include_unlisted: bool,
) -> None:
    """Create the project page and optional virtual-folder landing pages."""
    project_path = project["path"]

    top_level = sorted(
        (record for record in records if record.get("depth") == 0 and record["source_kind"] == "toc"),
        key=lambda record: record.get("sidebar_position", 9999),
    )
    links = [
        f"- [{escape_text(record['title'])}]({wiki_page_url(project_path, record['doc_path'])})"
        for record in top_level
    ]

    fm = wiki_frontmatter(
        title=project["title"],
        description=f"Справочная система «{project['title']}»",
        tags=[project["title"], "Help&Manual"],
    )
    content = [
        fm,
        "",
        f"# {escape_text(project['title'])}",
        "",
        "Выберите раздел справочной системы:",
        "",
        *links,
        "",
    ]
    (export / f"{project_path}.md").write_text("\n".join(content), encoding="utf-8")

    if include_unlisted:
        unlisted_path = export / project_path / "unlisted.md"
        unlisted_path.parent.mkdir(parents=True, exist_ok=True)
        unlisted_fm = wiki_frontmatter(
            title="Не включено в оглавление",
            description=f"Дополнительные темы проекта «{project['title']}»",
            tags=[project["title"], "Help&Manual", "Не включено в оглавление"],
        )
        unlisted_path.write_text(
            f"{unlisted_fm}\n\n# Не включено в оглавление\n\n"
            "Эти страницы присутствуют в проекте Help&Manual, но отсутствуют в основном оглавлении.\n",
            encoding="utf-8",
        )


def write_site_files(export: Path) -> None:
    """Create only real Wiki.js pages in the storage repository."""
    project_links = [
        f"- [{escape_text(project['title'])}](/" + project["path"] + ")"
        for project in PROJECTS
    ]

    home = "\n".join(
        [
            wiki_frontmatter(
                title="Справочная система ИнфоКлиника",
                description="Документация по настройке и работе системы",
                tags=["ИнфоКлиника", "Help&Manual"],
            ),
            "",
            "# Справочная система ИнфоКлиника",
            "",
            "Документация по настройке и работе системы.",
            "",
            "## Разделы",
            "",
            *project_links,
            "",
        ]
    )
    (export / "home.md").write_text(home, encoding="utf-8")

def convert(
    output: Path,
    *,
    brandname: str = DEFAULT_BRANDNAME,
    compilation_date: str | None = None,
) -> dict:
    if output.exists():
        # Keep Git history when regenerating directly into an initialized repo.
        for existing in output.iterdir():
            if existing.name == ".git":
                continue
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
    output.mkdir(parents=True, exist_ok=True)
    write_site_files(output)

    template_variables = {
        "BRANDNAME": brandname,
        "AUTHOR": brandname,
        "SDS": "Смарт Дельта Системс",
        "DATE": compilation_date or datetime.now().astimezone().strftime("%d.%m.%Y"),
    }

    stats = {
        "recovered_xml": [],
        "parse_errors": [],
        "missing_images": set(),
        "broken_topic_links": set(),
    }
    report: list[dict] = []
    all_written = 0
    all_images = 0

    for project in PROJECTS:
        topic_index = index_topics(project["topics"])
        records, referenced = collect_topicrefs(project, stats)

        unlisted = sorted(set(topic_index) - referenced)
        used_paths = {record["doc_path"] for record in records}
        for index, topic_key in enumerate(unlisted, start=1):
            source = topic_index[topic_key]
            title = topic_title(source)
            records.append(
                {
                    "href": source.stem,
                    "title": title,
                    "doc_path": unique_path(
                        Path("unlisted")
                        / f"{wiki_path_segment(source.stem, f'unlisted-{index}')}.md",
                        used_paths,
                    ),
                    "sidebar_position": index,
                    "is_category": False,
                    "source_kind": "unlisted",
                    "modified": None,
                    "depth": 1,
                }
            )

        links = {record["href"].lower(): record["doc_path"] for record in records}
        image_exact, image_by_name = build_image_index(project["images"])
        baggage_by_name = build_baggage_index(project.get("baggage"))

        project_root = output / project["path"]
        project_root.mkdir(parents=True, exist_ok=True)
        write_landing_pages(
            output,
            project,
            records,
            include_unlisted=bool(unlisted),
        )

        written = 0
        missing_topics: list[str] = []

        for record in records:
            source = topic_index.get(record["href"].lower())
            if source is None:
                missing_topics.append(record["href"])
                continue

            target = project_root / record["doc_path"]
            target.parent.mkdir(parents=True, exist_ok=True)

            ctx = {
                "links": links,
                "project_name": project["name"],
                "project_title": project["title"],
                "project_path": project["path"],
                "asset_path": project["asset_path"],
                "image_exact": image_exact,
                "image_by_name": image_by_name,
                "baggage_by_name": baggage_by_name,
                "stats": stats,
                "variables": template_variables,
            }
            target.write_text(convert_topic(source, record, ctx), encoding="utf-8")
            written += 1

        images = copy_images(project, output)
        baggage = copy_baggage(project, output)
        all_written += written
        all_images += images

        report.append(
            {
                "project": project["name"],
                "wiki_path": project["path"],
                "topics_total": len(topic_index),
                "topics_in_toc": len(records) - len(unlisted),
                "topics_written": written,
                "unlisted_topics": len(unlisted),
                "missing_from_topics": sorted(set(missing_topics)),
                "images_copied": images,
                "baggage_copied": baggage,
            }
        )

    result = {
        "topics_written": all_written,
        "images_copied": all_images,
        "recovered_xml": sorted(set(stats["recovered_xml"])),
        "parse_errors": stats["parse_errors"],
        "missing_images": sorted(stats["missing_images"]),
        "broken_topic_links": sorted(stats["broken_topic_links"]),
        "projects": report,
    }
    report_path = output.parent / f"{output.name}-conversion-report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result["report_path"] = str(report_path)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT, help="Каталог Wiki.js-экспорта")
    parser.add_argument(
        "--brandname",
        default=DEFAULT_BRANDNAME,
        help="Значение переменной Help&Manual BRANDNAME",
    )
    parser.add_argument(
        "--compilation-date",
        help="Значение переменной DATE (по умолчанию текущая дата в формате ДД.ММ.ГГГГ)",
    )
    args = parser.parse_args()
    result = convert(
        args.output.resolve(),
        brandname=args.brandname,
        compilation_date=args.compilation_date,
    )
    print(f"Готово. Тем: {result['topics_written']}; изображений: {result['images_copied']}.")
    print(f"Восстановлено XML: {len(result['recovered_xml'])}; ошибок разбора: {len(result['parse_errors'])}.")
    print(f"Ненайденных изображений: {len(result['missing_images'])}; битых ссылок: {len(result['broken_topic_links'])}.")
    print(f"Результат: {args.output.resolve()}")
    print(f"Отчёт: {result['report_path']}")


if __name__ == "__main__":
    main()
