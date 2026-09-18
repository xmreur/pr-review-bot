import re
from dataclasses import dataclass


@dataclass
class DiffLine:
    path: str
    line: int
    side: str
    content: str
    changed: bool


@dataclass
class ParsedFile:
    path: str
    lines: list[DiffLine]


HUNK_RE = re.compile(
    r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
)


def parse_patch(path: str, patch: str) -> ParsedFile:
    lines = []

    old_line = None
    new_line = None

    for raw_line in patch.splitlines():
        hunk = HUNK_RE.match(raw_line)

        if hunk:
            old_line = int(hunk.group(1))
            new_line = int(hunk.group(2))
            continue

        if old_line is None or new_line is None:
            continue

        # Added line
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            lines.append(
                DiffLine(
                    path=path,
                    line=new_line,
                    side="RIGHT",
                    content=raw_line[1:],
                    changed=True,
                )
            )
            new_line += 1
            continue

        # Deleted line
        if raw_line.startswith("-") and not raw_line.startswith("---"):
            lines.append(
                DiffLine(
                    path=path,
                    line=old_line,
                    side="LEFT",
                    content=raw_line[1:],
                    changed=True,
                )
            )
            old_line += 1
            continue

        # Context line
        if raw_line.startswith(" "):
            lines.append(
                DiffLine(
                    path=path,
                    line=new_line,
                    side="RIGHT",
                    content=raw_line[1:],
                    changed=False,
                )
            )
            old_line += 1
            new_line += 1
            continue

        # "\ No newline at end of file"
        if raw_line.startswith("\\"):
            continue

    return ParsedFile(
        path=path,
        lines=lines,
    )


def parse_files(files: list[dict]) -> list[ParsedFile]:
    parsed = []

    for file in files:
        patch = file.get("patch")

        if not patch:
            continue

        parsed.append(
            parse_patch(
                file["filename"],
                patch,
            )
        )

    return parsed


def build_diff(files: list[dict]) -> str:
    chunks = []

    for file in files:
        filename = file["filename"]
        patch = file.get("patch")

        if not patch:
            continue

        chunks.append(
            f"--- FILE: {filename} ---\n{patch}"
        )

    return "\n\n".join(chunks)


def build_numbered_diff(files: list[dict]) -> str:
    parsed_files = parse_files(files)

    chunks = []

    for parsed in parsed_files:
        lines = [
            f"--- FILE: {parsed.path} ---"
        ]

        for line in parsed.lines:
            if line.side == "LEFT":
                prefix = "-"
            elif line.changed:
                prefix = "+"
            else:
                prefix = " "

            lines.append(
                f"{prefix} {line.line:5d} | {line.content}"
            )

        chunks.append("\n".join(lines))

    return "\n\n".join(chunks)


def split_patch_into_hunks(
    path: str,
    patch: str,
) -> list[dict]:
    """
    Split a GitHub unified diff patch into individual hunks.

    Each returned item has the same shape as a GitHub file object,
    containing only one @@ hunk.

    Example:

        [
            {
                "filename": "src/foo.ts",
                "patch": "@@ -10,5 +10,6 @@\\n..."
            },
            {
                "filename": "src/foo.ts",
                "patch": "@@ -50,4 +51,5 @@\\n..."
            }
        ]

    Keeping the @@ header is important because it contains the
    original source line numbers.
    """

    hunks: list[dict] = []

    current: list[str] | None = None

    for line in patch.splitlines():
        if line.startswith("@@"):
            if current is not None:
                hunks.append(
                    {
                        "filename": path,
                        "patch": "\n".join(current),
                    }
                )

            current = [line]
            continue

        # Ignore diff metadata before the first hunk.
        if current is None:
            continue

        current.append(line)

    if current is not None:
        hunks.append(
            {
                "filename": path,
                "patch": "\n".join(current),
            }
        )

    return hunks


def find_comment_line(
    parsed_files: list[ParsedFile],
    path: str,
    line: int,
    side: str,
):
    for parsed in parsed_files:
        if parsed.path != path:
            continue

        for diff_line in parsed.lines:
            if (
                diff_line.line == line
                and diff_line.side == side
                and diff_line.changed
            ):
                return diff_line

    return None


def extract_context(
    content: str,
    start_line: int,
    end_line: int,
    context_lines: int = 30,
    changed_lines: set[int] | None = None,
) -> str:
    """
    Extract source lines surrounding a changed region.

    Line numbers are 1-based.

    If changed_lines is given, only those lines get the '>>>'
    marker. Otherwise the whole [start_line, end_line] range is
    marked (legacy behavior, kept for compatibility).
    """

    source_lines = content.splitlines()

    start = max(1, start_line - context_lines)
    end = min(
        len(source_lines),
        end_line + context_lines,
    )

    output = []

    for number in range(start, end + 1):
        line = source_lines[number - 1]

        if changed_lines is not None:
            marked = number in changed_lines
        else:
            marked = start_line <= number <= end_line

        marker = ">>>" if marked else "   "

        output.append(
            f"{marker} {number:5d} | {line}"
        )

    return "\n".join(output)


# ------------------------------------------------------------------
# Context-budget helpers (false-positive / overflow optimization)
# ------------------------------------------------------------------

# Files that are never worth LLM review budget. Reviewing them only
# burns context and produces false positives on generated code.
SKIP_SUFFIXES = (
    ".min.js",
    ".min.css",
    ".map",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pdf",
    ".lock",
)

SKIP_BASENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
}

SKIP_DIRS = (
    "vendor/",
    "node_modules/",
    "dist/",
    "build/",
    "__pycache__/",
)

TEST_PATH_RE = re.compile(
    r"(^|/)(test|tests|__tests__|__mocks__|e2e)(/|$)"
    r"|\.(test|spec)\."
    r"|_test\."
    r"|(^|/)(conftest|fixtures?)(/|$)",
    re.IGNORECASE,
)


def is_test_file(filename: str) -> bool:
    """Test/spec files: reviewing them at src severity only adds noise."""
    return bool(TEST_PATH_RE.search(filename.lower()))


def normalize_code(text: str) -> str:
    """Collapse whitespace/quotes so evidence fuzzy-matches diff lines."""
    text = text.strip().strip("`'\"")
    text = re.sub(r"\s+", " ", text)
    return text.strip("`'\" ")


def is_skippable_file(filename: str, patch: str | None = None) -> bool:
    name = filename.lower()

    if any(name.endswith(s) for s in SKIP_SUFFIXES):
        return True

    base = name.rsplit("/", 1)[-1]

    if base in SKIP_BASENAMES:
        return True

    if any(part in SKIP_DIRS for part in [name]):
        return True

    # Huge auto-generated diffs (e.g. lockfile updates that slipped
    # through) are not reviewable in a 16k window.
    if patch is not None and len(patch) > 40000:
        return True

    return False


def merge_ranges(
    changed: list[int],
    pad: int,
    file_len: int,
) -> list[tuple[int, int]]:
    """Merge changed lines + padding into disjoint ranges."""
    if not changed:
        return []

    points = sorted(set(changed))
    ranges: list[tuple[int, int]] = []

    cur_start = max(1, points[0] - pad)
    cur_end = min(file_len, points[0] + pad)

    for line in points[1:]:
        start = max(1, line - pad)
        end = min(file_len, line + pad)

        # Overlapping or adjacent (gap of 1 context line is not
        # worth a "... gap ..." break) -> merge.
        if start <= cur_end + 2:
            cur_end = max(cur_end, end)
        else:
            ranges.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    ranges.append((cur_start, cur_end))

    return ranges


SCOPE_RE = re.compile(
    r"^\s*(def |class |func |function |fn |public |private |protected |"
    r"if |for |while |switch |match |try:?|catch |export (default )?function |"
    r"export (default )?class |const .*=>|let .*=>)"
)


def find_scope_start(
    source_lines: list[str],
    line_no: int,
    max_scan: int = 60,
) -> int | None:
    """
    Heuristic: walk backwards from line_no to find the enclosing
    definition (def/class/function/...). Returns 1-based line
    number or None. Language-agnostic, indentation-aware.
    """
    idx = line_no - 1

    if idx < 0 or idx >= len(source_lines):
        return None

    try:
        current_indent = len(source_lines[idx]) - len(
            source_lines[idx].lstrip()
        )
    except Exception:
        return None

    for back in range(idx - 1, max(-1, idx - max_scan - 1), -1):
        text = source_lines[back]
        stripped = text.lstrip()

        if not stripped:
            continue

        indent = len(text) - len(stripped)

        if indent < current_indent and (
            SCOPE_RE.match(stripped)
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("function ")
            or stripped.startswith("fn ")
        ):
            return back + 1

    return None


def extract_focused_context(
    content: str,
    changed_lines: list[int] | set[int],
    pad: int = 12,
    max_chars: int = 6000,
    max_line_len: int = 500,
) -> str:
    """
    Compact, high-signal context for one file.

    - Merges overlapping pad windows (no duplicated context).
    - Marks ONLY truly changed lines with '>>>'.
    - Prepends the enclosing function/class header when the
      changed region sits deep inside a scope, so the model sees
      the signature without paying for the whole body.
    - Caps output at max_chars with per-range truncation.
    """
    source_lines = content.splitlines()
    total = len(source_lines)

    if not source_lines or not changed_lines:
        return ""

    changed_set = set(changed_lines)
    changed_sorted = sorted(changed_set)

    ranges = merge_ranges(changed_sorted, pad, total)

    output: list[str] = []
    used_chars = 0

    # Collect scope headers first (cheap, high value).
    headers: list[int] = []

    for start, end in ranges:
        scope = find_scope_start(source_lines, start)

        if (
            scope is not None
            and scope < start
            and scope not in headers
            and all(
                not (s <= scope <= e)
                for s, e in ranges
            )
        ):
            headers.append(scope)

    if headers:
        output.append("DEFINITIONS:")

        for h in sorted(headers)[:5]:
            line = source_lines[h - 1][:max_line_len]

            output.append(f"    {h:5d} | {line}")

            used_chars += len(line) + 16

        output.append("")

    for i, (start, end) in enumerate(ranges):
        if i > 0:
            output.append("    ...")

        for number in range(start, end + 1):
            raw = source_lines[number - 1]

            if len(raw) > max_line_len:
                raw = raw[:max_line_len] + "…[truncated]"

            marker = ">>>" if number in changed_set else "   "
            rendered = f"{marker} {number:5d} | {raw}"

            # Hard cap: stop adding lines, note truncation so the
            # model knows context is incomplete (better than silent
            # cutoff, which causes hallucinations).
            if used_chars + len(rendered) > max_chars:
                output.append(
                    "    ...[context truncated: file too large]"
                )
                return "\n".join(output)

            output.append(rendered)
            used_chars += len(rendered) + 1

    return "\n".join(output)

def get_changed_line_range(
    patch: str,
) -> tuple[int, int] | None:
    """
    Return the minimum and maximum new-file line numbers
    represented by changed '+' lines in a patch.
    """

    parsed = parse_patch("", patch)

    changed = [
        line.line
        for line in parsed.lines
        if line.changed and line.side == "RIGHT"
    ]

    if not changed:
        return None

    return min(changed), max(changed)