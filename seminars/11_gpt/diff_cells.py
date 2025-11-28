from IPython.core.magic import register_cell_magic, register_line_magic
from IPython.display import HTML
from IPython.utils.capture import capture_output
from difflib import HtmlDiff, unified_diff
from IPython import get_ipython
import shlex

# persistent stores (reuse if already present)
try:
    _saved_cells
except NameError:
    _saved_cells = {}

try:
    _saved_outputs
except NameError:
    _saved_outputs = {}

try:
    _saved_codes
except NameError:
    _saved_codes = {}

# helpers
def _normalize_whitespace(text):
    return "\n".join(" ".join(line.split()) for line in text.splitlines())

def _make_html_side_by_side(a_lines, b_lines, a_name, b_name, context_lines=3):
    return HtmlDiff().make_file(a_lines, b_lines, fromdesc=a_name, todesc=b_name, context=True, numlines=context_lines)

def _fetch_output_only(name):
    return _saved_outputs.get(name, None)

def _get_transformed_code(ip, cell_src):
    """
    Return IPython's transformed code for the given cell source string.
    Tries a few APIs for compatibility across IPython versions.
    """
    # prefer high-level API if present
    try:
        if hasattr(ip, "input_transformer_manager"):
            return ip.input_transformer_manager.transform_cell(cell_src)
    except Exception:
        pass
    # fallback to older transform_cell API
    try:
        if hasattr(ip, "transform_cell"):
            return ip.transform_cell(cell_src)
    except Exception:
        pass
    # last-resort: return original source (no transform)
    return cell_src

def _left_align_diff_html(html):
    """
    Modifies the HTMLDiff table so that all cells are left-aligned.
    We inject CSS that overrides the default 'align="right"' attribute.
    """
    css = """
    <style>
      table.diff td, table.diff th {
          text-align: left !important;
      }
      table.diff .diff_header {
          text-align: left !important;
      }
      table.diff .diff_next {
          text-align: left !important;
      }
    </style>
    """
    return css + html


# Main: save_cell that saves raw text, executes, and optionally saves the transformed code
@register_cell_magic
def save_cell(line, cell):
    """
    %%save_cell NAME [--capture-output] [--capture-code]
    Save the raw text of the cell under NAME, then execute the cell as usual.

    --capture-output : also capture runtime textual outputs into _saved_outputs[NAME]
    --capture-code   : capture the code that IPython actually transforms/executes into _saved_codes[NAME]
    """
    tokens = shlex.split(line)
    if not tokens:
        raise ValueError("Provide a name: %%save_cell NAME [--capture-output] [--capture-code]")
    name = tokens[0]
    capture_output_flag = "--capture-output" in tokens[1:]
    capture_code_flag = "--capture-code" in tokens[1:]

    # Save raw text
    _saved_cells[name] = cell

    ip = get_ipython()

    # If requested, capture the transformed code IPython would run
    if capture_code_flag:
        try:
            transformed = _get_transformed_code(ip, cell)
            # ensure it ends with newline for nicer diffs
            if transformed and not transformed.endswith("\n"):
                transformed += "\n"
            _saved_codes[name] = transformed
        except Exception as e:
            _saved_codes[name] = f"# <error capturing transformed code: {e}>\n{cell if cell.endswith(chr(10)) else cell + chr(10)}"

    # Execute cell: possibly capturing runtime outputs
    if capture_output_flag:
        with capture_output() as cap:
            res = ip.run_cell(cell)
        # build textual representation of captured outputs
        parts = []
        if cap.stdout:
            parts.append(cap.stdout)
        if cap.stderr:
            parts.append("[stderr]\n" + cap.stderr)
        if getattr(cap, "outputs", None):
            display_texts = []
            for out in cap.outputs:
                try:
                    data = out.get('data', {})
                    text = data.get('text/plain')
                    if text is None:
                        display_texts.append(repr(out))
                    else:
                        display_texts.append(text)
                except Exception:
                    display_texts.append(repr(out))
            if display_texts:
                parts.append("[display outputs]\n" + "\n".join(display_texts))
        if not parts and hasattr(res, "result") and res.result is not None:
            parts.append(repr(res.result))
        final = "\n".join(parts)
        if final and not final.endswith("\n"):
            final += "\n"
        _saved_outputs[name] = final
    else:
        # normal execution (outputs appear in output area)
        res = ip.run_cell(cell)

    info = f"Saved raw cell as '{name}' (length {len(cell)} chars)."
    if capture_code_flag:
        info += f" Captured transformed code length: {len(_saved_codes.get(name,'')).__str__()} chars."
    if capture_output_flag:
        info += f" Captured output length: {len(_saved_outputs.get(name,'')).__str__()} chars."
    if getattr(res, "error_in_exec", None):
        info += " Note: execution finished with an exception (see above)."
    return HTML(f"<b>{info}</b>")

# save_output remains available to explicitly capture outputs
@register_cell_magic
def save_output(line, cell):
    """
    %%save_output NAME
    Execute the cell and capture textual outputs into _saved_outputs[NAME].
    """
    name = line.strip()
    if not name:
        raise ValueError("Provide a name: %%save_output NAME")
    ip = get_ipython()
    with capture_output() as cap:
        res = ip.run_cell(cell)

    parts = []
    if cap.stdout:
        parts.append(cap.stdout)
    if cap.stderr:
        parts.append("[stderr]\n" + cap.stderr)
    if getattr(cap, "outputs", None):
        display_texts = []
        for out in cap.outputs:
            try:
                data = out.get('data', {})
                text = data.get('text/plain')
                if text is None:
                    display_texts.append(repr(out))
                else:
                    display_texts.append(text)
            except Exception:
                display_texts.append(repr(out))
        if display_texts:
            parts.append("[display outputs]\n" + "\n".join(display_texts))
    if not parts and hasattr(res, "result") and res.result is not None:
        parts.append(repr(res.result))
    final = "\n".join(parts)
    if final and not final.endswith("\n"):
        final += "\n"
    _saved_outputs[name] = final
    return HTML(f"<b>Saved output:</b> '{name}' (length {len(final)} chars)")

# diff_cells now prefers saved transformed code, then raw cell, then outputs
@register_cell_magic
def diff_cells(line, cell):
    """
    %%diff_cells NAME1 NAME2 [--ignore-space] [--unified] [--prefer raw|code|output]
    Diff two saved items. Preference order is:
      1) _saved_codes (if present for a name)
      2) _saved_cells
      3) _saved_outputs

    Options:
      --ignore-space : collapse whitespace differences
      --unified      : show plain unified text diff instead of HTML side-by-side
      --prefer TYPE  : force preference order; TYPE in {code, raw, output}
    """
    args = shlex.split(line)
    if len(args) < 2:
        return HTML("<b>Error:</b> usage: %%diff_cells NAME1 NAME2 [--ignore-space] [--unified] [--prefer raw|code|output]")

    # parse flags
    ignore_space = False
    unified_flag = False
    prefer = None
    names = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--ignore-space":
            ignore_space = True
        elif tok == "--unified":
            unified_flag = True
        elif tok == "--prefer":
            if i + 1 >= len(args):
                return HTML("<b>Error:</b> --prefer requires an argument (raw|code|output)")
            prefer = args[i+1].lower()
            if prefer not in {"code", "raw", "output"}:
                return HTML("<b>Error:</b> --prefer expects one of: code, raw, output")
            i += 1
        else:
            names.append(tok)
        i += 1

    if len(names) != 2:
        return HTML("<b>Error:</b> provide exactly two names (after removing flags)")

    name1, name2 = names

    def fetch_preference(name):
        # order of preference
        order = []
        if prefer == "code":
            order = ["code", "raw", "output"]
        elif prefer == "raw":
            order = ["raw", "code", "output"]
        elif prefer == "output":
            order = ["output", "code", "raw"]
        else:
            order = ["code", "raw", "output"]

        for typ in order:
            if typ == "code" and name in _saved_codes:
                return _saved_codes[name]
            if typ == "raw" and name in _saved_cells:
                return _saved_cells[name]
            if typ == "output" and name in _saved_outputs:
                return _saved_outputs[name]
        return None

    a = fetch_preference(name1)
    b = fetch_preference(name2)
    missing = [n for n, v in [(name1, a), (name2, b)] if v is None]
    if missing:
        return HTML(f"<b>Error:</b> Unknown saved name(s) under any selected preference: {', '.join(missing)}")

    if ignore_space:
        a_text = _normalize_whitespace(a)
        b_text = _normalize_whitespace(b)
    else:
        a_text = a
        b_text = b

    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()

    if unified_flag:
        udiff = "\n".join(unified_diff(a_lines, b_lines, fromfile=name1, tofile=name2, lineterm=""))
        if not udiff:
            udiff = "(no differences)\n"
        return HTML(f"<pre>{udiff}</pre>")
    else:
        html = _make_html_side_by_side(a_lines, b_lines, name1, name2)
        return HTML(_left_align_diff_html(html))


# convenience: line-magic to diff saved codes only (errors if missing)
@register_line_magic
def diff_codes(line):
    """
    %diff_codes NAME1 NAME2 [--ignore-space] [--unified]
    Diff the transformed code saved under _saved_codes for the two names.
    """
    args = shlex.split(line)
    if len(args) < 2:
        return HTML("<b>Error:</b> usage: %diff_codes NAME1 NAME2 [--ignore-space] [--unified]")

    ignore_space = False
    unified_flag = False
    names = []
    for tok in args:
        if tok == "--ignore-space":
            ignore_space = True
        elif tok == "--unified":
            unified_flag = True
        else:
            names.append(tok)
    if len(names) != 2:
        return HTML("<b>Error:</b> provide exactly two names (after removing flags)")
    n1, n2 = names

    a = _saved_codes.get(n1, None)
    b = _saved_codes.get(n2, None)
    missing = [n for n, v in [(n1, a), (n2, b)] if v is None]
    if missing:
        return HTML(f"<b>Error:</b> Transformed code missing for: {', '.join(missing)}. Use %%save_cell --capture-code NAME")

    if ignore_space:
        a_text = _normalize_whitespace(a)
        b_text = _normalize_whitespace(b)
    else:
        a_text = a
        b_text = b

    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()

    if unified_flag:
        udiff = "\n".join(unified_diff(a_lines, b_lines, fromfile=n1, tofile=n2, lineterm=""))
        if not udiff:
            udiff = "(no differences)\n"
        return HTML(f"<pre>{udiff}</pre>")
    else:
        html = _make_html_side_by_side(a_lines, b_lines, n1, n2)
        return HTML(_left_align_diff_html(html))
