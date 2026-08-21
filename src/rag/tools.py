"""Deterministic answers for the questions this model reliably gets wrong.

The model cannot use tools. Given a calculator and few-shot examples it still
answered "47 times 83" with "The answer is 48", and asked to follow a CALC()
pattern it produced "12 + 83 = 47". It cannot emit a call, and it could not
read the result back if it did -- shown a passage saying 60 days it answers
6 days.

So the tool runs here instead, before the model is involved, and its result is
handed to the caller as a fact to display rather than as text for the model to
paraphrase. The number the reader sees is computed, not generated.

Nothing here executes model output. Expressions are parsed to an AST and
evaluated against a whitelist of nodes; eval() is never called.
"""

import ast
import datetime
import math
import operator
import re
import statistics

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Words people use instead of symbols. Order matters: "divided by" has to be
# replaced before "by" would ever be looked at.
_WORDS = [
    (r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"),
    (r"\btimes\b", "*"), (r"\bplus\b", "+"), (r"\bminus\b", "-"),
    (r"\bover\b", "/"), (r"\bto the power of\b", "**"), (r"\bpower of\b", "**"), (r"\bsquared\b", "**2"),
    (r"\bpercent of\b", "/100*"), (r"\bx\b", "*"), (r"×", "*"), (r"÷", "/"),
]

MAX_POW = 1e6          # 2**999999 is a denial of service, not a calculation

# Named functions the expression parser accepts. Everything here is pure and
# cheap; nothing that touches the filesystem, the network or the interpreter.
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "log": math.log10, "ln": math.log, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "factorial": lambda n: math.factorial(int(n)),
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > MAX_POW):
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and (args[0] > 170 or args[0] < 0):
            raise ValueError("factorial out of range")
        return _FUNCS[node.func.id](*args)
    raise ValueError("unsupported expression")


def _tidy(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("not a finite number")
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:,.6g}"
    return f"{value:,}"


def arithmetic(question):
    """Compute an arithmetic question, or return None if it is not one.

    Returns (expression, answer) so the caller can show the working, not just
    the number -- a bare figure with no visible sum is exactly as unverifiable
    as one the model made up.
    """
    q = question.lower().strip().rstrip("?")
    for pattern, sym in _WORDS:
        q = re.sub(pattern, sym, q)

    # Drop the framing, keep the sum.
    q = re.sub(r"^(what\s+is|whats|what's|calculate|compute|how much is)\s*", "", q)
    q = q.replace(",", "").strip()

    # Must be only maths, and must contain an actual operator -- otherwise
    # "42" alone would come back as a calculation.
    if not q or not re.fullmatch(r"[\d\sa-z+\-*/%().,]+", q):
        return None
    # An operator or a named function -- otherwise "42" and "hello" both look
    # like calculations.
    if not re.search(r"[+\-*/%]", q) and not re.search(
            r"\b(" + "|".join(_FUNCS) + r")\s*\(", q):
        return None

    try:
        value = _eval(ast.parse(q, mode="eval"))
        return q, _tidy(value)
    except Exception:
        return None




# ---------------------------------------------------------------------------
# unit conversion
# ---------------------------------------------------------------------------

# Everything reduces to one base unit per dimension, so a conversion is two
# multiplications and no lookup table of pairs.
_UNITS = {
    # length -> metres
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0),
    "km": ("length", 1000.0), "inch": ("length", 0.0254),
    "inches": ("length", 0.0254), "in": ("length", 0.0254),
    "ft": ("length", 0.3048), "feet": ("length", 0.3048),
    "foot": ("length", 0.3048), "yard": ("length", 0.9144),
    "yards": ("length", 0.9144), "mile": ("length", 1609.344),
    "miles": ("length", 1609.344),
    # mass -> kilograms
    "mg": ("mass", 1e-6), "g": ("mass", 0.001), "gram": ("mass", 0.001),
    "grams": ("mass", 0.001), "kg": ("mass", 1.0), "kilogram": ("mass", 1.0),
    "kilograms": ("mass", 1.0), "lb": ("mass", 0.45359237),
    "lbs": ("mass", 0.45359237), "pound": ("mass", 0.45359237),
    "pounds": ("mass", 0.45359237), "oz": ("mass", 0.028349523),
    "ounce": ("mass", 0.028349523), "ounces": ("mass", 0.028349523),
    "tonne": ("mass", 1000.0), "ton": ("mass", 1000.0),
    # volume -> litres
    "ml": ("volume", 0.001), "l": ("volume", 1.0), "litre": ("volume", 1.0),
    "litres": ("volume", 1.0), "liter": ("volume", 1.0),
    "liters": ("volume", 1.0), "gallon": ("volume", 3.785411784),
    "gallons": ("volume", 3.785411784),
    # time -> seconds
    "second": ("time", 1.0), "seconds": ("time", 1.0), "sec": ("time", 1.0),
    "minute": ("time", 60.0), "minutes": ("time", 60.0), "min": ("time", 60.0),
    "hour": ("time", 3600.0), "hours": ("time", 3600.0),
    "day": ("time", 86400.0), "days": ("time", 86400.0),
    "week": ("time", 604800.0), "weeks": ("time", 604800.0),
    # data -> bytes
    "kb": ("data", 1e3), "mb": ("data", 1e6), "gb": ("data", 1e9),
    "tb": ("data", 1e12), "byte": ("data", 1.0), "bytes": ("data", 1.0),
}

# Temperature is affine, not a ratio, so it cannot ride the table above.
_TEMP = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f",
         "k": "k", "kelvin": "k"}


def _to_celsius(v, unit):
    return v if unit == "c" else (v - 32) * 5 / 9 if unit == "f" else v - 273.15


def _from_celsius(v, unit):
    return v if unit == "c" else v * 9 / 5 + 32 if unit == "f" else v + 273.15


def convert(question):
    """Unit conversion, or None."""
    q = question.lower().strip().rstrip("?")
    m = re.search(r"(-?[\d.,]+)\s*([a-z]+)\s*(?:in|to|into|=)\s*([a-z]+)", q)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    src, dst = m.group(2), m.group(3)

    if src in _TEMP and dst in _TEMP:
        out = _from_celsius(_to_celsius(value, _TEMP[src]), _TEMP[dst])
        return f"{_tidy(value)}°{src.upper()[0]}", f"{_tidy(out)}°{dst.upper()[0]}"

    if src in _UNITS and dst in _UNITS:
        dim_a, fa = _UNITS[src]
        dim_b, fb = _UNITS[dst]
        # Refusing a cross-dimension request matters: 5 km "in" kg is a typo or
        # a joke, and answering it with a number would be worse than declining.
        if dim_a != dim_b:
            return None
        return f"{_tidy(value)} {src}", f"{_tidy(value * fa / fb)} {dst}"
    return None


# ---------------------------------------------------------------------------
# percentages
# ---------------------------------------------------------------------------

def percentage(question):
    q = question.lower().strip().rstrip("?")
    q = re.sub(r"^(what\s+is|whats|what's)\s*", "", q).replace(",", "")

    m = re.search(r"([\d.]+)\s*%?\s*(?:percent|%)?\s*(?:increase|rise)\s*(?:on|of|from)?\s*([\d.]+)", q)
    if m:
        pct, base = float(m.group(1)), float(m.group(2))
        return f"{_tidy(base)} + {_tidy(pct)}%", _tidy(base * (1 + pct / 100))

    m = re.search(r"([\d.]+)\s*%?\s*(?:percent|%)?\s*(?:decrease|discount|off)\s*(?:on|of|from)?\s*([\d.]+)", q)
    if m:
        pct, base = float(m.group(1)), float(m.group(2))
        return f"{_tidy(base)} − {_tidy(pct)}%", _tidy(base * (1 - pct / 100))

    m = re.search(r"(?:from|change)\s*([\d.]+)\s*to\s*([\d.]+)", q)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a:
            delta = (b - a) / a * 100
            sign = "+" if delta >= 0 else ""
            return f"{_tidy(a)} → {_tidy(b)}", f"{sign}{_tidy(delta)}%"
    return None


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d %Y",
                 "%d %b %Y", "%b %d %Y")


def _parse_date(text):
    text = text.strip().replace(",", "")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def dates(question):
    q = question.lower().strip().rstrip("?")

    m = re.search(r"days?\s+between\s+(.+?)\s+and\s+(.+)$", q)
    if m:
        a, b = _parse_date(m.group(1)), _parse_date(m.group(2))
        if a and b:
            return f"{a} → {b}", f"{abs((b - a).days):,} days"

    m = re.search(r"(\d+)\s+days?\s+(after|before|from)\s+(.+)$", q)
    if m:
        n, direction, base = int(m.group(1)), m.group(2), _parse_date(m.group(3))
        if base:
            out = base + datetime.timedelta(days=-n if direction == "before" else n)
            return f"{base} {'−' if direction == 'before' else '+'} {n} days", out.strftime("%A, %-d %B %Y")

    m = re.search(r"what day (?:of the week )?(?:is|was)\s+(.+)$", q)
    if m:
        d = _parse_date(m.group(1))
        if d:
            return str(d), d.strftime("%A")
    return None


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def stats(question):
    q = question.lower().strip().rstrip("?")
    m = re.search(r"\b(mean|average|median|sum|total|min|minimum|max|maximum)\b"
                  r"[^\d\-]*((?:-?[\d.,]+[\s,]+){1,}-?[\d.,]+)", q)
    if not m:
        return None
    want = m.group(1)
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(2))]
    if len(nums) < 2:
        return None

    fn = {"mean": statistics.fmean, "average": statistics.fmean,
          "median": statistics.median, "sum": sum, "total": sum,
          "min": min, "minimum": min, "max": max, "maximum": max}[want]
    listed = ", ".join(_tidy(n) for n in nums)
    return f"{want} of {listed}", _tidy(fn(nums))


def solve(question):
    """Try every tool in turn. Returns a dict to display, or None.

    Order matters. Conversion and percentages are checked before plain
    arithmetic, because "15 percent of 240" and "5 km in miles" both contain
    something the calculator would otherwise take a partial swing at.
    """
    for name, fn in (("converter", convert),
                     ("percentage", percentage),
                     ("calendar", dates),
                     ("statistics", stats),
                     ("calculator", arithmetic)):
        hit = fn(question)
        if hit:
            expression, answer = hit
            return {"tool": name, "expression": expression, "answer": answer}
    return None
