"""Run the Python the model wrote, and report what actually happened.

The model saw 2.2B tokens of code -- roughly 450x less than StarCoder-1B. It
produces code that reads plausibly and often does not run: an off-by-one, a
name that was never defined, a base case that recurses forever. Prose hides
that. Execution does not.

So the same move as the calculator: stop asking the reader to trust the model
and check it instead. Extract the code block, run it in a subprocess with a
timeout, and put the verdict on screen next to the reply. A card saying
"NameError: 'n' is not defined" is worth more than a paragraph claiming the
function is correct.

This runs code the MODEL wrote in response to a user prompt, so treat it as
untrusted: separate process, no network, hard timeout, output truncated, and
killed by process group so a fork cannot outlive it.
"""

import os
import re
import subprocess
import sys
import tempfile

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

# Where unfenced code starts. This model mostly does not fence its output --
# it emits "def reverse_string(s):" straight into the prose -- so requiring
# fences meant the check never ran on the replies that needed it most.
_CODE_START = re.compile(
    r"^\s*(?:def |class |import |from \w+ import|print\(|for |while |if |@)",
    re.M)

TIMEOUT = 8
MAX_OUTPUT = 2000

# Imports that only make sense for reaching off the machine or changing it.
# The sandbox below is the real boundary; this is a cheap first refusal so the
# obvious cases never reach it.
_BLOCKED = re.compile(
    r"^\s*(?:import|from)\s+(socket|urllib|requests|http|ftplib|smtplib|"
    r"telnetlib|shutil|subprocess|multiprocessing|ctypes)\b", re.M)


def extract_code(text):
    """The last fenced Python block, or None.

    The last one on purpose: when the model revises itself mid-reply, the final
    block is the one it is standing behind.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        code = blocks[-1].strip()
        return code or None

    # Unfenced: take everything from the first line that opens a statement.
    # Trailing prose is left in deliberately rather than guessed at -- if it
    # is really prose the snippet fails to parse, and "SyntaxError" is the
    # honest verdict on a reply that mixed the two.
    m = _CODE_START.search(text)
    if not m:
        return None
    code = text[m.start():].strip()
    return code if len(code.splitlines()) >= 2 else None


def run(code, stdin=""):
    """Execute `code` and return what happened. Never raises."""
    if _BLOCKED.search(code):
        return {"status": "refused",
                "detail": "uses networking or process control"}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snippet.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", path],
                input=stdin, capture_output=True, text=True, timeout=TIMEOUT,
                cwd=tmp,
                # A fresh session, so killing the group takes any child with it.
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp,
                     "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout",
                    "detail": f"still running after {TIMEOUT}s"}
        except Exception as e:                      # noqa: BLE001
            return {"status": "error", "detail": str(e)[:200]}

    if proc.returncode == 0:
        out = proc.stdout.strip()
        return {"status": "ok", "output": out[:MAX_OUTPUT] or "(no output)"}

    # The traceback's last line is the part a reader acts on; the frames above
    # it are noise once the code is on screen directly above.
    err = proc.stderr.strip().splitlines()
    return {"status": "failed",
            "detail": err[-1][:300] if err else f"exited {proc.returncode}"}


def check(reply):
    """Extract and run the code in `reply`. None when there is none."""
    code = extract_code(reply)
    if not code:
        return None
    result = run(code)
    result["code"] = code[:MAX_OUTPUT]
    return result
