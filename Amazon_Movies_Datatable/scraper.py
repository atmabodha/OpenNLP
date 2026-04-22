"""
Run AmazonPrimeMovies.ipynb cell‑by‑cell like Colab, but:
• Executes every '!pip …' line safely.
• Preserves Python indentation blocks.
• Disables SSL verification ONLY during notebook execution so
  corporate proxies with self‑signed certs cannot break TMDB calls.
"""

import sys, subprocess, shlex, nbformat, pathlib, textwrap, functools
import requests, urllib3, ssl

# ── 0.  Disable SSL (verify=False) just for notebook run ────────────
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_original_request = requests.Session.request
@functools.wraps(_original_request)
def _patched_request(self, method, url, **kw):
    kw.setdefault("verify", False)
    return _original_request(self, method, url, **kw)
requests.Session.request = _patched_request   # patch every requests call
requests.request = _patched_request          # patch the top‑level shortcut
# ─────────────────────────────────────────────────────────────────────

NOTEBOOK_PATH = pathlib.Path(__file__).with_name("AmazonPrimeMovies.ipynb")
CSV_OUT       = pathlib.Path(__file__).with_name("prime_movies_enriched.csv")


def _run_pip(line: str):
    args = [sys.executable, "-m"] + shlex.split(line.lstrip("!"))
    print("    ↳", " ".join(args))
    subprocess.run(args, check=True)


def _run_shell(line: str):
    print("    ↳", line)
    subprocess.run(line.lstrip("!"), shell=True, check=True)


def _exec_block(code: str, env: dict, cell_idx: int):
    try:
        exec(code, env)
    except Exception as e:
        print(f"❌  Exception in cell {cell_idx}: {e}")
        raise


def run_notebook():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"{NOTEBOOK_PATH.name} not found – copy your Colab notebook here."
        )

    print(f"📔 Executing notebook {NOTEBOOK_PATH.name}")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    env = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}

    for idx, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue

        print(f"\n── Cell {idx} {'─'*60}")
        lines = textwrap.dedent(cell.source).splitlines()
        buffer: list[str] = []

        def flush():
            if buffer:
                _exec_block("\n".join(buffer), env, idx)
                buffer.clear()

        for line in lines:
            if line.startswith("!pip"):
                flush(); _run_pip(line)
            elif line.startswith("!"):
                flush(); _run_shell(line)
            else:
                buffer.append(line)
        flush()

    print("\n✅ Notebook finished")
    return env.get("final_df")  # or None if notebook writes CSV only


# allow CLI execution
if __name__ == "__main__":
    run_notebook()
