"""Build 23L-0871_VisionGuard.pdf from report.html.

Inlines the terminal evidence files (<!--EV:name.txt-->), fills in name / commit / date,
then prints the page to PDF with headless Chrome.

  python docs/report/build_report.py --name "Your Full Name"
"""
import argparse
import html
import re
import subprocess
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="[YOUR FULL NAME]")
    ap.add_argument("--out", default=str(ROOT / "23L-0871_VisionGuard.pdf"))
    a = ap.parse_args()

    src = (HERE / "report.html").read_text(encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    def evidence(m):
        text = (HERE / "evidence" / m.group(1)).read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;]*m", "", text).replace("\ufeff", "")
        return f"<pre>{html.escape(text.rstrip())}</pre>"

    out = re.sub(r"<!--EV:([\w.]+)-->", evidence, src)
    out = (out.replace("{{STUDENT_NAME}}", html.escape(a.name))
              .replace("{{COMMIT}}", commit)
              .replace("{{DATE}}", date.today().strftime("%d %B %Y")))
    built = HERE / "report_built.html"
    built.write_text(out, encoding="utf-8")

    chrome = next(p for p in CHROME if Path(p).is_file())
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={a.out}", built.as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {a.out} (commit {commit[:7]})")


if __name__ == "__main__":
    main()
