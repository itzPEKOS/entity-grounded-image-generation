"""Make review.html: a page for PEOPLE to label every generated image as OK / wrong.

    python scripts/make_review.py outputs/eval_dev
    -> open outputs/eval_dev/review.html in a browser, label, press "Export labels.csv",
       save it as outputs/eval_dev/labels.csv

The page shows the request, the reference photos and every attempt image, but NOT the system's
verdict, so the label is your own judgement. Labels are kept in the browser while you work
(close and reopen is fine); "Import" loads a labels.csv back (e.g. to continue, or a friend's file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from egig.evaluation import TAGS, attempt_numbers, find_runs, load_run  # noqa: E402


def collect(root: Path) -> list:
    data = []
    for d in find_runs(root):
        run = load_run(d)
        rel = lambda p: Path(p).resolve().relative_to(root.resolve()).as_posix()  # noqa: E731
        entities = []
        for e in run.spec.entities:
            refs = [rel(r.crop_path) for r in e.references if Path(r.crop_path).exists()]
            entities.append({"name": e.name, "count": e.count, "refs": refs})
        attempts = [{"n": n, "img": rel(run.image(n))} for n in attempt_numbers(d)]
        if attempts:
            data.append({"id": run.id, "prompt": run.prompt, "entities": entities, "attempts": attempts})
    return data


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Label images</title>
<style>
:root{color-scheme:light dark;--line:#8884;--ok:#1a7f37;--bad:#c62828}
body{font:15px/1.45 system-ui,sans-serif;margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:Canvas;border-bottom:1px solid var(--line);padding:10px 0;display:flex;
  gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:17px} .run{border:1px solid var(--line);border-radius:8px;margin:18px 0;padding:12px 14px}
.prompt{font-size:18px;font-weight:600;margin:2px 0 8px}.id{opacity:.6;font-size:13px}
.refs{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px}.refs figure{margin:0}
.refs img{height:84px;border-radius:4px;margin-right:4px}.refs figcaption{font-size:13px;opacity:.75}
.attempts{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.att{border:2px solid var(--line);border-radius:8px;padding:8px}.att.ok{border-color:var(--ok)}.att.wrong{border-color:var(--bad)}
.att img{width:100%;border-radius:4px;display:block}.choice{display:flex;gap:14px;margin:6px 0;font-weight:600}
.tags{display:none;flex-wrap:wrap;gap:4px 12px;font-size:13px}.att.wrong .tags{display:flex}
.att input[type=text]{width:100%;box-sizing:border-box;margin-top:6px}
button{font:inherit;padding:5px 12px;cursor:pointer}.done{opacity:.7}
</style></head><body>
<header><b>Label images</b><span id="progress"></span>
<label>Your name <input id="labeler" size="10"></label>
<button id="export">Export labels.csv</button>
<label><button onclick="document.getElementById('imp').click()">Import labels.csv</button>
<input id="imp" type="file" accept=".csv" hidden></label>
<span class="done">Judge each image yourself: does it show what the request asks for? Click an image to open it full size.</span>
</header><main id="list"></main>
<script>
const DATA = __DATA__, TAGS = __TAGS__, KEY = "egig-labels:" + __KEY__;
let labels = JSON.parse(localStorage.getItem(KEY) || "{}");
const $ = s => document.querySelector(s);
$("#labeler").value = localStorage.getItem(KEY + ":who") || "";
$("#labeler").oninput = e => localStorage.setItem(KEY + ":who", e.target.value);
const save = () => { localStorage.setItem(KEY, JSON.stringify(labels)); progress(); };
const total = DATA.reduce((n, r) => n + r.attempts.length, 0);
function progress(){ const n = Object.values(labels).filter(l => l.human).length;
  $("#progress").textContent = `${n} / ${total} labelled`; }
function el(tag, attrs, ...kids){ const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) k === "class" ? e.className = v : e.setAttribute(k, v);
  for (const k of kids) e.append(k); return e; }
function render(){
  const list = $("#list"); list.innerHTML = "";
  for (const run of DATA){
    const box = el("section", {class: "run"}, el("div", {class: "id"}, run.id), el("div", {class: "prompt"}, run.prompt));
    const refs = el("div", {class: "refs"});
    for (const e of run.entities){
      const fig = el("figure", {}); for (const r of e.refs) fig.append(el("img", {src: r, loading: "lazy"}));
      fig.append(el("figcaption", {}, `${e.count > 1 ? e.count + " x " : ""}${e.name}` + (e.refs.length ? " (real photos)" : "")));
      refs.append(fig); }
    box.append(refs);
    const grid = el("div", {class: "attempts"});
    for (const a of run.attempts){
      const k = run.id + "|" + a.n, lab = labels[k] || (labels[k] = {human: "", tags: [], note: ""});
      const card = el("div", {class: "att " + (lab.human || "")});
      const link = el("a", {href: a.img, target: "_blank"}, el("img", {src: a.img, loading: "lazy"}));
      const choice = el("div", {class: "choice"}, `Attempt ${a.n}:`);
      for (const [v, text] of [["ok", "OK"], ["wrong", "Wrong"]]){
        const r = el("input", {type: "radio", name: k}); r.checked = lab.human === v;
        r.onchange = () => { lab.human = v; card.className = "att " + v; save(); };
        choice.append(el("label", {}, r, " " + text)); }
      const tags = el("div", {class: "tags"});
      for (const [t, help] of Object.entries(TAGS)){
        const c = el("input", {type: "checkbox", title: help}); c.checked = lab.tags.includes(t);
        c.onchange = () => { lab.tags = c.checked ? [...lab.tags, t] : lab.tags.filter(x => x !== t); save(); };
        tags.append(el("label", {title: help}, c, " " + t)); }
      const note = el("input", {type: "text", placeholder: "note (optional): what is wrong?"}); note.value = lab.note;
      note.oninput = () => { lab.note = note.value; save(); };
      card.append(link, choice, tags, note); grid.append(card); }
    box.append(grid); list.append(box); }
  progress(); }
const q = s => /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
$("#export").onclick = () => {
  const who = $("#labeler").value.trim(), rows = [["run_id", "attempt", "human", "tags", "note", "labeler"]];
  for (const run of DATA) for (const a of run.attempts){ const l = labels[run.id + "|" + a.n] || {};
    rows.push([run.id, String(a.n), l.human || "", (l.tags || []).join(";"), l.note || "", who]); }
  const blob = new Blob(["﻿" + rows.map(r => r.map(q).join(",")).join("\n") + "\n"], {type: "text/csv"});
  const a = el("a", {href: URL.createObjectURL(blob), download: "labels.csv"}); document.body.append(a); a.click(); a.remove(); };
function parseCSV(text){ const rows = []; let row = [], cur = "", inq = false;
  for (let i = 0; i < text.length; i++){ const ch = text[i];
    if (inq){ if (ch === '"'){ if (text[i + 1] === '"'){ cur += '"'; i++; } else inq = false; } else cur += ch; }
    else if (ch === '"') inq = true; else if (ch === ","){ row.push(cur); cur = ""; }
    else if (ch === "\n"){ row.push(cur.replace(/\r$/, "")); rows.push(row); row = []; cur = ""; } else cur += ch; }
  if (cur || row.length){ row.push(cur); rows.push(row); } return rows; }
$("#imp").onchange = async e => { const f = e.target.files[0]; if (!f) return;
  const rows = parseCSV((await f.text()).replace(/^﻿/, "")), head = rows.shift(), ix = n => head.indexOf(n);
  let n = 0; for (const r of rows){ if (!r[ix("run_id")] || !r[ix("human")]) continue;
    labels[r[ix("run_id")] + "|" + r[ix("attempt")]] = {human: r[ix("human")],
      tags: (r[ix("tags")] || "").split(";").filter(Boolean), note: r[ix("note")] || ""}; n++; }
  save(); render(); alert(`Imported ${n} labels`); e.target.value = ""; };
render();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, help="folder holding the run folders (e.g. outputs/eval_dev)")
    args = ap.parse_args()
    data = collect(args.runs)
    if not data:
        print(f"no runs with images under {args.runs}")
        return 1
    js = lambda v: json.dumps(v, ensure_ascii=False).replace("</", "<\\/")  # noqa: E731  (safe inside <script>)
    page = (PAGE.replace("__DATA__", js(data)).replace("__TAGS__", js(TAGS))
            .replace("__KEY__", js(args.runs.resolve().name)))
    out = args.runs / "review.html"
    out.write_text(page, encoding="utf-8")
    n = sum(len(r["attempts"]) for r in data)
    print(f"wrote {out}: {len(data)} prompts, {n} images to label")
    return 0


if __name__ == "__main__":
    sys.exit(main())
