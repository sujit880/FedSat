#!/usr/bin/env python3
"""
Generate a filled LaTeX results section by reading every
RESULTS/**/test_stats.csv and extracting the last row (final metrics).

Writes output to Latex/sections/results.tex (overwrites).

Run from repo root: python3 scripts/generate_results_tex.py
"""
import csv
import glob
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "RESULTS")
OUT_TEX = os.path.join(ROOT, "Latex", "sections", "results.tex")

PREFERRED = ["test_accuracy","test_acc","accuracy","acc","top1","global_accuracy"]


def find_csvs():
    pattern = os.path.join(RESULTS_DIR, "**", "test_stats.csv")
    return sorted(glob.glob(pattern, recursive=True))


def read_rows(path):
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows, (reader.fieldnames if reader.fieldnames else [])
    except Exception:
        return [], []


def pick_metric_from_rows(rows):
    """Return (value, key) by searching rows for preferred keys (last non-empty).
    Falls back to the last numeric field that is not 'round'."""
    if not rows:
        return None, None
    # search preferred keys for last non-empty occurrence
    for key in PREFERRED:
        for r in reversed(rows):
            if key in r and r[key] != "":
                try:
                    return float(r[key]), key
                except:
                    return r[key], key
    # fallback: look in last row for first numeric field excluding 'round'
    last = rows[-1]
    for k, v in last.items():
        if k.lower() == 'round':
            continue
        try:
            fv = float(v)
            return fv, k
        except:
            continue
    # final fallback: any numeric value in prior rows excluding 'round'
    for r in reversed(rows):
        for k, v in r.items():
            if k.lower() == 'round':
                continue
            try:
                fv = float(v)
                return fv, k
            except:
                continue
    return None, None


def make_entries(csv_paths):
    entries = []
    for p in csv_paths:
        rel = os.path.relpath(p, RESULTS_DIR)
        label = os.path.dirname(rel)
        rows, fields = read_rows(p)
        metric_val, metric_key = pick_metric_from_rows(rows)
        metrics = {}
        # gather metrics from the last row if present
        last = rows[-1] if rows else {}
        for k in ["test_accuracy", "test_acc", "accuracy", "acc", "top1", "precision", "recall", "f1", "global_accuracy"]:
            if k in last and last[k] != "":
                metrics[k] = last[k]
        entries.append({"path": p, "label": label, "metric": metric_val, "metric_key": metric_key, "metrics": metrics, "rows": rows})
    return entries


def render_tex(entries):
    header = (
        "\\section{Experiments}\n"
        "This section summarizes the final test statistics extracted from the experiment traces in RESULTS/.\n"
        "Each subsection below corresponds to a dataset-folder and lists per-run final metrics (last row of test\\_stats.csv).\n\n"
    )

    grouped = defaultdict(list)
    for e in entries:
        ds = e['label'].split(os.sep)[0] if e['label'] else 'misc'
        grouped[ds].append(e)

    parts = [header]
    for ds in sorted(grouped.keys()):
        parts.append(f"\\subsection{{{ds}}}\n")
        for e in sorted(grouped[ds], key=lambda x: x['label']):
            parts.append(f"\\subsubsection{{Run: {e['label']}}}\n")
            if e['metric'] is None:
                parts.append("No numeric test metric found in test\\_stats.csv for this run.\\\n\\n")
            else:
                parts.append(f"Final reported metric: {e['metric_key']} = {e['metric']}\\\n\\n")
            if e['metrics']:
                parts.append("\\begin{tabular}{ll}\\toprule\\nMetric & Value \\\\ \\midrule\\n")
                for k,v in e['metrics'].items():
                    parts.append(f"{k} & {v} \\\\ \\n")
                parts.append("\\bottomrule\\n\\end{tabular}\\n\\n")
            parts.append(f"\\noindent Source: \\texttt{{{e['path']}}}\\n\\n")

    summary = (
        "\\subsection{Summary}\n"
        "The table above lists per-run final test statistics extracted directly from each run's test\\_stats.csv. "
        "Raw JSON traces and figures are located under RESULTS/json_dump and RESULTS/figures respectively.\n"
    )
    parts.append(summary)
    return "".join(parts)


def main():
    csvs = find_csvs()
    if not csvs:
        print("No test_stats.csv found under RESULTS/")
        return 1
    entries = make_entries(csvs)
    tex = render_tex(entries)
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(tex)
    print(f"Wrote {OUT_TEX} with {len(entries)} runs.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
import csv, os, sys, textwrap, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "RESULTS")
OUT_TEX = os.path.join(ROOT, "Latex", "sections", "results.tex")

def find_csvs():
    paths = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "**", "test_stats.csv"), recursive=True):
        paths.append(p)
    return sorted(paths)

def read_last_row(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None, reader.fieldnames
        return rows[-1], reader.fieldnames

# choose metric column preference
PREFERRED = ["test_accuracy","accuracy","acc","test_acc","top1","global_accuracy"]

def pick_metric(row):
    if row is None:
        return None, {}
    # find preferred
    for k in PREFERRED:
        if k in row and row[k] != "":
            try:
                return float(row[k]), {k: row[k]}
            except:
                return row[k], {k: row[k]}
    # fallback: find first numeric field
    for k,v in row.items():
        try:
            fv = float(v)
            return fv, {k: v}
        except:
            continue
    return None, {}

def make_section_entries(csv_paths):
    entries = []
    for p in csv_paths:
        rel = os.path.relpath(p, RESULTS_DIR)
        parts = rel.split(os.sep)[:-1]  # dataset/settings/... directory names
        label = "/".join(parts)
        row, fields = read_last_row(p)
        metric_val, metric_map = pick_metric(row)
        # collect several common metrics if present
        metrics = {}
        if row:
            for key in ["test_accuracy","accuracy","acc","top1","f1","precision","recall","rounds_to_conv"]:
                if key in row and row[key] != "":
                    metrics[key] = row[key]
        entries.append({
            "path": p,
            "label": label,
            "metric": metric_val,
            "metric_map": metric_map,
            "all_metrics": metrics,
            "raw_row": row
        })
    return entries

def render_tex(entries):
    header = r"""\section{Experiments}
This section presents dataset and model settings and summarizes test statistics extracted from the experiment traces in RESULTS/. Each subsection below is generated from the per-run file test\_stats.csv stored with the run; numbers shown are the final test statistics (last row of the CSV).
"""
    body = [header]
    # group by dataset (first path component)
    grouped = {}
    for e in entries:
        ds = e["label"].split("/")[0] if e["label"] else "misc"
        grouped.setdefault(ds, []).append(e)
    for ds, lst in grouped.items():
        body.append(f"\\subsection{{{ds}}}\n")
        for e in lst:
            body.append(f"\\subsubsection{{Run: {e['label']}}}\n")
            if e["metric"] is None:
                body.append("No numeric test metric found in test\\_stats.csv for this run.\n\n")
                continue
            # brief paragraph
            body.append(f"Final reported metric: {list(e['metric_map'].keys())[0]} = {list(e['metric_map'].values())[0]}.\n\n")
            # small table of available metrics
            if e["all_metrics"]:
                body.append("\\begin{tabular}{ll}\n\\toprule\nMetric & Value \\\\\n\\midrule\n")
                for k,v in e["all_metrics"].items():
                    body.append(f"{k} & {v} \\\\\n")
                body.append("\\bottomrule\n\\end{tabular}\n\n")
            # path reference
            body.append(f"\\noindent Source: \\texttt{{{e['path']}}}\n\n")
    footer = "\n\\subsection{Summary}\nThe table above lists per-run final test statistics extracted directly from each run's test\\_stats.csv. Use these numbers to reproduce the paper tables/plots (raw JSON and figures are in RESULTS/json_dump and RESULTS/figures respectively).\n"
    body.append(footer)
    return "\n".join(body)

def main():
    csvs = find_csvs()
    if not csvs:
        print("No test_stats.csv files found under RESULTS/. Aborting.")
        sys.exit(1)
    entries = make_section_entries(csvs)
    tex = render_tex(entries)
    # write
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(tex)
    print(f"Wrote LaTeX results to {OUT_TEX}. Found {len(csvs)} runs.")

if __name__ == "__main__":
    main()