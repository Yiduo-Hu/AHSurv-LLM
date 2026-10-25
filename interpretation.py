#!/usr/bin/env python
"""LLM interpretation of the selected genes: function, pathway, theme.

Three LLM calls in a chain: annotate what a gene does, place it in the
pathways it drives, then read those pathways back as broad biological
themes. The prompts are sketches: adjust them to the ontology you use.
"""

import csv
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

# ---- Credentials: keep them in the environment, never in the file ----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "YOUR_LLM_BASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "YOUR_LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "YOUR_LLM_MODEL")

# ---- Run settings ----
GENE_LIST_FILE = Path("your_gene_list.txt")   # the genes the model selected
OUT_DIR = Path("interpretation_output")
TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_SLEEP = 5.0

SYSTEM_PROMPT = """You are a curator of molecular biology literature working on
lung adenocarcinoma. Answer with strict JSON only, no prose outside the JSON
object, and never invent a gene symbol, a pathway name or a mechanism."""

# Step 1: what the gene is and does
STEP1_PROMPT = """Annotate one human gene: its intrinsic function, the biological
processes it takes part in, and a short description of its role. Report only
what the published literature supports, and call a function uncertain when the
evidence is thin.

Reply with strict JSON:
{"gene": "{GENE}", "core_functions": ["..."], "biological_processes": ["..."],
 "functional_description": "..."}
"""

# Step 2: the pathways the gene drives
STEP2_PROMPT = """Place one human gene in the pathways it centrally takes part in: a
core enzyme, a receptor or channel, a signalling node, a transcription factor,
or a component the pathway is built on. A pathway that merely mentions the
gene does not count. Use the standard pathway names, and let the gene's
biology decide how many it gets. Check every candidate against the published
literature and keep only the documented ones.

The gene's own annotation, for context:
{ANNOTATION}

Reply with strict JSON:
{"gene": "{GENE}", "pathway": ["..."]}
"""

# Step 3: the pathways read back as broad themes
STEP3_PROMPT = """Read the pathways below and the genes behind them as broad biological
themes. A theme is what several pathways jointly drive; a pathway may sit in
several themes, and every pathway must end up in at least one. The number of
themes and their breadth follow from the biology, so prefer consolidation
over fragmentation. Name a theme after the biological process rather than a
gene, a pathway or a generic word, and keep it short and free of commas: it
is a table label. Check the grouping against the published literature and
correct it.

Reply with strict JSON:
{"macroscopic_modules": [{"module": "...", "pathway": ["..."]}]}
"""


def ask(client, prompt, required=(), expect_gene=None):
    """One chat completion, retried until the reply is a JSON object that
    carries the required keys and names the requested gene."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL, temperature=TEMPERATURE, messages=messages,
                response_format={"type": "json_object"})
            text = re.sub(r"<think>.*?</think>", "",
                          response.choices[0].message.content or "", flags=re.DOTALL)
            start, end = text.find("{"), text.rfind("}")
            obj = json.loads(text[start:end + 1]) if 0 <= start < end else {}
            if all(k in obj for k in required) and (
                    expect_gene is None
                    or str(obj.get("gene", "")).strip().upper() == expect_gene.upper()):
                return obj
            error = f"the reply does not match the schema: {sorted(obj)}"
        except Exception as exc:
            error = repr(exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"the LLM call failed after {MAX_RETRIES} attempts: {error}")


def cached_ask(client, cache, prompt, required=(), expect_gene=None):
    """The answer for one prompt: read the cache, else ask the LLM and save it."""
    if cache.exists():
        try:
            obj = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and all(k in obj for k in required):
                return obj
        except json.JSONDecodeError:
            pass
    obj = ask(client, prompt, required, expect_gene)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj


def read_genes(path):
    """Gene symbols, one per line; blank lines and # comments are skipped."""
    symbols = [line.strip().upper()
               for line in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(s for s in symbols if s and not s.startswith("#")))


def main(gene_list=GENE_LIST_FILE, out_dir=OUT_DIR):
    """The selected genes -> function -> pathway -> theme -> table."""
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    genes = read_genes(gene_list)

    # step 1: what each gene does
    annotations = {}
    for gene in genes:
        annotations[gene] = cached_ask(
            client, out_dir / "genes" / f"{gene}.json",
            STEP1_PROMPT.replace("{GENE}", gene),
            required=("core_functions", "biological_processes",
                      "functional_description"), expect_gene=gene)

    # step 2: the pathways each gene drives
    gene_pathway = {}
    for gene, annotation in annotations.items():
        prompt = (STEP2_PROMPT.replace("{GENE}", gene).replace(
            "{ANNOTATION}", json.dumps(annotation, ensure_ascii=False)))
        obj = cached_ask(client, out_dir / "pathways" / f"{gene}.json", prompt,
                         required=("pathway",), expect_gene=gene)
        gene_pathway[gene] = list(obj.get("pathway", []))

    # step 3: the pathways read back as broad themes
    gene_view = [{"gene": g, "annotation": annotations[g],
                  "pathway": gene_pathway[g]} for g in annotations]
    pathway_view = [{"pathway": p, "member_genes": sorted(
        g for g, pws in gene_pathway.items() if p in pws)}
        for p in dict.fromkeys(p for pws in gene_pathway.values() for p in pws)]
    prompt = (STEP3_PROMPT
              .replace("{GENES}", json.dumps(gene_view, ensure_ascii=False))
              .replace("{PATHWAYS}", json.dumps(pathway_view, ensure_ascii=False)))
    obj = ask(client, prompt, required=("macroscopic_modules",))
    themes = [t for t in obj.get("macroscopic_modules", []) if isinstance(t, dict)]

    # the table: one row per gene and pathway, with the theme it falls in
    pathway_themes = {}
    for theme in themes:
        for p in theme.get("pathway", []):
            pathway_themes.setdefault(p, []).append(theme.get("module", ""))
    rows = [(g, p, t) for g, pws in gene_pathway.items() for p in pws
            for t in (pathway_themes.get(p) or [""])]
    out_csv = out_dir / "gene_module_pathway.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "pathway", "macroscopic_module"])
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
