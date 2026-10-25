#!/usr/bin/env python
"""Gene prior weights (alpha) from LLM-mined literature evidence."""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from Bio import Entrez
from Bio.Medline import parse as MedlineParse
from openai import OpenAI

# ---- Credentials: keep them in the environment, never in the file ----
PUBMED_API_KEY = os.environ.get("NCBI_API_KEY", "YOUR_NCBI_API_KEY")
ENTREZ_EMAIL = os.environ.get("NCBI_EMAIL", "YOUR_EMAIL")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "YOUR_LLM_BASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "YOUR_LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "YOUR_LLM_MODEL")

# ---- Run settings ----
N_RUN = 100            # the repeated mining runs, k = 1 ... 100
MIN_APPEARANCES = 50   # a gene must recur in at least this many of the k runs
TAU = 0.05             # smallest alpha, i.e. the weight of the best supported gene
YEARS_BACK = 4         # how far back the literature search reaches

Entrez.email = ENTREZ_EMAIL
if PUBMED_API_KEY:
    Entrez.api_key = PUBMED_API_KEY

# One query over three axes: disease, outcome and a generic gene concept, so
# the corpus is broad and not biased to any particular gene.
QUERY_PROMPT = """Design ONE PubMed query for LUAD survival research.
Three axes, each ("phrase"[Title/Abstract]), joined by AND:
  DISEASE: lung adenocarcinoma / LUAD
  OUTCOME: overall survival / prognosis / mortality
  MOLECULAR ENTITY: a generic "gene" concept, not a specific gene
No specific gene symbol, no OR between axes, no date filter.
Reply with strict JSON: {"query": "<the full query>"}
"""

# Per paper: does a specific gene link to LUAD survival?
SCREEN_PROMPT = """You curate LUAD literature. Decide whether the paper reports a
SPECIFIC GENE associated with LUAD overall survival / prognosis.
INCLUDE cohorts, experiments or bioinformatics with a survival endpoint; EXCLUDE
recurrence only, therapy response only, pan-cancer without LUAD, mechanism only.
Use the HGNC gene symbol, not a protein alias or an old name; never invent a
gene, and list nothing if unsure.
Reply with strict JSON: {"genes": ["<HGNC symbol>"]}
"""


def ask(client, system_prompt, user_prompt, temperature):
    """One chat completion; the JSON reply is parsed."""
    response = client.chat.completions.create(
        model=LLM_MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"})
    text = (response.choices[0].message.content or "").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start >= 0 < end else {}
    except json.JSONDecodeError:
        return {}


def fetch_papers(query, retmax=10000, top_k_ratio=0.5, years_back=YEARS_BACK):
    """Search PubMed; returns the papers as {pmid, title, abstract}.

    The window ends today and reaches back years_back years.
    """
    date_max = datetime.now()
    date_min = date_max - timedelta(days=365 * years_back)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="relevance",
                            datetype="pdat",
                            mindate=date_min.strftime("%Y/%m/%d"),
                            maxdate=date_max.strftime("%Y/%m/%d"))
    record = Entrez.read(handle)
    handle.close()
    total = int(record.get("Count", 0))
    pmids = list(record.get("IdList", []))[:min(int(total * top_k_ratio), retmax)]
    if not pmids:
        return []
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
    papers = [{"pmid": str(r.get("PMID", "")), "title": str(r.get("TI", "")),
               "abstract": str(r.get("AB", ""))} for r in MedlineParse(handle)]
    handle.close()
    return papers


def screen_run(client, papers):
    """Screen the papers and score the genes: the share of evidence-bearing
    papers that mention a gene is its evidence rate in this run."""
    hits, valid = Counter(), 0
    for paper in papers:
        obj = ask(client, SCREEN_PROMPT, json.dumps(paper, ensure_ascii=False), 0.3)
        genes = {str(g).strip().upper() for g in obj.get("genes", []) if str(g).strip()}
        if genes:
            valid += 1
            hits.update(genes)
    return {g: c / valid for g, c in hits.items()} if valid else {}


def calibrate_runs(run_scores, n_run=N_RUN, min_appearances=MIN_APPEARANCES):
    """Merge the k = 1 ... n_run runs into one prior score R per gene.

    A gene is kept only when it recurs in at least min_appearances of the k
    runs. Its score corrects the evidence rate r for the noise of the
    appearance count: SE is the standard deviation of the count nu, i.e. of
    the number of k runs the gene showed up in, RSE = SE / nu, R = r (1-RSE).
    """
    per_gene = defaultdict(list)
    for scores in run_scores:          # one entry per k
        for gene in set(scores):
            per_gene[gene.upper()].append(float(scores[gene]))
    calibrated = {}
    for gene, values in per_gene.items():
        nu = len(values)               # how many of the k runs
        if nu < min_appearances:
            continue
        r = float(np.mean(values))
        se = float(np.sqrt(nu * (n_run - nu) / n_run))
        calibrated[gene] = r * (1.0 - se / nu)
    return calibrated


def alpha_tensor(gene_names, prior_scores, tau=TAU):
    """Prior weights aligned to gene_names, matched case-insensitively.

    alpha = 1 - (lg R - min(lg R)) / (max(lg R) - min(lg R)) * (1 - tau): the
    best supported gene gets tau, the weakest gets 1, a gene without a score
    keeps 1. The tensor is fixed: alpha never takes part in the gradients.
    """
    upper = {str(g).upper(): float(v) for g, v in prior_scores.items()}
    values = np.array([upper.get(str(g).upper(), 0.0) for g in gene_names])
    alpha = np.ones(len(values))
    scored = values > 0
    if scored.any():
        v = np.clip(values[scored], 1e-6, None)
        v = np.where(v == 1.0, v + np.exp(-1.0), v)   # a score of 1 sits on the log floor
        log_v = np.log10(v)
        lo, hi = log_v.min(), log_v.max()
        norm = (log_v - lo) / (hi - lo) if hi > lo else np.zeros_like(log_v)
        alpha[scored] = 1.0 - norm * (1.0 - tau)
    return torch.tensor(alpha, dtype=torch.float32)


def main(gene_names, prior_table=None):
    """Build the prior weights alpha for the given genes.

    Without a prior table the literature is mined from scratch, with one the
    scores of an earlier mining run are reused.
    """
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    if prior_table:
        df = pd.read_csv(prior_table)
        gene_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        score_col = next((c for c in df.columns if "relevance" in c.lower()), None)
        if gene_col is None or score_col is None:
            raise KeyError(f"no gene symbol / relevance score column in {prior_table}")
        values = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        prior_scores = {str(g).upper(): float(s) for g, s in zip(df[gene_col], values)}
    else:
        run_scores = []
        for k in range(1, N_RUN + 1):
            # a fresh query in every run k, so the runs explore different literature
            obj = ask(client, QUERY_PROMPT, "Generate exactly 1 PubMed query.", 0.9)
            papers = fetch_papers(str(obj.get("query", "")).strip())
            run_scores.append(screen_run(client, papers))
        # keep the genes that recur and correct their evidence across the runs
        prior_scores = calibrate_runs(run_scores)
    # R -> alpha, aligned to the gene order of the expression matrix
    return alpha_tensor(gene_names, prior_scores)


if __name__ == "__main__":
    # the gene symbols of the expression matrix, in column order
    alpha = main(your_gene_names)
