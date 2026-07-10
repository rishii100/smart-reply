"""Report generation: JSON summaries + self-contained HTML report.

The HTML report is fully self-contained (inline CSS, no external calls) — you
can open it in any browser.  It has three panels:

1. **Overall** — system composite score, per-signal means, per-category
   breakdown, score distribution.
2. **Per-response** — sortable table with incoming email, generated reply, every
   metric, judge rationale, composite score, pass/fail.
3. **Meta-evaluation** — discrimination results, human correlation, judge
   reliability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_reports(
    per_response: list[dict[str, Any]],
    summary: dict[str, Any],
    meta_eval: dict[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Write all report files and return their paths."""
    out = output_dir or config.REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["per_response"] = out / "per_response.json"
    save_json(per_response, paths["per_response"])

    paths["summary"] = out / "summary.json"
    save_json(summary, paths["summary"])

    if meta_eval:
        paths["meta_eval"] = out / "meta_eval.json"
        save_json(meta_eval, paths["meta_eval"])

    paths["html"] = out / "report.html"
    html = _build_html(per_response, summary, meta_eval)
    paths["html"].write_text(html, encoding="utf-8")

    return paths


# --------------------------------------------------------------------------- #
# HTML builder
# --------------------------------------------------------------------------- #
def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def _score_color(score: float) -> str:
    """Return a CSS color based on score (0–1)."""
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.65:
        return "#eab308"
    if score >= 0.4:
        return "#f97316"
    return "#ef4444"


def _badge(score: float, label: str = "") -> str:
    color = _score_color(score)
    text = f"{label} {score:.3f}" if label else f"{score:.3f}"
    return f'<span class="badge" style="background:{color}">{text.strip()}</span>'


def _build_html(
    per_response: list[dict[str, Any]],
    summary: dict[str, Any],
    meta_eval: dict[str, Any] | None,
) -> str:
    parts = [_HTML_HEAD]

    # --- Overall panel ---
    parts.append('<div class="panel">')
    parts.append('<h2>📊 Overall System Scores</h2>')
    oc = summary.get("overall_composite", 0)
    parts.append(f'<div class="big-score" style="color:{_score_color(oc)}">{oc:.4f}</div>')
    parts.append(f'<p class="sub">Composite score (mean across {summary.get("total_evaluated", 0)} test emails)</p>')
    
    # Score distribution
    dist = summary.get("score_distribution", {})
    parts.append('<div class="stats-grid">')
    for label, key in [("Min", "min"), ("Median", "median"), ("Max", "max")]:
        v = dist.get(key, 0)
        parts.append(f'<div class="stat-card"><div class="stat-value">{v:.4f}</div><div class="stat-label">{label}</div></div>')
    pr = summary.get("pass_rate", 0)
    parts.append(f'<div class="stat-card"><div class="stat-value">{pr:.0%}</div><div class="stat-label">Pass rate (≥{summary.get("pass_threshold", 0.65)})</div></div>')
    parts.append('</div>')

    # Signal means
    parts.append('<h3>Signal Means</h3>')
    parts.append('<div class="stats-grid">')
    for k, v in summary.get("signal_means", {}).items():
        w = config.COMPOSITE_WEIGHTS.get(k, 0)
        parts.append(f'<div class="stat-card"><div class="stat-value">{_badge(v)}</div>'
                      f'<div class="stat-label">{k} (w={w})</div></div>')
    parts.append('</div>')

    # Per-category
    parts.append('<h3>Per-Category Breakdown</h3>')
    parts.append('<table><tr><th>Category</th><th>Mean Composite</th></tr>')
    for cat, score in summary.get("category_scores", {}).items():
        parts.append(f'<tr><td>{_esc(cat)}</td><td>{_badge(score)}</td></tr>')
    parts.append('</table>')

    # Criterion means
    parts.append('<h3>Judge Criteria Means (1–5)</h3>')
    parts.append('<table><tr><th>Criterion</th><th>Mean Score</th></tr>')
    for crit, val in summary.get("criterion_means", {}).items():
        parts.append(f'<tr><td>{_esc(crit)}</td><td>{val:.2f}</td></tr>')
    parts.append('</table>')
    parts.append('</div>')

    # --- Per-response panel ---
    parts.append('<div class="panel">')
    parts.append(f'<h2>📝 Per-Response Results ({len(per_response)} emails)</h2>')
    for r in per_response:
        score = r["composite_score"]
        pass_fail = "✅ PASS" if r["pass"] else "❌ FAIL"
        parts.append(f'<div class="response-card">')
        parts.append(f'<div class="response-header">')
        parts.append(f'<span class="email-id">{_esc(r["email_id"])}</span>')
        parts.append(f'<span class="category-tag">{_esc(r.get("category", ""))}</span>')
        parts.append(f'{_badge(score, "composite")} <span class="pass-fail">{pass_fail}</span>')
        parts.append('</div>')

        parts.append(f'<details><summary>📧 Subject: {_esc(r.get("subject", "(no subject)"))}</summary>')
        parts.append(f'<div class="email-section"><strong>Incoming:</strong><div class="email-body">{_esc(r["incoming"])}</div></div>')
        parts.append(f'<div class="email-section"><strong>Gold reply:</strong><div class="email-body">{_esc(r.get("gold_reply", ""))}</div></div>')
        parts.append(f'<div class="email-section"><strong>Generated reply:</strong><div class="email-body generated">{_esc(r["generated_reply"])}</div></div>')

        # Metrics
        parts.append('<div class="metrics-grid">')
        for k, v in r.get("composite_signals", {}).items():
            parts.append(f'<div>{_badge(v, k)}</div>')
        parts.append('</div>')

        # Reference metrics
        ref = r.get("reference_metrics", {})
        if ref:
            parts.append('<div class="metrics-grid">')
            for k, v in ref.items():
                parts.append(f'<div class="ref-metric">{k}: {v:.4f}</div>')
            parts.append('</div>')

        # Judge rationale
        judge = r.get("judge", {})
        parts.append('<div class="judge-detail"><strong>Judge rationale:</strong><ul>')
        for crit in config.JUDGE_CRITERIA:
            entry = judge.get(crit, {})
            if isinstance(entry, dict):
                s = entry.get("score", "?")
                rat = entry.get("rationale", "")
                parts.append(f'<li><strong>{crit}</strong> ({s}/5): {_esc(rat)}</li>')
        parts.append('</ul></div>')

        # Key-point coverage
        kpc = r.get("key_point_coverage", {})
        if kpc.get("coverage"):
            parts.append('<div class="judge-detail"><strong>Key-point coverage:</strong><ul>')
            for c in kpc["coverage"]:
                icon = "✅" if c.get("covered") else "❌"
                parts.append(f'<li>{icon} {_esc(c.get("point", ""))}</li>')
            parts.append('</ul></div>')

        parts.append('</details>')
        parts.append('</div>')
    parts.append('</div>')

    # --- Meta-evaluation panel ---
    if meta_eval:
        parts.append('<div class="panel">')
        parts.append('<h2>🔬 Meta-Evaluation: Is the Metric Valid?</h2>')

        # Discrimination
        disc = meta_eval.get("discrimination", {})
        if disc:
            parts.append('<h3>1. Discrimination (Adversarial Test)</h3>')
            pa = disc.get("pairwise_accuracy", 0)
            parts.append(f'<p>Pairwise accuracy (gold > corruption): <strong>{pa:.1%}</strong> '
                          f'({disc.get("pairwise_correct", 0)}/{disc.get("pairwise_total", 0)})</p>')
            parts.append('<table><tr><th>Condition</th><th>Mean Score</th></tr>')
            for cond, score in disc.get("condition_means", {}).items():
                parts.append(f'<tr><td>{_esc(cond)}</td><td>{_badge(score)}</td></tr>')
            parts.append('</table>')

        # Human correlation
        human = meta_eval.get("human_correlation", {})
        if human.get("correlations"):
            parts.append('<h3>2. Human-Anchor Correlation</h3>')
            parts.append(f'<p>Based on {human.get("n_samples", 0)} hand-rated samples:</p>')
            parts.append('<table><tr><th>Metric</th><th>Pearson</th><th>Spearman</th></tr>')
            for m, c in human["correlations"].items():
                parts.append(f'<tr><td>{_esc(m)}</td><td>{c["pearson"]:.4f}</td><td>{c["spearman"]:.4f}</td></tr>')
            parts.append('</table>')

        # Reliability
        rel = meta_eval.get("judge_reliability", {})
        if rel:
            parts.append('<h3>3. Judge Reliability</h3>')
            parts.append(f'<p>Mean score std across {rel.get("k_runs", 0)} repeated runs: '
                          f'<strong>{rel.get("mean_std", 0):.4f}</strong></p>')
            parts.append(f'<p><em>{_esc(rel.get("interpretation", ""))}</em></p>')
            if rel.get("items"):
                parts.append('<table><tr><th>Email</th><th>Scores</th><th>Mean</th><th>Std</th></tr>')
                for item in rel["items"]:
                    scores_str = ", ".join(f"{s:.3f}" for s in item["scores"])
                    parts.append(f'<tr><td>{_esc(item["email_id"])}</td>'
                                  f'<td>{scores_str}</td>'
                                  f'<td>{item["mean"]:.4f}</td>'
                                  f'<td>{item["std"]:.4f}</td></tr>')
                parts.append('</table>')

        parts.append('</div>')

    parts.append(_HTML_FOOT)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML template pieces
# --------------------------------------------------------------------------- #
_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Email Reply Evaluation Report</title>
<style>
:root {
  --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
  --text: #e2e8f0; --text2: #94a3b8; --accent: #38bdf8;
  --green: #22c55e; --yellow: #eab308; --orange: #f97316; --red: #ef4444;
  --radius: 12px; --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text); font-family: var(--font);
  line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto;
}
h1 { font-size: 2rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, var(--accent), #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.4rem; margin-bottom: 1rem; color: var(--accent); }
h3 { font-size: 1.1rem; margin: 1.5rem 0 0.75rem; color: var(--text); }
p { margin-bottom: 0.75rem; }
.panel {
  background: var(--surface); border-radius: var(--radius); padding: 1.5rem;
  margin-bottom: 1.5rem; border: 1px solid var(--surface2);
}
.big-score { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; }
.sub { color: var(--text2); font-size: 0.9rem; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 1rem; margin: 1rem 0; }
.stat-card { background: var(--surface2); border-radius: 8px; padding: 1rem; text-align: center; }
.stat-value { font-size: 1.3rem; font-weight: 600; }
.stat-label { font-size: 0.8rem; color: var(--text2); margin-top: 0.25rem; }
table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; }
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--surface2); }
th { color: var(--accent); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
  font-weight: 600; color: #fff; }
.response-card { background: var(--surface2); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;
  transition: transform 0.15s; }
.response-card:hover { transform: translateY(-1px); }
.response-header { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
.email-id { font-weight: 600; color: var(--accent); }
.category-tag { background: #3b0764; color: #c084fc; padding: 2px 8px; border-radius: 999px;
  font-size: 0.75rem; }
.pass-fail { font-size: 0.85rem; }
details { margin-top: 0.75rem; }
summary { cursor: pointer; color: var(--text2); font-size: 0.9rem; }
summary:hover { color: var(--text); }
.email-section { margin: 0.75rem 0; }
.email-body { background: var(--bg); border-radius: 6px; padding: 0.75rem; margin-top: 0.25rem;
  font-size: 0.85rem; color: var(--text2); max-height: 200px; overflow-y: auto; }
.email-body.generated { border-left: 3px solid var(--accent); }
.metrics-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0; }
.ref-metric { background: var(--bg); padding: 2px 8px; border-radius: 6px; font-size: 0.8rem;
  color: var(--text2); }
.judge-detail { margin: 0.75rem 0; font-size: 0.85rem; }
.judge-detail ul { margin-left: 1.25rem; margin-top: 0.25rem; }
.judge-detail li { margin-bottom: 0.25rem; color: var(--text2); }
</style>
</head>
<body>
<h1>🤖 AI Email Reply — Evaluation Report</h1>
<p class="sub">Generated by the email-reply evaluation harness</p>
"""

_HTML_FOOT = """\
<div class="panel" style="text-align:center; color: var(--text2); font-size: 0.8rem;">
  <p>Report generated by <strong>email-reply</strong> evaluation harness.
  Self-contained — no external dependencies.</p>
</div>
</body>
</html>
"""
