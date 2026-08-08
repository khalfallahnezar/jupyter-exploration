# NewsBot Intelligence System 2.0 — Executive Summary
Nezar Khalfallah | ITAI 2373 — Natural Language Processing | Final Project

## Overview

NewsBot Intelligence System 2.0 converts raw, unstructured news text into structured
intelligence — category, sentiment, key entities, discovered themes, and a plain-language
summary — and adds three new capabilities beyond the midterm system: automatic topic discovery,
multilingual understanding, and a natural-language query interface. It is built and validated
against 1,000 real BBC News articles across five categories (business, entertainment, politics,
sport, tech).

## Value Proposition

Organizations that monitor news at volume — media-monitoring firms, market-intelligence teams,
PR/communications departments, and research groups — currently rely on manual triage: analysts
read articles one at a time to tag topic, sentiment, and mentioned entities. NewsBot 2.0
automates that first pass:

- **Routing:** classifies each article into the correct desk with 97% held-out accuracy, and
  flags the roughly 3% of low-confidence cases for a human reviewer instead of silently
  mis-filing them — a routing system an analyst can actually trust.
- **Monitoring:** extracts people, organizations, and places, and maps which of them tend to be
  mentioned together — supporting an "alert me when this company's coverage turns negative"
  workflow directly.
- **Summarizing:** compresses articles to roughly 30% of their original length while preserving
  the sentences a graph-ranking algorithm identifies as most representative, so an analyst can
  scan a day's coverage far faster than reading every article in full.
- **Discovering themes:** surfaces recurring topics automatically (via LDA), including sub-themes
  within a single category that a flat classifier can't see on its own — useful for spotting an
  emerging story before it's obvious from headlines alone.
- **Multilingual reach:** detects the language of a source article and translates it to English
  before analysis, so the same pipeline can ingest non-English coverage without a separate
  system.
- **Self-service querying:** a plain-English query interface ("show me positive tech news",
  "how many politics articles are there") lets a non-technical analyst explore the corpus
  without writing code or learning a query language.

## Measured Results

| Capability | Result |
|---|---|
| Classification accuracy (held-out) | 97.0% (macro-F1 0.97) |
| Distinct topics discovered | 8, with low overlap between topics |
| Average summary compression | ~3.4x shorter than the source article |
| Internal + unit test coverage | 25/25 tests passing |

These are measured, not estimated — every figure comes from an actual run of the notebook
against the full 1,000-article sample used in this submission.

## Return on Investment (Illustrative)

At 97% routing accuracy, the system can safely auto-file the large majority of incoming articles
and escalate only the small low-confidence remainder for manual review — directly reducing the
analyst-hours spent on first-pass triage. For a team currently reading, say, 200 articles a day
by hand, automating classification, entity extraction, and summarization removes the majority of
that reading time, leaving analysts to spend their time on judgment calls (interpreting a trend,
deciding what's newsworthy) rather than mechanical tagging.

## Competitive Context

Commercial media-monitoring platforms (e.g. Meltwater, Cision) offer comparable classification
and sentiment features, typically as a paid SaaS product with per-seat licensing. NewsBot 2.0
demonstrates the same core capability set — classification, sentiment, entity extraction, topic
discovery, summarization, multilingual support, natural-language querying — built on open-source
components with no licensing cost, at a scale (thousands of articles) appropriate for a team or
department rather than an enterprise-wide deployment.

## Limitations to Consider Before Production Use

- Built and validated on English-language BBC News; performance on other news sources or
  writing styles has not been separately measured.
- The dataset used has no publish dates, so time-based trend analysis (topic and sentiment
  evolution) is demonstrated using document order as a stand-in — a production deployment with
  dated articles would need no code changes to get true time-series trends.
- Translation relies on a live external service; a fully offline deployment would need a
  self-hosted translation model as a fallback.

## Recommendation

NewsBot 2.0 is ready to pilot on a real, ongoing news stream. The architecture (see Technical
Documentation) is modular enough that swapping in dated articles, a larger corpus, or a
transformer-based embedding model requires no redesign — only configuration changes.
