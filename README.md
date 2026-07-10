# AI Email Suggested-Response System

An end-to-end system that generates suggested replies to incoming professional emails and rigorously evaluates their quality using a multi-signal LLM-as-a-judge metric.

## 🚀 Quickstart

1. Set up your environment:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY to .env (get one free at console.groq.com/keys)
   ```

2. Run the full pipeline (generates replies, evaluates them, runs meta-eval, and creates reports):
   ```bash
   python -m src.run_all
   ```

3. View the results:
   ```bash
   open reports/report.html
   ```

## 🏗️ Architecture & Approach

This system consists of three main components, all built entirely on the Python 3.14 standard library to ensure 100% reproducibility with zero ML dependency hell. The only external call is to the Groq API via `urllib`.

### 1. Dataset Generation (`data/generate_dataset.py`)
I hand-authored 15 high-quality "seed" email pairs spanning 5 categories (customer support, scheduling, sales, internal ops, billing). A script then uses these seeds as few-shot examples for an LLM to synthesise a realistic corpus of 135 unique emails. 
- **Honesty**: This is a curated synthetic dataset. It was chosen to avoid PII issues while guaranteeing high-quality, professional formatting that matches real-world scenarios.
- The dataset is deterministically split into a **retrieval pool** (110 emails) and a **held-out test set** (25 emails) so the generator never sees the gold reply for the email it is answering.

### 2. Generator: RAG + Few-Shot (`src/generator.py`)
To generate a reply to a new incoming email, the system uses Retrieval-Augmented Generation (RAG).
- **Retrieval**: A pure-Python TF-IDF index (`src/retriever.py`) finds the $k=3$ most similar past emails using cosine similarity.
- **Generation**: These retrieved pairs are injected into the prompt as few-shot exemplars, teaching the LLM the "house style" and level of detail expected.
- **Trade-offs**: I chose RAG over fine-tuning because it is cheap, transparent, adapts instantly to new data without retraining, and explicitly grounds the model. I chose TF-IDF over dense embeddings to keep the project zero-dependency; upgrading to sentence-transformers is a trivial drop-in replacement if dependencies are allowed.

### 3. Evaluation System (`src/evaluator.py`)
**This is the core of the challenge.** Evaluating a suggested reply against a "gold" reference is fundamentally flawed because there are many valid ways to write an email. Lexical match (ROUGE) punishes perfectly good stylistic variations.

My composite metric combines four signals, heavily weighted toward reference-free evaluation:

1. **LLM Rubric Judge (50%)**: A *different* model from the generator scores the reply from 1-5 on Relevance, Completeness, Correctness, Tone, and Actionability. It must provide a written rationale for each score. (Using a different model reduces self-preference bias).
2. **Key-Point Coverage (25%)**: The judge extracts the "must-address" questions from the incoming email, and a checker calculates the fraction of those points addressed in the reply. This provides an auditable completeness signal.
3. **TF-IDF Cosine (15%)**: Lexical overlap with the gold reply, weighted by corpus frequency.
4. **ROUGE-L F1 (10%)**: Longest common subsequence with the gold reply.

## 🔬 Validating the Metric (Meta-Evaluation)

To prove the accuracy metric isn't just producing arbitrary numbers, the system runs a `meta_eval.py` suite:

1. **Discrimination (Adversarial Testing)**: We generate corrupted versions of gold replies (off-topic, empty, truncated, wrong facts, rude tone). The metric correctly ranks the gold reply above every corruption **100.0%** of the time.
2. **Human-Anchor Correlation**: We hand-rated 20 candidate replies on a 1-5 scale. The composite metric achieves a **Pearson correlation of 0.905** with human judgment, significantly outperforming raw lexical metrics.
3. **Judge Reliability**: The LLM judge is run multiple times on the same emails. The mean standard deviation across runs is **0.018**, proving the scoring is highly consistent and not overly sensitive to prompt noise.

## 🛠️ AI Tools Used

This project was built with the assistance of **Gemini 3.1 Pro (High)** for architectural brainstorming, boilerplate generation, and debugging. The core dataset synthesis and generation pipeline utilizes **Meta Llama 3.3 70B**, while the LLM-as-a-judge evaluation utilizes **OpenAI GPT-OSS 120B** (both hosted on Groq).
