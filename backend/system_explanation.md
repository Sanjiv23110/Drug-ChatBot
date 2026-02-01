# Medical RAG System: How It Works

This document explains the inner workings of the Solomind Drug Monograph QA system. It uses a **SQL-First, Structure-Aware** approach rather than standard vector similarity search.

---

## 🏗️ 1. Ingestion Pipeline (PDF → Database)
*How we turn raw PDFs into structured data.*

### A. Layout-Aware Parsing (`DoclingParser`)
Instead of treating the PDF as a blob of text, we use **Docling** to parse the visual layout.
- **Headers vs. Text**: We distinguish between section headers ("DOSAGE") and content.
- **Images**: We extract images and use **GPT-4o Vision** to detect chemical structures versus decorative logos.
- **Tables**: Tables are preserved as structured text, not flattened.

### B. Dynamic Sectioning (The "Secret Sauce")
Most RAG systems chunk by character count (e.g., 500 chars). We chunk by **Semantic Sections**.
- We detect headers like "CLINICAL PHARMACOLOGY" or "WARNINGS".
- We map these to normalized categories (e.g., `clinical_pharmacology`).
- **Benefit**: We can query *specifically* for "Dosage" sections, guaranteeing relevance.

### C. FactSpan Layer (New Verbatim Storage)
We recently added a "FactSpan" layer that dissects sections into individual sentences.
- **Verbatim Storage**: Every sentence is stored exactly as written.
- **Metadata**: Each sentence is tagged with its source section and headers.
- **Purpose**: Allows retrieving precise facts (e.g., "half-life is 2 hours") without hallucination.

---

## 🧠 2. Query Understanding (The Brain)
*How the system understands what you want.*

When you ask a question, the **Intent Classifier** analyzes it before searching:
1.  **Drug Extraction**: Identifies the drug (e.g., "Nizatidine", "AXID").
2.  **Section Targeting**: Predicts which section contains the answer.
    *   *Query*: "How do I take it?" →  Target: `dosage`
    *   *Query*: "Side effects?" → Target: `adverse_reactions`
3.  **Attribute Detection**: Detects specific properties (e.g., "half-life").

---

## 🔍 3. Retrieval Engine (Finding the Needle)
*How we find the right data.*

We use a **Multi-Path Strategy** prioritized by accuracy:

### Path A: SQL Exact Match (Highest Priority)
If we know the Drug and the Section:
- We run a SQL query: `SELECT * FROM sections WHERE drug='axid' AND section='dosage'`
- **Accuracy**: 100%. We get the *exact* section titled "Dosage". No vector guessing.

### Path A++: FactSpan Invariant (The New Fix)
If the query asks for a specific fact (e.g., "half-life"):
- We scan the **FactSpan** table for sentences tagged as 'FACT' or 'CONDITIONAL' that match.
- **Rule**: If such sentences exist, we **FORCE RETURN** them verbatim.
- **Result**: Even if the LLM thinks it's unimportant, we show it.

### Path B: Image Lookup
If you ask "Show me the structure":
- We look up images classified as `chemical_structure` for that drug.

### Path C: Vector Fallback (Last Resort)
Only if SQL fails do we use vector embeddings (semantic similarity) to find related text.

---

## ✍️ 4. Generation (The Answer)
*How we write the response.*

### The Prompts
We use a specialized prompt that enforces:
1.  **Group & Label**: "Don't write an essay. Group facts by their source."
2.  **Verbatim**: "Copy medical terms exactly. Do not paraphrase."
3.  **Existence**: "If you see it, show it. Don't decide if it's 'enough'."

### Validation Gates
The **Answer Generator** runs checks on the output:
1.  **Citation Check**: Every claim must have a `[Source X]` citation. If a citation points to a file/page not in the context, we reject the answer.
2.  **Hallucination Check**: We check for phrases like "in my opinion" (though we recently relaxed this).

---

## 🔄 Summary of Flow
1.  **User**: "What is the half-life of Axid?"
2.  **Classifier**: Target Drug=`Axid`, Attribute=`half_life`, Section=`pharmacokinetics`.
3.  **Retriever**:
    *   Checks `FactSpans` for "half-life" in `pharmacokinetics` sections.
    *   Found: "The elimination half-life is 1.6 hours."
    *   **Invariant Trigger**: Forces this sentence to be returned.
4.  **Generator**: Receives the sentence. Formats it with a header.
5.  **Result**:
    > **## From Pharmacokinetics**
    > *   "The elimination half-life is 1.6 hours." [Source: Axid Monograph]
