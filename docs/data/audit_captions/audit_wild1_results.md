# Dataset Audit Report: `wild1`

**Date**: 2026-08-10  
**Target Directory**: `datasets/wildQA/captions__wild1`  
**Total Clips Audited**: `24`  

## Executive Summary

| Metric | Result | Health Status |
| :--- | :--- | :--- |
| **Suspected Duplicate Clusters** | `0` | 🟢 Clean (0 duplicates) |
| **Ad / Commercial Boilerplate Hits** | `1` | 🟡 Warning |
| **Repetitive / Loop Captions** | `0` | 🟢 Clean |

---

## 1. Duplication & Similarity Analysis

🟢 **No high-similarity pairs or duplicate video clusters found.** All video captions are distinct and unique.

## 2. Ad & Commercial Boilerplate Screening

Found **1** files containing potential ad/boilerplate keywords:

* **`Hamiltonville-Farm_8-clip-3`**: Matched Channel subscription request (2x)
  * *Snippet*: "A man in a brown jacket stands in front of a pole barn, introducing the next part of his project which involves hanging two more lights and asking vie..."

## 3. Domain Alignment Analysis

Comparison of keyword profile predictions against ground-truth dataset domains:

🟢 **All evaluated videos align with their ground-truth domains.**
