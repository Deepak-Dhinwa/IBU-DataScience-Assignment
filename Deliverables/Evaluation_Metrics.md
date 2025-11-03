# Evaluation Metrics

Three metrics to measure agent output quality: accuracy, conciseness, and pattern extraction quality.

---

## Metric 1: Context Grounding Score

**Why**: Detects hallucinations. Ensures outputs are supported by retrieved context, not fabricated.

**Calculation**:
```
Score = (Supported Claims / Total Claims) × 100
```
- **Supported Claims**: Claims verifiable in retrieved context
- **Total Claims**: All factual statements in output
- **Target**: ≥ 85%

**How**: Check each claim against retrieved chunks (exact or semantic match). Count supported vs. unsupported.

---

## Metric 2: Conciseness Compliance Score

**Why**: Agents have word limits. Ensures outputs stay concise and don't exceed limits.

**Calculation**:
```
Score = 100 - ((Actual Words - Target Words) / Target Words) × 100
```
- **Actual Words**: Word count of output
- **Target Words**: Agent-specific limit (Insight Writer: 125, Pattern Miner: 150, Global Stratified Miner: 250)
- **Target**: ≥ 90%

**How**: Count words, calculate deviation from target, apply penalty if over limit.

---

## Metric 3: Evidence‑Backed Pattern Rate

**Why**: We want patterns to be supported by the retrieved context without heavy scoring logic.

**Calculation**:
```
Score = (Patterns with Explicit Evidence / Total Patterns) × 100
```
- **Explicit Evidence**: A quoted span or exact number/percentage from context
- **Total Patterns**: Count of distinct patterns reported
- **Target**: ≥ 80%

**How**: For each pattern, check if it includes a quote or a numeric value found in the context. Count and divide.

---

## Baseline Targets

| Metric | Target | Minimum |
|--------|--------|---------|
| Context Grounding | ≥ 85% | ≥ 70% |
| Conciseness | ≥ 90% | ≥ 75% |
| Evidence‑Backed Pattern Rate | ≥ 80% | ≥ 65% |
