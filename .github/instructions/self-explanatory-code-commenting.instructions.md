---
description: "Guidelines for agent-authored code: mandatory robust docstrings + intent-level comments for control flow and multi-step blocks. Optimized for strongly-typed Python repos."
applyTo: "**/*.py"
---

# Intent-First Docstrings & Comments (Python, strongly typed)

## Core Principle

Write code that is readable, but **assume the maintainer may not know the intent** (common with agent-authored code).
Therefore:

- **Docstrings are mandatory** for classes and functions/methods (including private helpers).
- Inline comments are used to explain **intent, flow, and decision logic**—especially around iteration and branching.
- Avoid low-value “narrate the obvious” comments.

The goal is that a reader can understand the **purpose, usage, and flow** without reverse-engineering the implementation.

---

## 1) Mandatory class docstrings (robust)

Every class must have a docstring that covers, at minimum:

- **Purpose**: what the class represents or coordinates.
- **Responsibilities**: what it does and does *not* do (scope boundaries).
- **How it is intended to be used**: lifecycle, typical call pattern, collaboration with other objects.
- **High-level flow**: the main steps the class performs or orchestrates.
- **Key invariants / constraints**: expectations that must hold (e.g., sorted inputs, non-null IDs, caching semantics).
- **Important side effects**: I/O, persistence, network calls, mutation, concurrency considerations.
- **Attributes** (when non-obvious): what the stored fields mean and how they are populated.

Preferred structure (Google-style, typed) for consistency:

```python
class Example:
    """
    One-sentence summary of the class’s role.

    Purpose:
        What problem this class solves and where it fits in the system.

    Usage:
        Typical usage pattern (brief pseudo-code is acceptable).

    Flow:
        High-level steps this class performs, in order.

    Invariants / Constraints:
        Key assumptions that callers must respect.

    Side Effects:
        External calls, mutation, persistence, logging, etc.

    Attributes:
        field_name (Type): Meaning and lifecycle (if not obvious from the name).
    """
````

---

## 2) Mandatory function/method docstrings (robust, C#-like completeness)

Every function/method must have a docstring that includes:

* **Purpose** and behavior (what it accomplishes).
* **Parameters**: meaning, constraints, and how used (types included, even if hinted).
* **Returns**: meaning and shape of return value (or explicitly say `None` for procedures).
* **Raises**: key exceptions that are part of the contract (not every incidental exception, but contract-relevant ones).
* **Side effects**: if it mutates inputs, writes to disk/DB, emits events, etc.

Template:

```python
def method(self, x: XType, y: YType) -> ReturnType:
    """
    One-sentence summary of the method’s outcome.

    Purpose:
        What this method is responsible for and why it exists.

    Args:
        x (XType): Meaning, constraints, and how it influences behavior.
        y (YType): Meaning, constraints, and how it influences behavior.

    Returns:
        ReturnType: What is returned and how to interpret it.

    Raises:
        SomeError: When/why this is raised (contract-level).

    Side Effects:
        Describe mutations, I/O, persistence, logging, caching, etc.
    """
```

Notes:

* Keep docstrings accurate and **contract-oriented**. If behavior changes, docstrings must be updated.
* If the method is a thin wrapper around another call, say so explicitly (and why the wrapper exists).
* For `@property` accessors, docstrings should describe **what is exposed and the semantics** (cached vs computed, cost, invariants).

---

## 3) Loops and list comprehensions must be explained

Any `for` loop, `while` loop, or non-trivial list/dict/set comprehension must have an **intent comment** immediately above it.

Good:

```python
# Aggregate per-user totals while preserving first-seen ordering for stable output.
for row in rows:
    ...
```

For comprehensions, if the intent cannot be explained cleanly in one short comment, prefer expanding to an explicit loop.

Good:

```python
# Keep only canonicalized URLs that pass validation; used to drive dedupe and matching.
valid_urls = [u for u in urls if (u := canonicalize(u)) and is_valid(u)]
```

Better (when complex):

```python
# Canonicalize + validate URLs before feeding them into the dedupe pipeline.
valid_urls: list[str] = []
for raw in urls:
    canon = canonicalize(raw)
    if canon and is_valid(canon):
        valid_urls.append(canon)
```

---

## 4) Branching must explain decision logic (if/elif/else, match/case)

For any conditional branching beyond a trivial guard clause, add a comment that explains:

* **The decision criteria** (what distinguishes branches).
* **Why the ordering matters** (if it does).
* **The business/system rationale** (why this branching exists).

Good:

```python
# Branch by identifier quality:
# - Prefer stable external IDs when present (prevents duplicate entities).
# - Fall back to email match when IDs are missing.
# - Finally, create a new entity if no safe match exists.
if external_id:
    ...
elif email:
    ...
else:
    ...
```

For `match/case`, include a short “routing table” explanation:

```python
# Dispatch by event type:
# - "upsert" modifies or creates a record
# - "delete" tombstones but preserves audit trail
# - other events are ignored (forward compatibility)
match event.type:
    ...
```

---

## 5) “Forest through the trees”: comment multi-step blocks that achieve a larger objective

When a sequence of tactical lines collectively accomplishes a larger goal, precede the block with a **meta-what + why** comment.

Good:

```python
# Enrich the dataset with original publish metadata:
# inner-join on `source_id`, then remove duplicates to ensure one row per canonical entity.
merged = left.merge(right, on="source_id", how="inner")
merged = merged.drop_duplicates(subset=["canonical_id"])
```

If the block is substantial, strongly prefer extracting it into a helper method so the **docstring becomes the primary explanation**. If extraction is not done now, write the comment so that refactoring later is straightforward (describe inputs/outputs of the block).

---

## 6) Do not number notes

In code comments and docstrings, **do not** use fragile numbered notes like:

* `NOTE 1: ...`
* `NOTE 2: ...`

Prefer comments without tags for general explanations. But if a tag is necessary (e.g. follow-up is needed), use unnumbered tags instead:

* `TODO: ...`
* `WARNING: ...`
* `PERF: ...`
* `SECURITY: ...`

---

## 7) “What vs why”: allow “meta-what” when it explains intent at the right level

We still avoid line-by-line narration (e.g., “increment counter”), but we explicitly allow “meta-what” comments that describe what a *block* of code is doing, especially when the intent is not obvious from individual lines.

Rule of thumb:

* **Bad**: Restates a single obvious line.
* **Good**: Explains the intent of a loop/branch/multi-step block and how it supports the method’s purpose.

---

## Anti-patterns (still avoid)

* Outdated comments that contradict code.
* Changelog/history comments in source.
* Decorative dividers.
* Commented-out dead code.

---

## Quality checklist

Before finalizing code:

* Docstrings exist for every class and every function/method.
* Docstrings explain purpose, usage, flow, args, returns, and contract-level raises/side effects.
* Loops/comprehensions have intent comments (or are expanded for clarity).
* Branching has decision-logic comments.
* Multi-step blocks have meta-what + rationale comments.
* No numbered notes in comments/docstrings.
* Comments remain accurate and add real explanatory value.