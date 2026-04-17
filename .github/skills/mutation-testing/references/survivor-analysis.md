# Survivor Analysis Protocol

How to correctly analyze and respond to surviving mutations.

---

## Core Principle

**When a mutation survives, NEVER write a test purely to kill it.**

A surviving mutation is a question: *"Does this logic change matter?"* Your job is to answer that question — not just make it go away.

---

## Required Workflow

For each survivor:

### 1. Identify what changed

What logic was mutated? Examples:
- `False` → `True`
- `>` → `>=`
- `and` → `or`
- `if x:` → `if not x:`

### 2. Understand the intent

Read the docstring, function name, and surrounding code. What *should* happen here?

### 3. Decide

| Situation | Action |
|-----------|--------|
| Intent is clear, current code is **correct** | Write a test asserting the correct behaviour (test passes, mutation is now killed legitimately) |
| Intent is clear, current code is **wrong** | Write a test asserting the correct behaviour (test **fails** on purpose — file a bug report, do NOT "fix" the test to match the wrong code) |
| Intent is **unclear** | Do NOT write a test. Report the ambiguity to the developer and wait for a decision. |

---

## The Mutation-Score Chasing Trap

Writing tests that assert the *current broken behaviour* purely to boost kill rate is **worse than leaving mutations alive**:

- It encodes bugs as expected behaviour
- It gives false confidence ("96% kill rate!") while actively hiding defects
- It makes genuine bug fixes break the test suite, making tests adversarial to developers

---

## Red Flags

Stop and report if you catch yourself:

- [ ] Writing a test and immediately checking whether it kills the surviving mutation (score-first thinking)
- [ ] Asserting a value you are not certain is the *intended correct* value
- [ ] Changing a test assertion from "what should be true" to "what currently happens"
- [ ] Getting a higher kill rate but the test feels semantically wrong

---

## Example Analysis

**Surviving mutation:**
```
line=24  operator=core/ReplaceFalseWithTrue  def=__init__
```

**Original code:**
```python
def __init__(self, autodetect_encoding=False):
```

**Mutated code:**
```python
def __init__(self, autodetect_encoding=True):
```

**Analysis:**
1. What changed? Default value flipped from `False` to `True`
2. What's the intent? The docstring says "autodetect if enabled" — so `False` means "don't autodetect by default"
3. Decision: The current code (`False`) is correct. Write a test that creates an instance with default args and asserts `obj.autodetect_encoding == False`

**Wrong approach:** Write `assert obj.works_fine()` — this doesn't validate the actual logic that was mutated.
