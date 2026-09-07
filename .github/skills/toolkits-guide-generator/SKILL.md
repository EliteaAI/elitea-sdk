---
name: "toolkits-guide-generator"
description: "Generate or refresh the ELITEA SDK toolkits reference guide (toolkits_guide.md). The guide lists every registered toolkit with its configuration fields and available tools — used by agents as a knowledge base for tool selection and pipeline configuration. Use this skill when the guide is missing, stale, or when new toolkits have been added."
---

# Toolkit Guide Skill

This skill generates `.github/skills/toolkits-guide-generator/output/toolkits_guide.md` — a structured Markdown reference of all 47+ registered toolkits, their configuration fields, and available tools.

## When to Use This Skill

- An agent needs to know which toolkits exist and what tools they expose
- The guide is missing or needs to be refreshed after toolkit changes
- A new toolkit was added to `elitea_sdk/tools/__init__.py`
- An agent is configuring a pipeline and needs to look up available tools

## Key Paths

| Path | Purpose |
|------|---------|
| `elitea_sdk/tools/__init__.py` | Toolkit registry (`AVAILABLE_TOOLS`) — source of truth |
| `elitea_sdk/tools/<name>/` | Individual toolkit source directories |
| `.github/skills/toolkits-guide-generator/generate_toolkits_guide.py` | Generator script |
| `.github/skills/toolkits-guide-generator/output/toolkits_guide.md` | Generated output (knowledge base) |

## How to Generate / Refresh the Guide

Run from the repository root:

```bash
# AST-only mode — works without toolkit deps installed (recommended default)
python3 .github/skills/toolkits-guide-generator/generate_toolkits_guide.py --ast-only

# Full import mode — richer detail, requires: pip install -e '.[tools]'
python3 .github/skills/toolkits-guide-generator/generate_toolkits_guide.py

# Compact mode — tool names + descriptions only, no parameter tables (smaller file)
python3 .github/skills/toolkits-guide-generator/generate_toolkits_guide.py --ast-only --compact
```

Expected output:
```
✓ Generated .github/skills/toolkits-guide-generator/output/toolkits_guide.md
  Toolkits total      : 47
  Via AST (fallback)  : 47
  File size           : ~60 KB
```

## Guide Structure

Each toolkit section contains:

1. **Header** — toolkit label and registry key (e.g. `## GitHub (\`github\`)`)
2. **Categories / Tags** — from toolkit metadata
3. **Configuration table** — all config fields with types, required/optional, descriptions
4. **Tools** — each available tool as a subsection (`#### \`tool_name\``) with description and parameter table

## Generation Strategy

The script uses a two-pass approach:

| Pass | When used | Method |
|------|-----------|--------|
| Import | Toolkit deps installed (`pygithub`, `atlassian-python-api`, etc.) | Full Pydantic introspection via `model_construct().get_available_tools()` |
| AST | Deps missing (default) | Source-file AST parsing across all `.py` files in each toolkit directory |

The AST pass resolves:
- Inline string descriptions
- Named string constants (e.g. `GET_ISSUE_PROMPT`)
- Docstrings via `self.method.__doc__`

## Updating After Toolkit Changes

After adding a new toolkit or modifying `get_available_tools()` in an existing wrapper, re-run the generator:

```bash
python3 .github/skills/toolkits-guide-generator/generate_toolkits_guide.py --ast-only
```
