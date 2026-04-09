# Community Module Migration Guide

## Community Module Changes

### 1. Module Location
- **Old**: `elitea_sdk.runtime.community`
- **New**: `elitea_sdk.community` (top-level)

### 2. Tool Management
- **New**: Community module now provides its own `get_tools()` and `get_toolkits()` functions
- **Benefit**: Self-contained community tool management, similar to the tools module pattern
- **Runtime Integration**: Runtime module now imports and delegates community tools instead of managing them directly

## Before (Old Structure)
```
src/elitea_sdk/
├── runtime/
│   ├── community/  # ❌ Old location
│   │   ├── analysis/
│   │   ├── browseruse/
│   │   ├── deep_researcher/
│   │   ├── eda/
│   │   └── utils.py
│   └── ...
└── tools/
```

## After (New Structure)
```
src/elitea_sdk/
├── runtime/
├── tools/
└── community/      # ✅ New location
    ├── analysis/
    ├── browseruse/
    ├── deep_researcher/
    ├── eda/
    └── utils.py
```

## Import Changes Required

### Old Imports (No longer work)
```python
# ❌ These imports will fail
from elitea_sdk.runtime.community.analysis.jira_analyse import AnalyseJira
from elitea_sdk.runtime.community.browseruse import BrowserUseToolkit
from elitea_sdk.runtime.community.utils import some_function
```

### New Imports (Use these instead)
```python
# ✅ Use these new imports
from elitea_sdk.community.analysis.jira_analyse import AnalyseJira
from elitea_sdk.community.browseruse import BrowserUseToolkit
from elitea_sdk.community.utils import some_function

# ✅ Or use the new consolidated imports
from elitea_sdk.community import get_tools, get_toolkits
from elitea_sdk.community import AnalyseJira, BrowserUseToolkit
```

## Files Updated

The following files have been automatically updated:
- `src/elitea_sdk/__init__.py` - Added community module export
- `src/elitea_sdk/community/__init__.py` - Added module structure with `get_tools()` and `get_toolkits()` functions
- `src/elitea_sdk/runtime/toolkits/tools.py` - Refactored to import and delegate community tools instead of managing them directly
- `pyproject.toml` - Updated to reference external requirements files instead of duplicating dependencies
- `requirements-*.txt` - Created separate requirements files for each module
- `MODULE_STRUCTURE.md` - Updated documentation

## Dependency Management Improvements

The project now uses a hybrid dependency management approach:
- **pyproject.toml**: References external requirements files using `dynamic` and `tool.setuptools.dynamic`
- **requirements-*.txt**: Contains the actual dependency lists for each module
- **Benefits**: No duplication, easier maintenance, supports both pip and setuptools workflows

## New Community Module Architecture

The community module now follows the same pattern as the tools module:
- Provides `get_tools(tools_list, elitea_client, llm)` for tool initialization
- Provides `get_toolkits()` for toolkit configuration schemas
- Exports individual toolkits for direct import
- Self-contained tool management without runtime dependency

## Installation

The community module now has its own optional dependency group:

```bash
# Install just community module
pip install elitea_sdk[community]

# Install all modules including community
pip install elitea_sdk[all]

# Install using requirements file
pip install -r requirements-community.txt
```

## Action Required

If you have any code that imports from `elitea_sdk.runtime.community`, update those imports to use `elitea_sdk.community` instead.
