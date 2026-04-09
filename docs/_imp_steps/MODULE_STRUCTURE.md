# EliteA SDK Module Structure

The EliteA SDK has been reorganized into three main modules:

## Modules

### Runtime (`elitea_sdk.runtime`)
Contains core functionality for building langchain agents:
- **agents/**: Agent implementations and parsers
- **clients/**: Core client implementations (artifact, datasource, prompt, etc.)
- **langchain/**: LangChain-specific integrations
- **llamaindex/**: LlamaIndex integrations
- **llms/**: Language model implementations
- **toolkits/**: Runtime toolkits
- **tools/**: Core tools
- **utils/**: Runtime utilities (including streamlit)

### Tools (`elitea_sdk.tools`)
Contains integrations with various services and platforms:
- Project management: Jira, Azure DevOps, Rally, TestRail, etc.
- Version control: GitHub, GitLab, Bitbucket
- Cloud services: AWS, Azure, GCP
- Document processing: Office365, PDF, PowerPoint
- Communication: Email, ServiceNow
- And many more...

### Community (`elitea_sdk.community`)
Contains community extensions and utilities:
- **analysis/**: Analysis tools and utilities (AnalyseJira, AnalyseAdo, AnalyseGitLab, AnalyseGithub)
- **browseruse/**: Browser automation tools (BrowserUseToolkit)
- **deep_researcher/**: Research and investigation tools
- **eda/**: Exploratory data analysis tools
- **utils.py**: Community utility functions

The community module provides its own `get_tools()` and `get_toolkits()` functions for managing community-specific tools, similar to the tools module.

## Installation Options

### 1. Minimal Core Installation
```bash
pip install elitea_sdk
```
This installs only the core dependencies.

### 2. Modular Installation
```bash
# Install runtime module
pip install elitea_sdk[runtime]

# Install tools module  
pip install elitea_sdk[tools]

# Install community extensions
pip install elitea_sdk[community]

# Install everything
pip install elitea_sdk[all]
```

### 3. Using Requirements Files
```bash
# Core dependencies
pip install -r requirements.txt

# Runtime module
pip install -r requirements-runtime.txt

# Tools module
pip install -r requirements-tools.txt

# Community extensions
pip install -r requirements-community.txt
```

## Dependency Management

The project uses a hybrid approach for managing dependencies:
- **Core dependencies**: Defined directly in `pyproject.toml` 
- **Module dependencies**: Defined in separate `requirements-*.txt` files and referenced in `pyproject.toml`
- **Development dependencies**: Defined directly in `pyproject.toml` (simple and stable)

This approach:
- ✅ Avoids duplication between `pyproject.toml` and requirements files
- ✅ Keeps module-specific dependencies organized in separate files
- ✅ Allows easy maintenance of complex dependency lists
- ✅ Supports both pip and setuptools installation methods

## Usage

### Importing Runtime Components
```python
from elitea_sdk.runtime.utils.streamlit import run_streamlit
from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.langchain.assistant import EliteAAssistant
```

### Importing Tools
```python
from elitea_sdk.tools.github import get_tools as get_github
from elitea_sdk.tools.jira import JiraToolkit
from elitea_sdk.tools.openapi import get_tools as get_openapi
```

### Importing Community Extensions
```python
from elitea_sdk.community.utils import community_function
from elitea_sdk.community.analysis import analysis_tools
from elitea_sdk.community.browseruse import browser_tools

# Or use the consolidated functions
from elitea_sdk.community import get_tools, get_toolkits
from elitea_sdk.community import AnalyseJira, BrowserUseToolkit
```

## Migration Notes

- The main import structure remains the same for backward compatibility
- `elitea_local.py` has been updated to use the new runtime module path
- **Community module** is now at the top level (`elitea_sdk.community`) instead of under runtime
- Dependencies are now organized by module for better dependency management
- Optional dependencies allow for lighter installations based on your needs
- Any imports from `elitea_sdk.runtime.community` should be updated to `elitea_sdk.community`

## Development

For development, install all dependencies:
```bash
pip install elitea_sdk[all,dev]
```

This includes development tools like pytest, black, flake8, and mypy.
