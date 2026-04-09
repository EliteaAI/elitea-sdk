# EliteA SDK CLI - Installation and Usage Guide

## What is elitea-cli?

`elitea-cli` is a command-line interface for testing EliteA SDK toolkits and agents directly from your terminal. It provides an alternative to the Streamlit web interface, offering:

- **Fast, scriptable testing** - Perfect for development workflows
- **GitHub Copilot integration** - Direct terminal access for AI assistance
- **CI/CD compatibility** - JSON output for automation
- **Simple authentication** - Uses `.env` files like SDK tests
- **No browser required** - Pure command-line experience

## Installation

### 1. Install the SDK with CLI Support

```bash
cd elitea-sdk

# Install with CLI dependencies
pip install -e ".[cli,runtime]"

# Or just CLI if runtime is already installed
pip install -e ".[cli]"
```

This installs:
- `click` - CLI framework
- `rich` - Terminal formatting
- `python-dotenv` - Already included in SDK

### 2. Set Up Authentication

Create a `.env` file in your working directory:

```bash
cat > .env <<EOF
DEPLOYMENT_URL=https://api.elitea.ai
PROJECT_ID=123
API_KEY=your_api_key_here
EOF
```

### 3. Verify Installation

```bash
# Check if CLI is installed
elitea-cli --help

# Verify configuration
elitea-cli config
```

## Quick Start

### Chat with an Agent

```bash
# 1. Set up auth (if not done)
cat > .env <<EOF
DEPLOYMENT_URL=https://api.elitea.ai
PROJECT_ID=123
API_KEY=your_api_key
EOF

# 2. Chat with local agent
elitea-cli agent chat .github/agents/sdk-dev.agent.md

# 3. Or run single query
elitea-cli agent run .github/agents/sdk-dev.agent.md \
    "How do I create a new toolkit?"
```

### Test a Toolkit in 3 Steps

```bash
# 1. List available toolkits
elitea-cli toolkit list

# 2. Check toolkit schema
elitea-cli toolkit schema jira

# 3. Test a tool
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123
```

### Create Your First Config

```bash
# Create Jira configuration file
cat > jira-config.json <<EOF
{
  "base_url": "https://jira.company.com",
  "cloud": true,
  "jira_configuration": {
    "username": "user@company.com",
    "api_key": "$JIRA_API_KEY"
  }
}
EOF

# Test it
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123
```

## Core Commands

### Configuration

```bash
# Show current configuration
elitea-cli config

# Use different .env file
elitea-cli --env-file .env.staging config

# Enable debug logging
elitea-cli --debug toolkit list
```

### Toolkit Management

```bash
# List all toolkits
elitea-cli toolkit list

# Show failed imports
elitea-cli toolkit list --failed

# Get toolkit schema
elitea-cli toolkit schema jira

# List available tools
elitea-cli toolkit tools jira --config jira-config.json
```

### Testing Toolkits

```bash
# Test with config files
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --params params.json

# Test with inline parameters
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123 \
    --param fields=summary,status

# Get JSON output
elitea-cli --output json toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123
```

## Common Workflows

### Developing a New Toolkit

```bash
# 1. Check if your toolkit is registered
elitea-cli toolkit list | grep mytoolkit

# 2. Check the schema
elitea-cli toolkit schema mytoolkit

# 3. Create test config
cat > mytoolkit-config.json <<EOF
{
  "api_key": "test_key",
  "base_url": "https://api.example.com"
}
EOF

# 4. Test each tool
elitea-cli --debug toolkit test mytoolkit \
    --tool my_first_tool \
    --config mytoolkit-config.json \
    --param param1=value1

# 5. Iterate: edit code, test again
# (No need to restart anything - CLI loads fresh each time)
```

### Testing Multiple Scenarios

```bash
# Create a test script
cat > test-jira.sh <<'EOF'
#!/bin/bash
set -e

echo "Testing Jira toolkit..."

# Test 1: Get issue
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123

# Test 2: Search issues  
elitea-cli toolkit test jira \
    --tool search_issues \
    --config jira-config.json \
    --param jql="project = PROJ AND status = Open"

echo "All tests passed!"
EOF

chmod +x test-jira.sh
./test-jira.sh
```

### Debugging Issues

```bash
# Enable debug logging
elitea-cli --debug toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123

# Check if toolkit loaded
elitea-cli toolkit list | grep jira

# Check for import failures
elitea-cli toolkit list --failed

# Verify configuration
elitea-cli config
```

## Advanced Usage

### JSON Output for Scripting

```bash
# Capture output
result=$(elitea-cli --output json toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123)

# Parse with jq
echo "$result" | jq -r '.result.summary'
echo "$result" | jq -r '.execution_time_seconds'

# Check success
if echo "$result" | jq -e '.success' > /dev/null; then
    echo "Test passed"
else
    echo "Test failed: $(echo "$result" | jq -r '.error')"
    exit 1
fi
```

### Environment Variable Substitution

```bash
# Template config file
cat > jira-config.template.json <<EOF
{
  "base_url": "\${JIRA_URL}",
  "cloud": true,
  "jira_configuration": {
    "username": "\${JIRA_USER}",
    "api_key": "\${JIRA_API_KEY}"
  }
}
EOF

# Substitute variables
export JIRA_URL="https://jira.company.com"
export JIRA_USER="user@company.com"
export JIRA_API_KEY="secret"

envsubst < jira-config.template.json > jira-config.json

# Test
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123
```

### CI/CD Integration

```yaml
# .github/workflows/test-toolkits.yml
name: Test Toolkits

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install SDK with CLI
        run: |
          cd elitea-sdk
          pip install -e ".[cli,runtime]"
      
      - name: Create .env
        run: |
          echo "DEPLOYMENT_URL=${{ secrets.DEPLOYMENT_URL }}" > .env
          echo "PROJECT_ID=${{ secrets.PROJECT_ID }}" >> .env
          echo "API_KEY=${{ secrets.API_KEY }}" >> .env
      
      - name: Test Jira Toolkit
        run: |
          elitea-cli --output json toolkit test jira \
            --tool get_issue \
            --config jira-config.json \
            --param issue_key=PROJ-123 > result.json
          
          # Check if test passed
          if ! jq -e '.success' result.json; then
            echo "Test failed!"
            jq '.error' result.json
            exit 1
          fi
```

## Comparison: CLI vs Streamlit

| Feature | CLI | Streamlit |
|---------|-----|-----------|
| **Speed** | Fast (< 1s startup) | Slow (5-10s startup) |
| **Automation** | Excellent (scriptable) | Poor (web-only) |
| **Copilot Integration** | Perfect | None |
| **Output Format** | Text or JSON | Visual only |
| **Debugging** | Debug logs | Breakpoints + UI |
| **Use Case** | Development, CI/CD | Interactive exploration |
| **Learning Curve** | Low (command-line) | Medium (web UI) |

**When to use CLI:**
- Development and testing
- CI/CD pipelines
- Scripting and automation
- GitHub Copilot workflows
- Quick iteration

**When to use Streamlit:**
- First-time toolkit exploration
- Visual debugging
- Non-technical users
- Agent chat testing
- Need conversation history UI

## Troubleshooting

### "Missing required configuration"

**Problem**: `.env` file not found or incomplete

**Solution**:
```bash
# Check current config
elitea-cli config

# Create complete .env file
cat > .env <<EOF
DEPLOYMENT_URL=https://api.elitea.ai
PROJECT_ID=123
API_KEY=your_api_key
EOF
```

### "Toolkit 'xxx' not found"

**Problem**: Toolkit not installed or name incorrect

**Solution**:
```bash
# List available toolkits
elitea-cli toolkit list

# Check failed imports
elitea-cli toolkit list --failed

# Install toolkit dependencies if needed
pip install -e ".[tools]"
```

### "Tool 'xxx' not found"

**Problem**: Tool name doesn't match

**Solution**:
```bash
# List available tools for toolkit
elitea-cli toolkit tools jira --config jira-config.json

# Use exact tool name from list
```

### "Command 'elitea-cli' not found"

**Problem**: CLI not installed or not in PATH

**Solution**:
```bash
# Reinstall with CLI support
cd elitea-sdk
pip install -e ".[cli]"

# Or use Python module syntax
python -m elitea_sdk.cli toolkit list
```

## Getting Help

### Command Help

```bash
# General help
elitea-cli --help

# Command-specific help
elitea-cli toolkit --help
elitea-cli toolkit test --help
```

### With GitHub Copilot

```bash
# Ask sdk-dev agent for CLI help
@sdk-dev /cli-testing how do I test a toolkit?

# Get toolkit-specific guidance
@sdk-dev how do I test jira toolkit with CLI?
```

### Documentation

- **CLI Testing Prompt**: `@sdk-dev /cli-testing`
- **SDK Dev Agent**: `@sdk-dev`
- **Complete Guide**: `.github/prompts/cli-testing.prompt.md`

## Examples

### Example 1: Simple Tool Test

```bash
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123
```

### Example 2: Multiple Parameters

```bash
elitea-cli toolkit test github \
    --tool get_issue \
    --config github-config.json \
    --param owner=octocat \
    --param repo=Hello-World \
    --param issue_number=1
```

### Example 3: Custom LLM Settings

```bash
elitea-cli toolkit test jira \
    --tool search_issues \
    --config jira-config.json \
    --param jql="project = PROJ" \
    --llm-model gpt-4o \
    --temperature 0.7 \
    --max-tokens 2000
```

### Example 4: JSON Output

```bash
elitea-cli --output json toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123 \
    | jq '.'
```

## Next Steps

1. **Install CLI**: `pip install -e ".[cli]"`
2. **Set up auth**: Create `.env` file
3. **Try it**: `elitea-cli toolkit list`
4. **Test toolkit**: `elitea-cli toolkit test ...`
5. **Read guide**: `@sdk-dev /cli-testing`

---

**Version**: 1.0  
**Date**: November 2025  
**Requires**: elitea-sdk >= 0.3.457, Python >= 3.10
