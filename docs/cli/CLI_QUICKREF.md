# elitea-cli Quick Reference

## Installation
```bash
cd elitea-sdk
pip install -e ".[cli]"
```

## Setup Authentication
```bash
cat > .env <<EOF
DEPLOYMENT_URL=https://api.elitea.ai
PROJECT_ID=123
API_KEY=your_api_key
EOF
```

## Essential Commands

### Configuration
```bash
elitea-cli config                     # Show configuration
elitea-cli --debug config             # Debug mode
elitea-cli --output json config       # JSON output
```

### Toolkits
```bash
elitea-cli toolkit list               # List all toolkits
elitea-cli toolkit schema jira        # Show schema
elitea-cli toolkit tools jira         # List tools
```

### Testing
```bash
# Basic test
elitea-cli toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123

# Multiple params
elitea-cli toolkit test github \
    --tool get_issue \
    --config github-config.json \
    --param owner=user \
    --param repo=myrepo \
    --param issue_number=42

# Custom LLM
elitea-cli toolkit test jira \
    --tool search_issues \
    --config jira-config.json \
    --param jql="project = PROJ" \
    --llm-model gpt-4o \
    --temperature 0.7

# JSON output
elitea-cli --output json toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123 | jq '.'
```

## Config File Example
```json
{
  "base_url": "https://jira.company.com",
  "cloud": true,
  "jira_configuration": {
    "username": "user@company.com",
    "api_key": "your_api_key"
  }
}
```

## Common Options
```bash
--env-file PATH          # Use different .env file
--debug                  # Enable debug logging
--output [text|json]     # Output format
```

## Scripting Example
```bash
#!/bin/bash
result=$(elitea-cli --output json toolkit test jira \
    --tool get_issue \
    --config jira-config.json \
    --param issue_key=PROJ-123)

if echo "$result" | jq -e '.success' > /dev/null; then
    echo "✓ Test passed"
    echo "$result" | jq -r '.result'
else
    echo "✗ Test failed: $(echo "$result" | jq -r '.error')"
    exit 1
fi
```

## Get Help
```bash
elitea-cli --help                     # General help
elitea-cli toolkit --help             # Toolkit commands
elitea-cli toolkit test --help        # Test command options
@sdk-dev /cli-testing                # Copilot prompt
```

## Documentation
- **Quick start**: `CLI_GUIDE.md`
- **Complete guide**: `.github/prompts/cli-testing.prompt.md`
- **Agent help**: `@sdk-dev`

## Troubleshooting

**Missing config?**
```bash
elitea-cli config  # Check what's missing
```

**Toolkit not found?**
```bash
elitea-cli toolkit list  # See available
```

**Tool not found?**
```bash
elitea-cli toolkit tools jira --config jira-config.json
```

**Debug issues?**
```bash
elitea-cli --debug toolkit test ...
```
