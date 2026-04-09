# Agent CLI Guide - Interactive and Handoff Modes

## Overview

The EliteA CLI supports working with agents in two modes:

1. **Interactive Mode** (`chat`) - Real-time conversation with an agent
2. **Handoff Mode** (`run`) - Single message execution for automation

Both modes support:
- Platform agents (from EliteA deployment)
- Local agent definitions (`.agent.md`, `.agent.yaml`, `.agent.json` files)
- Toolkit configurations
- Custom LLM settings

## Agent Definition Files

### Format: Markdown with YAML Frontmatter

Similar to GitHub Copilot agents, create `.agent.md` files:

```markdown
---
name: "my-custom-agent"
description: "Brief description of what the agent does"
model: "gpt-4o"
temperature: 0.7
max_tokens: 2000
tools:
  - jira
  - github
  - confluence
---

# Agent System Prompt

You are an expert assistant that helps with...

## Your Capabilities

- Capability 1
- Capability 2

## Guidelines

- Guideline 1
- Guideline 2
```

### Format: YAML

```yaml
# .github/agents/my-agent.agent.yaml
name: my-custom-agent
description: Brief description
model: gpt-4o
temperature: 0.7
max_tokens: 2000

tools:
  - jira
  - github

system_prompt: |
  You are an expert assistant that helps with...
  
  Your capabilities include:
  - Capability 1
  - Capability 2
```

### Format: JSON

```json
{
  "name": "my-custom-agent",
  "description": "Brief description",
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 2000,
  "tools": ["jira", "github"],
  "system_prompt": "You are an expert assistant..."
}
```

## Environment Variables

### Configuration in .env

The CLI supports configuration via environment variables in `.env` files:

```bash
# .env file
DEPLOYMENT_URL=https://your-deployment.elitea.ai
PROJECT_ID=12345
API_KEY=your_api_key

# Agent configuration
AGENTS_DIR=.github/agents

# Secrets for toolkits
JIRA_URL=https://company.atlassian.net
JIRA_EMAIL=dev@company.com
JIRA_API_KEY=your_token
GITHUB_APP_ID=123456
GITHUB_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----..."
```

### Variable Substitution

Use `${VAR_NAME}` or `$VAR_NAME` in agent definitions and toolkit configs:

**Agent with environment variables:**

```markdown
---
name: "dev-helper"
model: "gpt-4o"
tools:
  - jira
  - github
toolkit_configs:
  - file: "configs/jira-config.json"
  - config:
      toolkit_name: "github"
      type: "github"
      settings:
        github_app_id: "${GITHUB_APP_ID}"
        github_repository: "${GITHUB_REPO}"
---

# Development Helper

Working on ${PROJECT_NAME} project.
Repository: ${GITHUB_REPO}
```

**Toolkit config with secrets:**

```json
{
  "toolkit_name": "jira",
  "type": "jira",
  "settings": {
    "base_url": "${JIRA_URL}",
    "jira_configuration": {
      "username": "${JIRA_EMAIL}",
      "api_key": "${JIRA_API_KEY}"
    }
  }
}
```

### Configuring Agents Directory

By default, local agents are searched in `.github/agents`. Configure via:

```bash
# In .env file
AGENTS_DIR=custom/agents/path

# Or use --directory flag
elitea-cli agent list --local --directory custom/agents
```

## Toolkit Configuration

### Inline in Agent Definition

Include toolkit configs directly in agent YAML frontmatter:

```markdown
---
name: "my-agent"
tools:
  - jira
  - github
toolkit_configs:
  # Load from file
  - file: "configs/jira-config.json"
  
  # Inline configuration
  - config:
      toolkit_name: "github"
      type: "github"
      settings:
        github_app_id: "${GITHUB_APP_ID}"
        github_repository: "myorg/myrepo"
        selected_tools:
          - get_issue
          - list_pull_requests
---

# Agent prompt here...
```

### Via Command Line

Pass toolkit configs as options:

```bash
elitea-cli agent chat my-agent \
    --toolkit-config configs/jira-config.json \
    --toolkit-config configs/github-config.json
```

### Auto-Adding Toolkits to Tools

When you provide `--toolkit-config`, the toolkit is automatically added to the agent's tools list if not already present:

```bash
# Agent has tools: [jira]
# Command adds GitHub config
elitea-cli agent chat my-agent --toolkit-config github-config.json

# Agent now has tools: [jira, github]
# GitHub toolkit auto-added
```

This works with both command-line configs and configs from agent frontmatter.

## Commands

### List Agents

```bash
# List platform agents
elitea-cli agent list

# List local agent definition files
elitea-cli agent list --local

# List from custom directory
elitea-cli agent list --local --directory custom/agents
```

### Show Agent Details

```bash
# Show platform agent
elitea-cli agent show my-agent
elitea-cli agent show 123  # by ID

# Show local agent
elitea-cli agent show .github/agents/sdk-dev.agent.md

# JSON output
elitea-cli --output json agent show my-agent
```

### Interactive Chat Mode

```bash
# Chat with platform agent
elitea-cli agent chat my-agent

# Chat with local agent
elitea-cli agent chat .github/agents/sdk-dev.agent.md

# With toolkit configurations
elitea-cli agent chat my-agent \
    --toolkit-config jira-config.json \
    --toolkit-config github-config.json

# With custom model settings
elitea-cli agent chat my-agent \
    --model gpt-4o \
    --temperature 0.8 \
    --max-tokens 3000

# Continue previous conversation
elitea-cli agent chat my-agent --thread-id abc123
```

#### Interactive Commands

While in chat mode, use these commands:

- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/save` - Save conversation to file
- `/help` - Show help
- `exit` or `quit` - End conversation

Example session:

```
$ elitea-cli agent chat sdk-dev

🤖 Starting chat with platform agent: sdk-dev
Type your message and press Enter. Type 'exit' or 'quit' to end.

Commands:
  /clear    - Clear conversation history
  /history  - Show conversation history
  /save     - Save conversation to file
  /help     - Show this help

sdk-dev> How do I create a new toolkit?

To create a new toolkit in EliteA SDK, follow these steps:

1. Create the directory structure...
[agent response continues]

sdk-dev> What base class should I use?

Based on your requirements, you should use...

sdk-dev> /save
Save to file (default: conversation.json): my-session.json
Conversation saved to my-session.json

sdk-dev> exit

Goodbye! 👋
```

### Handoff Mode (Single Message)

```bash
# Simple query
elitea-cli agent run my-agent "What is the status of JIRA-123?"

# With local agent
elitea-cli agent run .github/agents/sdk-dev.agent.md \
    "Create a new toolkit for Stripe API"

# With toolkit configs
elitea-cli agent run my-agent "Search for critical bugs" \
    --toolkit-config jira-config.json

# JSON output for scripting
elitea-cli --output json agent run my-agent "Query" \
    | jq -r '.response'

# Save thread for continuation
elitea-cli agent run my-agent "Start complex task" \
    --save-thread thread.txt

# Continue from saved thread
THREAD_ID=$(jq -r '.thread_id' thread.txt)
elitea-cli agent run my-agent "Continue task" \
    --thread-id $THREAD_ID
```

## Toolkit Configurations

### Create Toolkit Config Files

```bash
# Jira config
cat > jira-config.json <<EOF
{
  "toolkit_name": "jira",
  "type": "jira",
  "settings": {
    "base_url": "https://jira.company.com",
    "cloud": true,
    "jira_configuration": {
      "username": "user@company.com",
      "api_key": "$JIRA_API_KEY"
    }
  }
}
EOF

# GitHub config
cat > github-config.json <<EOF
{
  "toolkit_name": "github",
  "type": "github",
  "settings": {
    "github_app_id": "123456",
    "github_app_private_key": "$GITHUB_PRIVATE_KEY",
    "github_repository": "owner/repo"
  }
}
EOF
```

### Use Multiple Toolkits

```bash
elitea-cli agent chat my-agent \
    --toolkit-config jira-config.json \
    --toolkit-config github-config.json \
    --toolkit-config confluence-config.json
```

## Example Workflows

### 1. Development Assistant

Create a local agent for development help:

```markdown
---
name: "dev-assistant"
description: "Helps with SDK development and testing"
model: "gpt-4o"
temperature: 0.3
tools:
  - github
  - jira
---

# Development Assistant

You are an expert development assistant for the EliteA SDK project.

## Your Role

- Help developers write and test toolkits
- Review code and suggest improvements
- Explain SDK patterns and best practices
- Debug issues and errors

## Guidelines

- Always provide working code examples
- Reference actual SDK files when relevant
- Test code before suggesting
- Follow SDK conventions
```

Use it:

```bash
# Interactive development help
elitea-cli agent chat .github/agents/dev-assistant.agent.md

dev-assistant> How do I add pagination to my API wrapper?

[Agent provides detailed guidance with code examples]

dev-assistant> Review this code: [paste code]

[Agent reviews and suggests improvements]
```

### 2. Automated Code Review

```bash
# Create review agent
cat > .github/agents/code-reviewer.agent.md <<'EOF'
---
name: "code-reviewer"
description: "Reviews code changes for quality and conventions"
model: "gpt-4o"
temperature: 0.2
---

# Code Reviewer

You review code changes for:
- Adherence to SDK patterns
- Code quality and style
- Potential bugs or issues
- Documentation completeness
EOF

# Use in CI/CD
git diff main | elitea-cli agent run .github/agents/code-reviewer.agent.md \
    "Review these changes: $(cat)" \
    --output json \
    | jq -r '.response' > review.txt
```

### 3. Interactive Debugging

```bash
# Start debugging session
elitea-cli agent chat sdk-dev \
    --toolkit-config my-toolkit-config.json

sdk-dev> I'm getting "Tool 'get_issue' not found" error

[Agent helps diagnose]

sdk-dev> Here's my code: [paste]

[Agent identifies issue]

sdk-dev> Can you show me the correct implementation?

[Agent provides solution]
```

### 4. Batch Processing

```bash
# Process multiple queries
cat queries.txt | while read query; do
    echo "Processing: $query"
    elitea-cli agent run my-agent "$query" \
        --output json \
        | jq -r '.response' >> results.txt
done
```

### 5. Multi-Step Task Automation

```bash
#!/bin/bash
# automated-workflow.sh

# Step 1: Analyze requirements
response=$(elitea-cli --output json agent run my-agent \
    "Analyze requirements for new feature" \
    --save-thread thread.json | jq -r '.response')

echo "Analysis: $response"

# Step 2: Create implementation plan
thread_id=$(jq -r '.thread_id' thread.json)
plan=$(elitea-cli --output json agent run my-agent \
    "Create implementation plan based on analysis" \
    --thread-id $thread_id | jq -r '.response')

echo "Plan: $plan"

# Step 3: Generate code
code=$(elitea-cli --output json agent run my-agent \
    "Generate code for the first step" \
    --thread-id $thread_id | jq -r '.response')

echo "$code" > generated.py
```

## Local Agent Development

### Directory Structure

```
.github/
└── agents/
    ├── sdk-dev.agent.md          # SDK development expert
    ├── code-reviewer.agent.md    # Code review agent
    ├── debugging.agent.md        # Debugging assistant
    └── documentation.agent.md    # Docs writer
```

### Sharing Agents

Agents are just files - share via:
- Git repository
- Team shared folder
- Package as part of project

### Versioning

Use Git to version your agent definitions:

```bash
git add .github/agents/
git commit -m "Update sdk-dev agent with new capabilities"
git tag agent-v1.2
```

## Best Practices

### 1. Keep Prompts Focused

```markdown
---
name: "jira-expert"
description: "Expert in Jira toolkit development and troubleshooting"
---

# Jira Toolkit Expert

You ONLY help with Jira toolkit development.

Don't try to help with other toolkits - refer users to appropriate agents.
```

### 2. Include Examples

```markdown
## Examples

### Good Tool Method
```python
def get_issue(self, issue_key: str) -> dict:
    """Get a Jira issue by key."""
    return self._make_request(f"/issue/{issue_key}")
```

### Using the API
```python
wrapper = JiraApiWrapper(...)
issue = wrapper.get_issue("PROJ-123")
```
```

### 3. Version Your Prompts

```markdown
---
name: "sdk-dev"
version: "2.0"
updated: "2025-11-27"
---
```

### 4. Test Locally First

```bash
# Test your agent before committing
elitea-cli agent chat .github/agents/new-agent.agent.md

# Verify it works correctly
new-agent> Test query

# Refine and iterate
```

### 5. Document Your Agents

Create an index file:

```markdown
# .github/agents/README.md

# Available Agents

## sdk-dev
**File**: `sdk-dev.agent.md`
**Purpose**: Expert in EliteA SDK toolkit development
**Usage**: `elitea-cli agent chat .github/agents/sdk-dev.agent.md`

## code-reviewer
**File**: `code-reviewer.agent.md`
**Purpose**: Reviews code for quality and conventions
**Usage**: `elitea-cli agent run .github/agents/code-reviewer.agent.md "Review: [code]"`
```

## Troubleshooting

### Agent Not Found

```bash
# Check local agents
elitea-cli agent list --local

# Check platform agents
elitea-cli agent list

# Show agent details
elitea-cli agent show .github/agents/my-agent.agent.md
```

### Toolkit Config Issues

```bash
# Verify toolkit config schema
elitea-cli toolkit schema jira

# Test toolkit separately
elitea-cli toolkit test jira --tool get_issue --config jira-config.json --param issue_key=TEST-1
```

### Conversation Not Persisting

- Thread persistence requires saving thread ID
- Use `--save-thread` to save thread info
- Pass `--thread-id` to continue

## Advanced Usage

### Custom Agent with Templates

```markdown
---
name: "template-agent"
description: "Uses templates for consistent responses"
---

# Template-Based Agent

## Templates

### Code Review Template
```
**File**: {filename}
**Issues Found**: {count}

{issues_list}

**Recommendations**:
{recommendations}
```

Always format code reviews using this template.
```

### Environment-Aware Agents

```bash
# Production agent
export ENV=production
elitea-cli agent run .github/agents/deploy-agent.agent.md \
    "Deploy to $ENV"

# Development agent
export ENV=development
elitea-cli agent run .github/agents/test-agent.agent.md \
    "Run tests in $ENV"
```

### Chaining Agents

```bash
#!/bin/bash
# Chain multiple agents

# Agent 1: Analyze
analysis=$(elitea-cli agent run analyzer.agent.md "Analyze this: $INPUT" --output json | jq -r '.response')

# Agent 2: Design (based on analysis)
design=$(elitea-cli agent run designer.agent.md "Design based on: $analysis" --output json | jq -r '.response')

# Agent 3: Implement (based on design)
code=$(elitea-cli agent run implementer.agent.md "Implement: $design" --output json | jq -r '.response')

echo "$code" > output.py
```

## Reference

### Agent Definition Schema

```yaml
name: string                    # Required: Agent name
description: string             # Recommended: Brief description
model: string                   # Optional: LLM model to use
temperature: float              # Optional: 0.0 to 1.0
max_tokens: integer             # Optional: Max response tokens
tools: array[string]            # Optional: Toolkit names to enable
system_prompt: string           # Required: Agent instructions
version: string                 # Optional: Version for tracking
updated: date                   # Optional: Last update date
```

### Command Quick Reference

```bash
# List
elitea-cli agent list                          # Platform agents
elitea-cli agent list --local                  # Local agent files

# Show
elitea-cli agent show AGENT                    # Details

# Chat (interactive)
elitea-cli agent chat AGENT                    # Start chat
elitea-cli agent chat AGENT --toolkit-config CONFIG

# Run (handoff)
elitea-cli agent run AGENT "MESSAGE"           # Single message
elitea-cli agent run AGENT "MESSAGE" --save-thread FILE
```

## Examples Directory

Create reusable agents in your project:

```
.github/agents/
├── README.md                     # Index of all agents
├── sdk-dev.agent.md             # SDK development
├── code-reviewer.agent.md       # Code review
├── test-writer.agent.md         # Test generation
├── doc-writer.agent.md          # Documentation
├── debugger.agent.md            # Debugging help
└── examples/
    ├── jira-expert.agent.md     # Jira specialist
    ├── github-expert.agent.md   # GitHub specialist
    └── aws-expert.agent.md      # AWS specialist
```

---

**Next Steps**:
1. Create your first agent: `.github/agents/my-agent.agent.md`
2. Test it: `elitea-cli agent chat .github/agents/my-agent.agent.md`
3. Use in workflows: `elitea-cli agent run ...`
