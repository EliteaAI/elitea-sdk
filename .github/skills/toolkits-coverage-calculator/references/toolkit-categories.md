# Toolkit Categories Reference

Complete list of toolkit classifications for coverage calculations.

---

## User-Facing Toolkits (Require Tests)

These toolkits expose tools for users to interact with external services. They **require test coverage**.

### By Domain

| Domain | Toolkits |
|--------|----------|
| **Version Control** | github, gitlab, gitlab_org, bitbucket, localgit, ado |
| **Issue Tracking** | jira, advanced_jira_mining, rally |
| **Documentation** | confluence, sharepoint |
| **Test Management** | xray, qtest, testrail, testio, zephyr, zephyr_enterprise, zephyr_essential, zephyr_scale, zephyr_squad, report_portal |
| **API Tools** | postman, openapi, custom_open_api |
| **Communication** | slack, gmail, yagmail |
| **CRM/ITSM** | salesforce, servicenow, keycloak, carrier |
| **Cloud** | aws, azure, gcp, k8s |
| **Data** | sql, pandas, elastic, bigquery, delta_lake |
| **Design** | figma |
| **Search** | azure_search |
| **Other** | ocr, pptx, memory, google_places |

---

## Framework Utilities (No Tests Required)

Infrastructure components that support toolkit development but don't expose user-facing tools.

| Utility | Location | Purpose |
|---------|----------|---------|
| **base** | `tools/base/` | BaseAction class, BaseTool, BaseToolkit |
| **browser** | `tools/browser/` | Browser automation support (empty) |
| **chunkers** | `tools/chunkers/` | Document chunking strategies |
| **llm** | `tools/llm/` | LLM integration utilities |
| **utils** | `tools/utils/` | Decorators, helpers, common functions |
| **vector_adapters** | `tools/vector_adapters/` | Vector store adapters |
| **code** | `tools/code/` | Code analysis base classes |

**Identifying characteristics:**
- No `get_available_tools()` method (or returns empty)
- Provides base classes or utilities
- Not directly invoked by users

---

## Container Directories

Directories that organize multiple related toolkits but don't contain tools themselves.

| Container | Sub-Toolkits | Notes |
|-----------|--------------|-------|
| `cloud/` | aws, azure, gcp, k8s | Cloud providers |
| `ado/` | repos, work_item, test_plan, wiki | Azure DevOps components |
| `aws/` | delta_lake | AWS-specific tools |
| `azure_ai/` | search | Azure AI services |
| `google/` | bigquery | Google Cloud services |

**Handling:** Analyze each subdirectory separately.
