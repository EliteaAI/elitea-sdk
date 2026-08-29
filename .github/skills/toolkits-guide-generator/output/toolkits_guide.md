# Alita SDK — Toolkits Guide

Auto-generated reference for all available toolkits and their tools.
Use this guide to understand each toolkit's configuration fields and available tools.

---

## Table of Contents

- [ADO boards (`ado_boards`)](#user-content-ado-boards)
- [ADO plans (`ado_plans`)](#user-content-ado-plans)
- [ADO repos (`ado_repos`)](#user-content-ado-repos)
- [ADO wiki (`ado_wiki`)](#user-content-ado-wiki)
- [Artifact (`artifact`)](#user-content-artifact)
- [Bitbucket (`bitbucket`)](#user-content-bitbucket)
- [Carrier (`carrier`)](#user-content-carrier)
- [Confluence (`confluence`)](#user-content-confluence)
- [OpenAPI (`custom_open_api`)](#user-content-custom-open-api)
- [Figma (`figma`)](#user-content-figma)
- [GitHub (`github`)](#user-content-github)
- [GitLab (`gitlab`)](#user-content-gitlab)
- [GitLab Org (`gitlab_org`)](#user-content-gitlab-org)
- [Google Places (`google_places`)](#user-content-google-places)
- [Jira (`jira`)](#user-content-jira)
- [Remote MCP (`mcp`)](#user-content-mcp)
- [Memory (`memory`)](#user-content-memory)
- [OpenAPI (`openapi`)](#user-content-openapi)
- [Postman (`postman`)](#user-content-postman)
- [PPTX (`pptx`)](#user-content-pptx)
- [QTest (`qtest`)](#user-content-qtest)
- [Rally (`rally`)](#user-content-rally)
- [Report Portal (`report_portal`)](#user-content-report-portal)
- [Salesforce (`salesforce`)](#user-content-salesforce)
- [ServiceNow (`service_now`)](#user-content-service-now)
- [Sharepoint (`sharepoint`)](#user-content-sharepoint)
- [Slack (`slack`)](#user-content-slack)
- [Sonar (`sonar`)](#user-content-sonar)
- [SQL (`sql`)](#user-content-sql)
- [TestIO (`testio`)](#user-content-testio)
- [Testrail (`testrail`)](#user-content-testrail)
- [XRAY cloud (`xray_cloud`)](#user-content-xray-cloud)
- [Zephyr Enterprise (`zephyr_enterprise`)](#user-content-zephyr-enterprise)
- [Zephyr Essential (`zephyr_essential`)](#user-content-zephyr-essential)
- [Zephyr Scale (`zephyr_scale`)](#user-content-zephyr-scale)
- [Zephyr Squad (`zephyr_squad`)](#user-content-zephyr-squad)

---

## ADO boards (`ado_boards`)
**Categories**: project management | **Tags**: work item management, issue tracking, agile boards
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ado_configuration` | AdoConfiguration | Yes |  |
| `project` | str | Yes | ADO project name |
| `limit` | Optional[int] | No | Default ADO boards result limit (can be overridden by agent instructions) |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `search_work_items`
> Search for work items using a WIQL query and dynamically fetch fields based on the query.

#### `create_work_item`
> Create a work item in Azure DevOps.

#### `update_work_item`
> Updates existing work item per defined data

#### `delete_work_item`
> Delete a work item from Azure DevOps by ID.

#### `get_work_item`
> Get a single work item by ID.

#### `link_work_items`
> Add the relation to the source work item with an appropriate attributes if any. User may pass attributes like name, etc.

#### `get_relation_types`
> Returns dict of possible relation types per syntax: 'relation name': 'relation reference name'.

#### `get_comments`
> Get comments for work item by ID.

#### `link_work_items_to_wiki_page`
> Links one or more work items to a specific wiki page using an ArtifactLink.

#### `unlink_work_items_from_wiki_page`
> Unlinks one or more work items from a specific wiki page by removing the ArtifactLink.

#### `get_work_item_type_fields`
> Get formatted information about available fields for a specific work item type.


---

## ADO plans (`ado_plans`)
**Categories**: test management | **Tags**: test case management, qa
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ado_configuration` | AdoConfiguration | Yes |  |
| `project` | str | Yes | ADO project name |
| `limit` | Optional[int] | No | ADO plans limit used for limitation of the list with results |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `create_test_plan`
> Create a test plan in Azure DevOps.

#### `delete_test_plan`
> Delete a test plan in Azure DevOps.

#### `get_test_plan`
> Get a test plan or list of test plans in Azure DevOps.

#### `create_test_suite`
> Create a test suite in Azure DevOps.

#### `delete_test_suite`
> Delete a test suite in Azure DevOps.

#### `get_test_suite`
> Get a test suite or list of test suites in Azure DevOps.

#### `add_test_case`
> Add a test case to a suite in Azure DevOps.

#### `create_test_case`
> Creates a new test case in specified suite in Azure DevOps.

#### `create_test_cases`
> Creates new test cases in specified suite in Azure DevOps.

#### `get_test_case`
> Get a test case from a suite in Azure DevOps with all custom fields.

#### `get_test_cases`
> Get test cases from a suite in Azure DevOps with all custom fields.

#### `get_all_test_case_fields_for_project`
> Get formatted information about available Test Case fields and their metadata.


---

## ADO repos (`ado_repos`)
**Categories**: code repositories | **Tags**: code, repository, version control
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ado_configuration` | AdoConfiguration | No |  |
| `project` | str | Yes | ADO project name |
| `repository_id` | str | Yes | ADO repository ID or name |
| `base_branch` | Optional[str] | No | ADO base branch (e.g., main) |
| `active_branch` | Optional[str] | No | ADO active branch (e.g., main) |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `list_branches_in_repo`
> Fetches a list of all branches in the repository.

#### `set_active_branch`
> Equivalent to `git checkout branch_name` for this Agent.

#### `list_files`
> Recursively fetches files from a directory in the repo.

#### `list_open_pull_requests`
> Fetches all open pull requests from the Azure DevOps repository.

#### `get_pull_request`
> Fetches particular pull request from the Azure DevOps repository.

#### `list_pull_request_files`
> Fetches the files and their diffs included in a pull request.

#### `create_branch`
> Create a new branch in Azure DevOps, and set it as the active bot branch.

#### `read_file`
> Read a file from this agent's branch, defined by self.active_branch,

#### `create_file`
> Creates a new file on the Azure DevOps repo

#### `update_file`

#### `delete_file`
> Deletes a file from the repository in Azure DevOps.

#### `get_work_items`
> Fetches a specific work item and its first 10 comments from Azure DevOps.

#### `comment_on_pull_request`
> Adds a comment to a pull request in Azure DevOps. Supports both general pull request comments and inline comments.

#### `create_pull_request`
> Creates a pull request in Azure DevOps from the active branch to the base branch mentioned in params.

#### `get_commits`
> Retrieves a list of commits from the repository.


---

## ADO wiki (`ado_wiki`)
**Categories**: documentation | **Tags**: knowledge base, documentation management, wiki
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ado_configuration` | AdoConfiguration | Yes |  |
| `project` | str | Yes | ADO project name |
| `default_wiki_identifier` | Optional[str] | No | Default Wiki Identifier (Wiki ID or wiki name). If provided, this identifier will be used when tools are invoked without explicitly specifying a wiki identifier. |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_wiki`

#### `get_wiki_page`

#### `get_wiki_page_by_path`

#### `get_wiki_page_by_id`

#### `delete_page_by_path`

#### `delete_page_by_id`

#### `modify_wiki_page`

#### `rename_wiki_page`


---

## Artifact (`artifact`)
**Categories**: storage | **Tags**: artifact, file storage, bucket, files
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bucket` | str | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `list_files`
> List files in the artifact bucket. By default lists immediate children (files and subfolders). Use folder parameter to scope listing to a specific prefix/path. Use recursive=True to get all files under the path.

#### `create_file`

#### `read_file`
> Read a file from the artifact bucket. Supports full filepath (/{bucket}/{filename}) from attachment descriptions or filename+bucket_name. For large text files that exceed size limits, use start_line/end_line to read specific portions.

#### `get_file_type`
> Detect the file type of a file using content analysis. More reliable than extension-based detection as it analyzes file magic bytes. Useful for verifying file types before processing or after generation.

#### `delete_file`
> Delete a file in the artifact

#### `append_data`

#### `create_new_bucket`
> Create a new bucket.


---

## Bitbucket (`bitbucket`)
**Categories**: code repositories | **Tags**: bitbucket, git, repository, code, version control
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project` | str | Yes | Project/Workspace |
| `repository` | str | Yes | Repository |
| `branch` | str | No | Main branch |
| `cloud` | Optional[bool] | No | Hosting Option |
| `bitbucket_configuration` | BitbucketConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `create_branch`

#### `delete_branch`

#### `list_branches_in_repo`

#### `list_files`

#### `create_pull_request`

#### `create_file`

#### `read_file`

#### `update_file`

#### `set_active_branch`

#### `get_pull_requests_commits`

#### `get_pull_request`

#### `get_pull_requests_changes`

#### `add_pull_request_comment`

#### `close_pull_request`


---

## Carrier (`carrier`)
**Categories**: testing | **Tags**: carrier, ticket management, log management
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_id` | Optional[str] | No | Optional project ID for scoped operations |
| `carrier_configuration` | CarrierConfiguration | Yes | Carrier Configuration |

### Tools

#### `get_ticket_list`

#### `create_ticket`

#### `get_reports`

#### `get_report_by_id`

#### `add_tag_to_report`

#### `create_excel_report`

#### `get_tests`

#### `get_test_by_id`

#### `run_test_by_id`

#### `create_backend_test`

#### `get_ui_reports`

#### `get_ui_report_by_id`

#### `get_ui_tests`

#### `run_ui_test`

#### `update_ui_test_schedule`

#### `create_ui_excel_report`

#### `create_ui_test`

#### `cancel_ui_test`


---

## Confluence (`confluence`)
**Categories**: documentation | **Tags**: confluence, wiki, knowledge base, documentation, atlassian
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `space` | str | Yes | Space |
| `api_version` | Literal['Auto', '2', '3'] | No | REST API version used for all Confluence operations.  • **Auto** (default) — automatically selected based on the Hosting setting in the linked credential (Cloud → V3, Server → V2) • **V3** — for Confluence Cloud (*.atlassian.net). Uses ADF for rich text content • **V2** — for Confluence Server / Data Center. Uses plain text and wiki markup  ⚠️ Using the wrong version may cause content formatting issues in pages and comments. |
| `limit` | int | No | Maximum number of pages to retrieve per individual API request. Controls the page size of each call — does not limit the total number of pages retrieved. (Default: 5) |
| `labels` | Optional[str] | No | Filter content retrieval to pages that have specific Confluence labels. Comma-separated list, no spaces around commas. Example: `meeting-notes,documentation,project-alpha` (Optional — leave empty to retrieve all content without label filtering) |
| `max_pages` | int | No | Maximum total number of pages to retrieve across all paginated requests. Prevents excessive data retrieval for large Confluence spaces. (Default: 10) |
| `number_of_retries` | int | No | How many times the toolkit should automatically retry a failed API request before reporting an error. Useful for handling transient network issues or temporary Confluence unavailability. (Default: 2) |
| `min_retry_seconds` | int | No | Minimum number of seconds to wait before attempting a retry after a failure. Acts as the lower bound of the retry backoff interval. (Default: 10) |
| `max_retry_seconds` | int | No | Maximum number of seconds to wait between retry attempts. Acts as the upper bound of the retry backoff interval. Retries will not wait longer than this value regardless of attempt number. (Default: 60) |
| `custom_headers` | Optional[dict] | No | Optional additional HTTP headers to include with every API request. Useful for custom authentication, routing, or proxy requirements. Must be valid JSON format. Example: `{"X-Custom-Header": "value", "X-Tenant-ID": "my-org"}` (Optional — leave empty if not required) |
| `confluence_configuration` | ConfluenceConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `create_page`
> Creates a page in the Confluence space. Represents content in html (storage) or wiki (wiki) formats

#### `create_pages`
> Creates a batch of pages in the Confluence space.

#### `delete_page`
> Deletes a page by its defined page_id or page_title

#### `update_page_by_id`
> Updates an existing Confluence page (using id or title) by replacing its content, title, labels

#### `update_page_by_title`
> Updates an existing Confluence page (using id or title) by replacing its content, title, labels

#### `update_pages`
> Update a batch of pages in the Confluence space.

#### `update_labels`
> Update a batch of pages in the Confluence space.

#### `get_page_tree`
> Gets page tree for the Confluence space

#### `get_pages_with_label`
> Gets pages with specific label in the Confluence space.

#### `list_pages_with_label`
> Lists the pages with specific label in the Confluence space.

#### `read_page_by_id`
> Reads a page by its id in the Confluence space. If id is not available, but there is a title - use get_page_id first.

#### `search_pages`
> Search pages in Confluence by query text in title or page content.

#### `search_by_title`
> Search pages in Confluence by query text in title.

#### `site_search`
> Search for pages in Confluence using site search by query text.

#### `get_page_with_image_descriptions`
> Get a Confluence page and augment any images in it with textual descriptions that include

#### `execute_generic_confluence`
> Generic Confluence Tool for Official Atlassian Confluence REST API to call, searching, creating, updating pages, etc.

#### `get_page_id_by_title`
> Provide page id from search result by title.

#### `get_page_attachments`
> Retrieve all attachments for a Confluence page, including core metadata (with creator, created, updated), comments,

#### `add_file_to_page`
> Upload file from artifact and add to Confluence page content. Images display inline with optional visible caption, other files as attachment links.


---

## OpenAPI (`custom_open_api`)
**Categories**: testing | **Tags**: openapi, swagger
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `spec` | str | No | OpenAPI specification |
| `api_key` | str | No | API key |

### Tools

#### `invoke_rest_api_by_spec`
> Use this tool to invoke external API according to OpenAPI specification.

#### `get_open_api_spec`
> Retrieves the OpenAPI (Swagger) specification for a given API endpoint. This tool helps in obtaining the necessary information to interact with an API using the "Invoke External API" tool.


---

## Figma (`figma`)
**Categories**: other | **Tags**: figma, design, ui/ux, prototyping, collaboration
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `number_of_threads` | Optional[int] | No |  |
| `global_limit` | Optional[int] | No | Global limit |
| `global_regexp` | Optional[str] | No | Global regex pattern |
| `figma_configuration` | FigmaConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_file_nodes`
> Reads a specified file nodes by field key from Figma.

#### `get_file`
> Reads a specified file by field key from Figma.

#### `get_file_versions`
> Retrieves the version history of a specified file from Figma.

#### `get_file_comments`
> Retrieves comments on a specified file from Figma.

#### `post_file_comment`
> Posts a comment to a specific file in Figma.

#### `get_file_images`
> Fetches URLs for server-rendered images from a Figma file based on node IDs.

#### `get_team_projects`
> Retrieves all projects for a specified team ID from Figma.

#### `get_project_files`
> Retrieves all files for a specified project ID from Figma.

#### `analyze_file`
> Comprehensive Figma file analyzer with LLM-powered insights.


---

## GitHub (`github`)
**Categories**: code repositories | **Tags**: github, git, repository, code, version control
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `github_configuration` | GithubConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `repository` | str | Yes | Github repository |
| `active_branch` | Optional[str] | No | Active branch |
| `base_branch` | Optional[str] | No | Github Base branch |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_issues`

#### `get_issue`

#### `comment_on_issue`

#### `list_open_pull_requests`

#### `get_pull_request`

#### `list_pull_request_diffs`

#### `create_pull_request`

#### `create_file`

#### `read_file`

#### `update_file`

#### `delete_file`

#### `list_files_in_main_branch`

#### `list_files_in_bot_branch`
> Lists files in the bot's currently active working branch.

#### `list_branches_in_repo`

#### `set_active_branch`

#### `create_branch`

#### `delete_branch`

#### `get_files_from_directory`

#### `search_issues`

#### `create_issue`

#### `update_issue`

#### `get_commits`
> Retrieves a list of commits from the repository.

#### `get_commit_changes`
> Retrieves the files changed in a specific commit.

#### `get_commits_diff`
> Retrieves the diff between two commits.

#### `apply_git_patch`
> Applies a git patch to the repository by parsing the unified diff format and updating files accordingly.

#### `apply_git_patch_from_file`
> Applies a git patch from a file stored in a specified bucket.

#### `trigger_workflow`
> Triggers a GitHub Actions workflow run manually.

#### `get_workflow_status`
> Gets the status and details of a specific GitHub Actions workflow run.

#### `get_workflow_logs`
> Gets the logs from a GitHub Actions workflow run.

#### `generic_github_api_call`
> Generic method to make API calls to GitHub.

#### `get_me`
> Get details of the authenticated GitHub user.

#### `search_code`
> Search for code in the configured repository using GitHub's code search.

#### `create_issue_on_project`

#### `update_issue_on_project`

#### `list_project_issues`

#### `search_project_issues`


---

## GitLab (`gitlab`)
**Categories**: code repositories | **Tags**: gitlab, git, repository, code, version control
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repository` | str | Yes | GitLab repository |
| `gitlab_configuration` | GitlabConfiguration | Yes |  |
| `branch` | str | No | Main branch |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `create_branch`

#### `delete_branch`

#### `list_branches_in_repo`

#### `list_files`

#### `list_folders`

#### `get_issues`

#### `get_issue`

#### `create_pull_request`

#### `comment_on_issue`

#### `comment_on_pr`

#### `create_file`

#### `read_file`

#### `update_file`

#### `append_file`

#### `delete_file`

#### `set_active_branch`
> Set the active branch in the repository.

#### `get_pr_changes`
> Get all changes from a pull request in git diff format.

#### `create_pr_change_comment`

#### `get_commits`
> Retrieve a list of commits from the repository.


---

## GitLab Org (`gitlab_org`)
**Categories**: code repositories | **Tags**: gitlab, git, repository, code, version control
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gitlab_configuration` | GitlabConfiguration | Yes |  |
| `repositories` | str | No | List of comma separated repositories user plans to interact with. Leave it empty in case you pass it in instruction. |
| `branch` | str | No | Main branch |

### Tools

#### `create_branch`

#### `set_active_branch`

#### `list_branches_in_repo`

#### `get_issues`

#### `get_issue`

#### `create_pull_request`

#### `comment_on_issue`

#### `create_file`

#### `read_file`

#### `update_file`

#### `delete_file`

#### `get_pr_changes`

#### `create_pr_change_comment`

#### `list_files`

#### `list_folders`

#### `append_file`

#### `get_commits`


---

## Google Places (`google_places`)
**Categories**: other | **Tags**: google, places, maps, location, geolocation
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `results_count` | Optional[int] | No | Results number to show |
| `google_places_configuration` | GooglePlacesConfiguration | Yes | Google Places Configuration |

### Tools

#### `places`
> Retrieve places based on a query using Google Places API.

#### `find_near`
> Find places near a specific location using Google Places API.


---

## Jira (`jira`)
**Categories**: project management | **Tags**: jira, atlassian, issue tracking, project management, task management
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `limit` | int | No | Maximum number of issues to retrieve per request. Keep this value low for better performance. (Default: 5) |
| `api_version` | Literal['Auto', '2', '3'] | No | REST API version used for all Jira operations.  • **Auto** (default) — automatically selected based on the Hosting setting in the linked credential (Cloud → V3, Server → V2) • **V3** — required for Jira Cloud (*.atlassian.net). Uses Atlassian Document Format (ADF) for comments and descriptions • **V2** — standard for Jira Server / Data Center (e.g., self-hosted instances). Uses plain text and wiki markup  ⚠️ Using the wrong version for your instance type may cause failures in comments, search, and attachments. |
| `labels` | Optional[str] | No | Specify labels to apply to created or updated Jira entities. Comma-separated list, no spaces around commas. Example: `alita,elitea,automation` (Optional) |
| `custom_headers` | Optional[dict] | No | Optional additional HTTP headers to include with every API request. Useful for custom authentication, routing, or proxy requirements. Must be valid JSON format. Example: `{"X-Custom-Header": "value", "X-Tenant-ID": "my-org"}` (Optional — leave empty if not required) |
| `verify_ssl` | bool | No | Enables SSL certificate verification for all API requests to your Jira instance.  • **Enabled** (recommended) — validates the server's SSL certificate for secure connections • **Disabled** — skips SSL verification. Use only for internal/self-signed certificate environments  ⚠️ Disabling SSL verification is not recommended in production environments. |
| `additional_fields` | Optional[str] | No | Custom Jira field IDs that should be accessible within this toolkit. Use Jira field IDs as they appear in your instance schema. Example: `customfield_10045,customfield_10100` (Optional — leave empty if no custom fields are needed) |
| `jira_configuration` | JiraConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `search_using_jql`
> Search for Jira issues using JQL.

#### `create_issue`
> Create an issue in Jira.

#### `update_issue`
> Update an issue in Jira.

#### `modify_labels`
> Updates labels of an issue in Jira.

#### `list_comments`
> Extract the comments related to specified Jira issue

#### `add_comments`
> Add a comment to a Jira issue.

#### `list_projects`
> List all projects in Jira.

#### `set_issue_status`
> Set new status for the issue in Jira. Used to move ticket through the defined workflow.

#### `get_specific_field_info`
> Get the specific field information from Jira by jira issue key and field name

#### `get_field_with_image_descriptions`
> Get a field from Jira issue and augment any images in it with textual descriptions that include

#### `get_comments_with_image_descriptions`
> Get all comments from Jira issue and augment any images in them with textual descriptions.

#### `get_remote_links`
> Get the remote links from the specified jira issue key

#### `link_issues`
> Link issues functionality for Jira issues. To link test to another issue ( test 'test' story, story 'is tested by test').

#### `get_attachments_content`
> Extract the content of all attachments related to a specified Jira issue key.

#### `add_file_to_issue_description`
> Upload file from artifact and add to issue description. Images/videos inline, others as links.

#### `update_comment_with_file`
> Upload file and add to existing comment. Images/videos inline, others as links.

#### `execute_generic_rq`
> Executes a generic JIRA tool request.


---

## Remote MCP (`mcp`)
**Categories**: other | **Tags**: remote tools, sse, http
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | str | Yes | MCP server HTTP URL |
| `headers` | Optional[Dict[str, str]] | No | HTTP headers for authentication and configuration |
| `client_id` | Optional[str] | No | OAuth Client ID (if applicable) |
| `client_secret` | Optional[SecretStr] *(secret)* | No | OAuth Client Secret (if applicable) |
| `scopes` | Optional[List[str]] | No | OAuth Scopes (if applicable) |
| `timeout` | Union[int, str] | No | Request timeout in seconds (1-3600) |
| `enable_caching` | bool | No | Enable caching of tool schemas and responses |
| `cache_ttl` | Union[int, str] | No | Cache TTL in seconds (60-3600) |
| `ssl_verify` | bool | No | Verify SSL certificates (disable for self-signed certs) |

*Tools are provided at runtime by the connected MCP server — the available tool list depends on the server configuration.*


---

## Memory (`memory`)
**Categories**: other | **Tags**: long-term memory, langgraph store
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `namespace` | str | Yes | Memory namespace for organizing memories |
| `pgvector_configuration` | PgVectorConfiguration | Yes |  |

### Tools

#### `manage_memory`
> Store information in long-term memory. Use this to remember important facts, user preferences, or any information that should persist across conversations.

#### `search_memory`
> Search through stored memories using natural language. Returns memories that are semantically similar to the query.

#### `get_memory`
> Retrieve a specific memory by its key.

#### `delete_memory`
> Delete a specific memory by its key.


---

## OpenAPI (`openapi`)
**Categories**: integrations | **Tags**: api, openapi, swagger
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `openapi_configuration` | OpenApiConfiguration | Yes | OpenAPI credentials configuration |
| `base_url` | Optional[str] | No | Optional base URL override (absolute, starting with http:// or https://). Use this when your OpenAPI spec has no `servers` entry, or when `servers[0].url` is not absolute (e.g. '/api/v3'). Example: 'https://petstore3.swagger.io'. |
| `spec` | str | Yes | OpenAPI specification (URL or raw JSON/YAML text). Used to generate per-operation tools (one tool per operationId). |

*Tools are generated dynamically at runtime from the provided OpenAPI spec — one tool per `operationId`.*


---

## Postman (`postman`)
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `postman_configuration` | Optional[PostmanConfiguration] | No | Postman Configuration |
| `collection_id` | str | Yes | Default collection ID |
| `environment_config` | dict | No | JSON configuration for request execution (auth headers, project IDs, base URLs, etc.) |

### Tools

#### `get_collections`
> Get all Postman collections accessible to the user

#### `get_collection`
> Get a specific Postman collection in flattened format with path-based structure

#### `get_folder`
> Get a specific folder in flattened format with path-based structure

#### `get_request_by_path`
> Get a specific request by path

#### `get_request_by_id`
> Get a specific request by ID

#### `get_request_script`
> Get the test or pre-request script content for a specific request

#### `search_requests`
> Search for requests across the collection

#### `analyze`
> Analyze collection, folder, or request for API quality, best practices, and issues

#### `execute_request`
> Execute a Postman request with environment variables and custom configuration

#### `update_collection_description`
> Update collection description

#### `update_collection_variables`
> Update collection variables

#### `update_collection_auth`
> Update collection authentication settings

#### `delete_collection`
> Delete a collection permanently

#### `duplicate_collection`
> Create a copy of an existing collection

#### `create_folder`
> Create a new folder in a collection

#### `update_folder`
> Update folder properties (name, description, auth)

#### `delete_folder`
> Delete a folder and all its contents permanently

#### `move_folder`
> Move a folder to a different location within the collection

#### `create_request`
> Create a new API request in a folder

#### `update_request_name`
> Update request name

#### `update_request_method`
> Update request HTTP method

#### `update_request_url`
> Update request URL

#### `update_request_description`
> Update request description

#### `update_request_headers`
> Update request headers

#### `update_request_body`
> Update request body

#### `update_request_auth`
> Update request authentication

#### `update_request_tests`
> Update request test scripts

#### `update_request_pre_script`
> Update request pre-request scripts

#### `delete_request`
> Delete an API request permanently

#### `duplicate_request`
> Create a copy of an existing API request

#### `move_request`
> Move an API request to a different folder


---

## PPTX (`pptx`)
**Categories**: office | **Tags**: presentation, office automation, document
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bucket_name` | str | Yes | Bucket name where PPTX files are stored |

### Tools

#### `fill_template`
> Fill a PPTX template with content based on the provided description.

#### `translate_presentation`
> Translate text in a PowerPoint presentation to another language


---

## QTest (`qtest`)
**Categories**: test management | **Tags**: quality assurance, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `qtest_configuration` | QtestConfiguration | Yes |  |
| `qtest_project_id` | int | Yes | QTest project id |
| `no_of_tests_shown_in_dql_search` | Optional[int] | No | Max number of items returned by dql search |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `search_by_dql`
> Search test cases in qTest using Data Query Language (DQL).

#### `create_test_cases`
> Create a test case in qTest.

#### `update_test_case`
> Update, change or replace data in the test case.

#### `add_file_to_test_case`
> Upload file from artifact storage to QTest test case or specific test step.

#### `find_test_case_by_id`

#### `delete_test_case`
> Delete test case by its qtest id. Id should be in format 3534653120.

#### `link_tests_to_jira_requirement`
> Link test cases to external Jira requirement. Provide Jira issue ID (e.g., PLAN-128) and list of test case IDs in format '["TC-123", "TC-234"]'

#### `link_tests_to_qtest_requirement`
> Link test cases to internal QTest requirement. Provide QTest requirement ID (e.g., RQ-15) and list of test case IDs in format '["TC-123", "TC-234"]'

#### `get_modules`
> :param int project_id: ID of the project (required)

#### `get_all_test_cases_fields_for_project`
> Get information about available test case fields and their valid values for the project. Shows which property values are allowed (e.g., Status: 'New', 'In Progress', 'Completed') based on the project configuration. Use force_refresh=true if project configuration has changed.

#### `find_test_cases_by_requirement_id`
> Find all test cases linked to a QTest requirement.

#### `find_requirements_by_test_case_id`
> Find all requirements linked to a test case (direct link: test-case 'covers' requirements).

#### `find_test_runs_by_test_case_id`
> Find all test runs associated with a test case.

#### `find_defects_by_test_run_id`
> Find all defects associated with a test run.

#### `search_entities_by_dql`

#### `find_entity_by_id`
> Find any QTest entity by its ID.


---

## Rally (`rally`)
**Categories**: project management | **Tags**: agile management, test management, scrum, kanban
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rally_configuration` | RallyConfiguration | Yes | Rally configuration |
| `workspace` | Optional[str] | No | Rally workspace |
| `project` | Optional[str] | No | Rally project |

### Tools

#### `get_types`
> Get available entity types from Rally.

#### `get_entities`
> Get user stories from Rally.

#### `get_project`
> Get a project from Rally by name.

#### `get_workspace`
> Get a workspace from Rally by name.

#### `get_user`
> Get a user from Rally by username.

#### `get_context`
> Get a user from Rally by username.

#### `create_artifact`
> Create an artifact in Rally.

#### `update_artifact`
> Update an artifact in Rally.


---

## Report Portal (`report_portal`)
**Categories**: testing | **Tags**: test reporting, test automation
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `report_portal_configuration` | ReportPortalConfiguration | Yes | Report Portal Configuration |

### Tools

#### `get_extended_launch_data_as_raw`
> Get Launch details as a raw

#### `get_extended_launch_data`
> Use the exported data from a specific launch to generate a comprehensive test report for management.

#### `get_launch_details`
> Retrieve detailed information about a launch to perform a root cause analysis of failures.

#### `get_all_launches`
> Analyze the data from all launches to track the progress of testing activities over time.

#### `find_test_item_by_id`
> Fetch specific test items to perform detailed analysis on individual test cases. It can evaluate

#### `get_test_items_for_launch`
> Compile all test items from a launch to create a test execution summary.

#### `get_logs_for_test_items`
> Process the logs for test items to assist in automated debugging.

#### `get_user_information`
> Use user information to personalize dashboards and reports. It can also analyze user activity to optimize

#### `get_dashboard_data`
> Analyze dashboard data to create executive summaries that highlight key performance indicators (KPIs),


---

## Salesforce (`salesforce`)
**Categories**: other | **Tags**: customer relationship management, cloud computing, marketing automation, salesforce
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_version` | str | No | Salesforce API Version |
| `salesforce_configuration` | SalesforceConfiguration | Yes | Salesforce Configuration |

### Tools

#### `create_case`
> Create a new Case

#### `create_lead`
> Create a new Lead

#### `search_salesforce`
> Search Salesforce with SOQL

#### `update_case`
> Update a Case

#### `update_lead`
> Update a Lead

#### `execute_generic_rq`
> Execute a generic Salesforce API request.


---

## ServiceNow (`service_now`)
**Categories**: other | **Tags**: incident management, problem management, change management, service catalog
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response_fields` | Optional[str] | No | Response fields |
| `servicenow_configuration` | ServiceNowConfiguration | Yes | ServiceNow Configuration |

### Tools

#### `get_incidents`
> Retrieves incidents from the ServiceNow database based on the provided filters.

#### `create_incident`
> Creates a new incident on the ServiceNow database.

#### `update_incident`
> Updates an existing incident on the ServiceNow database per the provided sys_id and


---

## Sharepoint (`sharepoint`)
**Categories**: office | **Tags**: microsoft, cloud storage, team collaboration, content management
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sharepoint_configuration` | SharepointConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `read_list`
> Reads a specified List in sharepoint site. Number of list items is limited by limit (default is 1000).

#### `get_lists`
> Returns all SharePoint lists available on the site with their titles, IDs, and descriptions.

#### `get_list_columns`
> Get all columns (fields) in a SharePoint list with their metadata.

#### `create_list_item`
> Create a new item in a SharePoint list.

#### `get_files_list`
> Lists all files including files from subfolders.

#### `read_document`
> Reads file located at the specified server-relative path.

#### `upload_file`
> Upload file to SharePoint document library.

#### `onenote_get_notebooks`
> List all OneNote notebooks in this SharePoint site.

#### `onenote_get_sections`
> List all sections in a specific OneNote notebook on this site.

#### `onenote_get_pages`
> List pages in a OneNote section on this site.

#### `onenote_get_page_content`
> Retrieve the raw HTML content of a OneNote page on this site.

#### `onenote_read_page`
> Read and parse a OneNote page into human-readable plain text.

#### `onenote_read_page_items`
> Read and parse a OneNote page into a structured collection of typed items.

#### `onenote_create_notebook`
> Create a new OneNote notebook in this SharePoint site.

#### `onenote_create_section`
> Create a new section in a OneNote notebook on this site.

#### `onenote_create_page`
> Create a new OneNote page in a section on this site from raw HTML.

#### `onenote_update_page`
> Update a OneNote page using Graph API PATCH commands.

#### `onenote_replace_page_content`
> Replace the entire body of a OneNote page with new HTML content.

#### `onenote_delete_page`
> Permanently delete a OneNote page on this site.

#### `onenote_list_attachments`
> List all file attachments on a OneNote page.

#### `onenote_read_attachment`
> Download and parse a single file attachment from a OneNote page.


---

## Slack (`slack`)
**Categories**: communication | **Tags**: slack, chat, messaging, collaboration
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `slack_configuration` | SlackConfiguration | No | Slack configuration |

### Tools

#### `send_message`

#### `read_messages`

#### `create_slack_channel`

#### `list_channel_users`

#### `list_workspace_users`

#### `invite_to_conversation`

#### `list_workspace_conversations`


---

## Sonar (`sonar`)
**Categories**: development | **Tags**: code quality, code security, code coverage, quality, sonarqube
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sonar_project_name` | str | Yes | Project name of the desired repository |
| `sonar_configuration` | SonarConfiguration | Yes | Sonar Configuration |

### Tools

#### `get_sonar_data`
> SonarQube Tool for interacting with the SonarQube REST API.


---

## SQL (`sql`)
**Categories**: development | **Tags**: sql, data management, data analysis
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dialect` | Literal[tuple(supported_dialects)] | No | Database dialect (mysql or postgres) |
| `database_name` | str | Yes | Database name |
| `sql_configuration` | SqlConfiguration | Yes | SQL Configuration |

### Tools

#### `execute_sql`
> Executes the provided SQL query on the configured database.

#### `list_tables_and_columns`
> Lists all tables and their columns in the configured database.


---

## TestIO (`testio`)
**Categories**: testing | **Tags**: test automation, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `testio_configuration` | TestIOConfiguration | Yes | TestIO Configuration |

### Tools

#### `list_products`
> Retrieve a list of all available products with optional filtering by product IDs.

#### `get_product`
> Retrieve detailed information about a specific product by its ID.

#### `list_features`
> Retrieve a comprehensive list of features across all products with optional filtering by feature IDs.

#### `get_feature`
> Retrieve detailed information about a specific feature by its ID.

#### `list_user_stories`
> Retrieve a list of user stories with optional filtering by story IDs.

#### `get_user_story`
> Retrieve detailed information about a specific user story by its ID.

#### `list_exploratory_tests`
> Retrieve a list of exploratory tests with optional filtering by product ID.

#### `get_exploratory_test`
> Retrieve detailed information about a specific exploratory test by its ID.

#### `create_exploratory_test`
> Create a new exploratory test with specified parameters including product, section, test type, devices, etc.

#### `list_test_cases`
> Retrieve a list of test cases for a specific product with optional filtering by section.

#### `get_test_case`
> Retrieve detailed information about a specific test case by its ID and product ID.

#### `confirm_bug_fix`
> Confirm the status of a bug fix with optional comments.

#### `get_test_cases_for_test`
> Retrieve detailed information about test cases for a particular launch (test)

#### `get_test_cases_statuses_for_test`
> Fetch information regarding statuses of executed test cases within a particular launch (test),

#### `list_bugs_for_test_with_filter`
> Retrieve detailed information about bugs associated with test cases


---

## Testrail (`testrail`)
**Categories**: test management | **Tags**: quality assurance, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `testrail_configuration` | Optional[TestRailConfiguration] | No |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_case`
> Extracts information about single test case from Testrail

#### `get_cases`
> Extracts a list of test cases in the specified format: `json`, `csv`, or `markdown`.

#### `get_cases_by_filter`
> Extracts test cases from a specified project based on given case attributes.

#### `add_case`
> Adds new test case into Testrail per defined parameters.

#### `add_cases`
> Adds new test cases into Testrail per defined parameters.

#### `update_case`
> Updates an existing test case. Partial updates are supported.

#### `delete_case`
> Deletes an existing test case.

#### `add_file_to_case`
> Upload file from artifact and attach to TestRail test case.

#### `get_suites`
> Extracts a list of test suites for a given project from Testrail


---

## XRAY cloud (`xray_cloud`)
**Categories**: test management | **Tags**: test automation, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `limit` | Optional[int] | No | Limit |
| `xray_configuration` | XrayConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_tests`
> get all tests

#### `create_test`
> Create new test in XRAY per defined XRAY graphql mutation

#### `create_tests`
> Create new tests in XRAY per defined XRAY graphql mutations

#### `execute_graphql`
> Executes custom graphql query or mutation

#### `add_attachment_to_test_step`
> Add an attachment to an existing test step using GraphQL mutation.

#### `get_test_step_attachments`
> Get attachments for a test or specific test step.


---

## Zephyr Enterprise (`zephyr_enterprise`)
**Categories**: test management | **Tags**: test automation, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `zephyr_configuration` | ZephyrEnterpriseConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_test_case`
> Retrieve test case data by id.

#### `search_zql`
> Retrieve Zephyr entities by zql.

#### `create_testcase`
> Creates test case per given test case properties as JSON.

#### `add_steps`
> Adds steps to the last test case version.

#### `get_testcases_by_zql`
> Retrieve testcases by zql.


---

## Zephyr Essential (`zephyr_essential`)
**Categories**: test management | **Tags**: test automation, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `zephyr_essential_configuration` | ZephyrEssentialConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `list_test_cases`
> List test cases with optional filters.

#### `create_test_case`
> Create a new test case.

#### `get_test_case`
> Retrieve details of a specific test case.

#### `update_test_case`
> Update an existing test case.

#### `get_test_case_links`
> Retrieve links associated with a test case.

#### `create_test_case_issue_link`
> Create an issue link for a test case.

#### `create_test_case_web_link`
> Create a web link for a test case.

#### `list_test_case_versions`
> List versions of a test case.

#### `get_test_case_version`
> Retrieve a specific version of a test case.

#### `get_test_case_test_script`
> Retrieve the test script of a test case.

#### `create_test_case_test_script`
> Create a test script for a test case.

#### `get_test_case_test_steps`
> List test steps of a test case.

#### `create_test_case_test_steps`
> Create test steps for a test case.

#### `list_test_cycles`
> List test cycles with optional filters.

#### `create_test_cycle`
> Create a new test cycle.

#### `get_test_cycle`
> Retrieve details of a specific test cycle.

#### `update_test_cycle`
> Update an existing test cycle.

#### `get_test_cycle_links`
> Retrieve links associated with a test cycle.

#### `create_test_cycle_issue_link`
> Create an issue link for a test cycle.

#### `create_test_cycle_web_link`
> Create a web link for a test cycle.

#### `list_test_executions`
> List test executions with optional filters.

#### `create_test_execution`
> Create a new test execution.

#### `get_test_execution`
> Retrieve details of a specific test execution.

#### `update_test_execution`
> Update an existing test execution.

#### `get_test_execution_test_steps`
> List test steps of a test execution.

#### `update_test_execution_test_steps`
> Update test steps of a test execution.

#### `sync_test_execution_script`
> Sync the test execution script.

#### `list_test_execution_links`
> List links associated with a test execution.

#### `create_test_execution_issue_link`
> Create an issue link for a test execution.

#### `list_projects`
> List all projects.

#### `get_project`
> Retrieve details of a specific project.

#### `list_folders`
> List folders with optional filters.

#### `create_folder`
> Create a new folder.

#### `get_folder`
> Retrieve details of a specific folder.

#### `find_folder_by_name`
> Find a folder by its name, ignoring case.

#### `delete_link`
> Delete a specific link.

#### `get_issue_link_test_cases`
> Retrieve test cases linked to an issue.

#### `get_issue_link_test_cycles`
> Retrieve test cycles linked to an issue.

#### `get_issue_link_test_plans`
> Retrieve test plans linked to an issue.

#### `get_issue_link_test_executions`
> Retrieve test executions linked to an issue.

#### `create_custom_executions`
> Create custom executions.

#### `create_cucumber_executions`
> Create cucumber executions.

#### `create_junit_executions`
> Create JUnit executions.

#### `retrieve_bdd_test_cases`
> Retrieve BDD test cases.

#### `healthcheck`
> Perform a health check on the API.


---

## Zephyr Scale (`zephyr_scale`)
**Categories**: test management | **Tags**: test automation, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_results` | int | No | Results count to show |
| `zephyr_configuration` | ZephyrConfiguration | Yes |  |
| `pgvector_configuration` | Optional[PgVectorConfiguration] | No |  |
| `embedding_model` | Optional[str] | No |  |

### Tools

#### `get_tests`
> Retrieves all test cases. Query parameters can be used to filter the results.

#### `get_test`
> Returns a test case for the given key

#### `get_test_steps`
> Returns the test steps for the given test case. Provides a paged response.

#### `create_test_case`
> Creates a test case. Fields priorityName and statusName will be set to default values if not informed.

#### `create_test_cases`
> Creates a bunch of test cases

#### `add_test_steps`
> Assigns a series of test steps to a test case.

#### `update_test_steps`
> Updates specific test steps in a test case.

#### `get_folders`
> Retrieves all folders. Query parameters can be used to filter the results: maxResults, startAt, projectKey, folderType

#### `update_test_case`
> Updates an existing test case.

#### `get_links`
> Returns links for a test case with specified key

#### `create_issue_links`
> Creates a link between a test case and a Jira issue

#### `create_web_links`
> Creates a link between a test case and a generic URL

#### `get_versions`
> Returns all test case versions for a test case with specified key. Response is ordered by most recent first.

#### `get_version`
> Retrieves a specific version of a test case

#### `get_test_script`
> Returns the test script for the given test case

#### `create_test_script`
> Creates or updates the test script for a test case

#### `search_test_cases`
> Searches for test cases using custom search API.

#### `get_tests_recursive`
> Retrieves all test cases recursively from a folder and all its subfolders.

#### `get_tests_by_folder_name`
> Retrieves all test cases from folders matching the specified name.

#### `get_tests_by_folder_path`
> Retrieves all test cases from a folder specified by its path.


---

## Zephyr Squad (`zephyr_squad`)
**Categories**: test management | **Tags**: test automation, test case management, test planning
*\[Info extracted from source; run with toolkit deps installed for full detail\]*

### Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `account_id` | str | Yes | AccountID for the user that is going to be authenticating |
| `access_key` | SecretStr *(secret)* | Yes | Generated access key |
| `secret_key` | SecretStr *(secret)* | Yes | Generated secret key |

### Tools

#### `get_test_step`
> Retrieve details for a specific test step in a Jira test case.

#### `update_test_step`
> Update the content or a specific test step in a Jira test case.

#### `delete_test_step`
> Remove a specific test step from a Jira test case.

#### `create_new_test_step`
> Add a new test step to a Jira test case.

#### `get_all_test_steps`
> List all test steps associated with a Jira test case.

#### `get_all_test_step_statuses`
> Retrieve all possible statuses for test steps in Jira.

#### `get_bdd_content`
> Retrieve BDD (Gherkin) content of an issue (feature or scenario).

#### `update_bdd_content`
> Replace BDD (Gherkin) content of an issue (feature or scenario).

#### `delete_bdd_content`
> Remove BDD (Gherkin) content of an issue (feature or scenario).

#### `create_new_cycle`
> Creates a Cycle from a JSON representation. If no VersionId is passed in the request, it will be defaulted to an unscheduled version

#### `create_folder`
> Creates a Folder from a JSON representation. Folder names within a cycle needs to be unique.

#### `add_test_to_cycle`
> Adds Tests(s) to a Cycle.

#### `add_test_to_folder`
> Adds Tests(s) to a Folder.

#### `create_execution`
> Creates an execution from a JSON representation.

#### `get_execution`
> Retrieves Execution and ExecutionStatus by ExecutionId


---
