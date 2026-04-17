# Reporting Results to ReportPortal

Results from any pytest run can be uploaded to ReportPortal by adding the `--reportportal` flag. Without this flag the plugin is dormant — no data is sent even if credentials are present.

---

## 1. Install the Reporting Dependencies

```bash
pip install -e '.[reporting]'
# installs: pytest-reportportal>=5.3, pytest-dotenv
```

---

## 2. Configure Credentials

Add the following to the project `.env` file (already included in `.env.example`):

```bash
RP_ENDPOINT=https://<your-rp-host>/api/receiver
RP_PROJECT=<project-uuid-or-name>
RP_API_KEY=<rp_api_key_token>
RP_LAUNCH=Alita SDK Loader Tests    # display name for the launch in RP
```

`conftest.py` automatically reads these at session start and injects them as pytest-reportportal ini options:

| Env var | pytest-reportportal ini key |
|---|---|
| `RP_ENDPOINT` | `rp_endpoint` |
| `RP_PROJECT` | `rp_project` |
| `RP_LAUNCH` | `rp_launch` |
| `RP_API_KEY` | `rp_uuid` |

`pyproject.toml` already sets `env_files = [".env"]` so the `.env` file is loaded automatically by the `pytest-dotenv` plugin.

---

## 3. Run Tests with RP Reporting

```bash
# Report all document loader tests
python -m pytest tests/runtime/langchain/document_loaders/ --reportportal -v

# Report a single loader
python -m pytest tests/runtime/langchain/document_loaders/test_alita_text_loader.py --reportportal -v

# Report a filtered subset (tags)
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_csv" --reportportal -v

# Report all unit tests
python -m pytest tests/runtime/ --reportportal -v
```

Each run creates a new **Launch** in ReportPortal using the name from `RP_LAUNCH`.

---

## 4. Run Without ReportPortal (local / fast iteration)

Simply omit `--reportportal`:

```bash
python -m pytest tests/runtime/langchain/document_loaders/ -v
```

To explicitly suppress the plugin when env vars are set:

```bash
python -m pytest tests/runtime/langchain/document_loaders/ -p no:reportportal -v
```
