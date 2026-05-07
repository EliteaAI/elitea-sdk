#!/usr/bin/env python3
"""
Execute a pipeline predict request and return raw API response.

Simplified runner that:
- Uses TEST_* prefixed environment variables
- Only executes the predict POST request
- Returns raw API response without processing
- No HITL handling, no result parsing, no test validation

Usage:
    python run_pipeline.py <pipeline_id> [options]
    python run_pipeline.py --name "Pipeline Name" [options]

Examples:
    # Execute by ID
    python run_pipeline.py 123

    # Execute by name
    python run_pipeline.py --name "My Pipeline"

    # With custom input
    python run_pipeline.py 123 --input '{"key": "value"}'

    # JSON output
    python run_pipeline.py 123 --json
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_env_var(name: str, required: bool = True) -> Optional[str]:
    """Get environment variable with TEST_ prefix."""
    value = os.getenv(name)
    if required and not value:
        print(f"Error: {name} environment variable not set", file=sys.stderr)
        sys.exit(1)
    return value


def get_auth_headers(include_content_type: bool = False) -> dict:
    """Get authentication headers."""
    api_key = get_env_var("TEST_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def get_pipeline_by_id(base_url: str, project_id: int, pipeline_id: int, headers: dict) -> Optional[dict]:
    """Get pipeline details by ID."""
    url = f"{base_url}/api/v2/elitea_core/application/prompt_lib/{project_id}/{pipeline_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


def get_pipeline_by_name(base_url: str, project_id: int, name: str, headers: dict) -> Optional[dict]:
    """Get pipeline details by name (exact match)."""
    url = f"{base_url}/api/v2/elitea_core/applications/prompt_lib/{project_id}?limit=500"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None

    data = response.json()
    for pipeline in data.get("rows", []):
        if pipeline.get("name") == name:
            return get_pipeline_by_id(base_url, project_id, pipeline["id"], headers)
    
    return None


def predict_pipeline(
    base_url: str,
    project_id: int,
    pipeline: dict,
    input_message: str = "execute",
    timeout: int = 120,
    verbose: bool = False,
) -> dict:
    """Execute pipeline predict request and return raw response.
    
    Args:
        base_url: Backend URL
        project_id: Project ID
        pipeline: Pipeline dict with id and versions
        input_message: Input message for the pipeline
        timeout: Request timeout in seconds
        verbose: Print debug information
        
    Returns:
        dict with keys:
            - success: bool
            - status_code: int
            - response: dict or str (raw API response)
            - execution_time: float
            - error: str (if failed)
    """
    start_time = time.time()
    headers = get_auth_headers(include_content_type=True)

    pipeline_id = pipeline["id"]
    pipeline_name = pipeline.get("name", f"Pipeline {pipeline_id}")

    # Get version ID from TEST_VERSION env var (required)
    test_version = os.getenv("TEST_VERSION")
    if not test_version:
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "execution_time": 0.0,
            "error": "TEST_VERSION environment variable not set"
        }
    
    try:
        version_id = int(test_version)
    except ValueError:
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "execution_time": 0.0,
            "error": f"Invalid TEST_VERSION value: {test_version} (must be an integer)"
        }

    if verbose:
        print(f"Executing: {pipeline_name} (ID: {pipeline_id}, Version: {version_id})", file=sys.stderr)

    # POST to v2 predict API
    predict_url = f"{base_url}/api/v2/elitea_core/predict/prompt_lib/{project_id}/{version_id}"
    payload = {
        "chat_history": [],
        "user_input": input_message
    }

    if verbose:
        print(f"POST {predict_url}", file=sys.stderr)
        print(f"Payload: {json.dumps(payload)}", file=sys.stderr)

    try:
        response = requests.post(
            predict_url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
    except requests.exceptions.Timeout:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "execution_time": execution_time,
            "error": f"Request timed out after {timeout}s"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "execution_time": execution_time,
            "error": f"Request failed: {e}"
        }

    execution_time = time.time() - start_time

    if verbose:
        print(f"Response: {response.status_code}", file=sys.stderr)

    # Return raw response
    if response.status_code in (200, 201):
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = response.text
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response": response_data,
            "execution_time": execution_time,
            "error": None
        }
    else:
        return {
            "success": False,
            "status_code": response.status_code,
            "response": response.text,
            "execution_time": execution_time,
            "error": f"HTTP {response.status_code}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Execute pipeline predict request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("pipeline_id", nargs="?", type=int, help="Pipeline ID to execute")
    parser.add_argument("--name", "-n", type=str, help="Pipeline name (alternative to ID)")
    parser.add_argument("--url", default=None, help="Backend URL (overrides TEST_DEPLOYMENT_URL)")
    parser.add_argument("--api-key", default=None, help="API key (overrides TEST_API_KEY)")
    parser.add_argument("--project-id", type=int, default=None, help="Project ID (overrides TEST_PROJECT_ID)")
    parser.add_argument("--input", "-i", type=str, default="execute", help="Input message for pipeline")
    parser.add_argument("--timeout", "-t", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate arguments
    if not args.pipeline_id and not args.name:
        parser.error("Either pipeline_id or --name is required")

    # Override env vars if provided via CLI
    if args.url:
        os.environ["TEST_DEPLOYMENT_URL"] = args.url
    if args.api_key:
        os.environ["TEST_API_KEY"] = args.api_key
    if args.project_id:
        os.environ["TEST_PROJECT_ID"] = str(args.project_id)

    # Load configuration
    base_url = get_env_var("TEST_DEPLOYMENT_URL").rstrip('/')
    project_id = int(get_env_var("TEST_PROJECT_ID"))
    headers = get_auth_headers()

    # Get pipeline
    if args.pipeline_id:
        pipeline = get_pipeline_by_id(base_url, project_id, args.pipeline_id, headers)
    else:
        pipeline = get_pipeline_by_name(base_url, project_id, args.name, headers)

    if not pipeline:
        error_msg = f"Pipeline not found: {args.pipeline_id or args.name}"
        if args.json:
            print(json.dumps({"success": False, "error": error_msg}))
        else:
            print(f"Error: {error_msg}")
        sys.exit(1)

    # Execute prediction
    result = predict_pipeline(
        base_url=base_url,
        project_id=project_id,
        pipeline=pipeline,
        input_message=args.input,
        timeout=args.timeout,
        verbose=args.verbose
    )

    # Output results
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'=' * 60}")
        print(f"Pipeline: {pipeline.get('name', 'Unknown')} (ID: {pipeline['id']})")
        print(f"Status Code: {result['status_code']}")
        print(f"Success: {result['success']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")

        if result['error']:
            print(f"Error: {result['error']}")

        print(f"\nRaw Response:")
        if isinstance(result['response'], dict):
            print(json.dumps(result['response'], indent=2, default=str))
        else:
            print(result['response'])

        print(f"{'=' * 60}")

    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
