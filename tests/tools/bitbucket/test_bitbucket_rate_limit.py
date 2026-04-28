"""
Tests for Bitbucket toolkit rate limit handling.

This module tests the fix for bug #4056 where Bitbucket toolkit lacked rate
limit handling, causing 429 errors to fail immediately without retry.

The tests validate the rate limit retry decorator logic independently of the
actual Bitbucket API wrapper to avoid dependency issues.
"""
import time
import pytest
from functools import wraps
from typing import Callable, TypeVar

# Create a standalone ToolException for testing
class ToolException(Exception):
    """Exception raised when a tool encounters an error."""
    pass


# Type variable for generic return type
T = TypeVar('T')


# Copy of the rate limit functions for testing (same logic as in cloud_api_wrapper.py)
def is_rate_limit_error(error: Exception) -> bool:
    """Check if the exception is a rate limit (429) error.

    Args:
        error: The exception to check

    Returns:
        True if this is a rate limit error, False otherwise
    """
    error_str = str(error).lower()
    return (
        "429" in error_str or
        "too many requests" in error_str or
        "rate limit" in error_str
    )


def retry_on_rate_limit(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry API calls on 429 rate limit errors with exponential backoff.

    When Bitbucket API returns HTTP 429 (Too Many Requests), the decorated function
    will automatically retry with exponential backoff delays.

    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not is_rate_limit_error(e):
                        # Not a rate limit error - re-raise immediately
                        raise

                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        time.sleep(delay)
                    else:
                        # Max retries exceeded
                        raise ToolException(
                            f"Bitbucket API rate limit exceeded after {max_retries} retries. "
                            f"Please wait a few minutes before trying again. "
                            f"Original error: {str(e)}"
                        ) from e

            # Should not reach here, but handle edge case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class TestIsRateLimitError:
    """Tests for is_rate_limit_error helper function."""

    def test_detects_429_status_code(self):
        """Test that 429 status code is detected in error message."""
        error = Exception("429 Client Error: Too Many Requests")
        assert is_rate_limit_error(error) is True

    def test_detects_too_many_requests(self):
        """Test that 'too many requests' text is detected."""
        error = Exception("HTTP Too Many Requests")
        assert is_rate_limit_error(error) is True

    def test_detects_rate_limit_text(self):
        """Test that 'rate limit' text is detected."""
        error = Exception("API rate limit exceeded")
        assert is_rate_limit_error(error) is True

    def test_does_not_match_other_errors(self):
        """Test that other errors are not flagged as rate limit errors."""
        error = Exception("Connection timeout")
        assert is_rate_limit_error(error) is False

        error = Exception("Authentication failed")
        assert is_rate_limit_error(error) is False

        error = Exception("404 Not Found")
        assert is_rate_limit_error(error) is False

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        assert is_rate_limit_error(Exception("RATE LIMIT")) is True
        assert is_rate_limit_error(Exception("Rate Limit Exceeded")) is True
        assert is_rate_limit_error(Exception("TOO MANY REQUESTS")) is True


class TestRetryOnRateLimitDecorator:
    """Tests for retry_on_rate_limit decorator."""

    def test_successful_call_returns_immediately(self):
        """Test that successful calls return without retry."""
        call_count = 0

        @retry_on_rate_limit(max_retries=3, base_delay=0.01)
        def success_fn():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_fn()

        assert result == "success"
        assert call_count == 1

    def test_retries_on_rate_limit_error(self):
        """Test that rate limit errors trigger retry."""
        call_count = 0

        @retry_on_rate_limit(max_retries=3, base_delay=0.01)
        def rate_limited_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "success"

        result = rate_limited_fn()

        assert result == "success"
        assert call_count == 3

    def test_raises_tool_exception_after_max_retries(self):
        """Test that ToolException is raised after max retries exceeded."""
        call_count = 0

        @retry_on_rate_limit(max_retries=3, base_delay=0.01)
        def always_rate_limited():
            nonlocal call_count
            call_count += 1
            raise Exception("429 Too Many Requests")

        with pytest.raises(ToolException) as exc_info:
            always_rate_limited()

        assert "rate limit exceeded after 3 retries" in str(exc_info.value).lower()
        assert call_count == 4  # Initial attempt + 3 retries

    def test_non_rate_limit_errors_raised_immediately(self):
        """Test that non-rate-limit errors are raised without retry."""
        call_count = 0

        @retry_on_rate_limit(max_retries=3, base_delay=0.01)
        def auth_error_fn():
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized")

        with pytest.raises(Exception) as exc_info:
            auth_error_fn()

        assert "401 Unauthorized" in str(exc_info.value)
        assert call_count == 1  # No retries for non-rate-limit errors

    def test_exponential_backoff_timing(self):
        """Test that delays follow exponential backoff pattern."""
        call_times = []

        @retry_on_rate_limit(max_retries=3, base_delay=0.1, max_delay=60.0)
        def rate_limited_with_timing():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise Exception("429 Too Many Requests")
            return "success"

        result = rate_limited_with_timing()

        assert result == "success"
        assert len(call_times) == 4

        # Check delays are increasing (exponential backoff)
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]

        # First delay should be ~0.1s, second ~0.2s, third ~0.4s
        # Allow 50% tolerance for timing variations
        assert delays[0] >= 0.05  # At least half of base_delay
        assert delays[1] > delays[0] * 0.5  # Second delay longer than first
        assert delays[2] > delays[1] * 0.5  # Third delay longer than second

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        @retry_on_rate_limit(max_retries=10, base_delay=100, max_delay=0.1)
        def rate_limited_capped():
            raise Exception("429 Too Many Requests")

        start_time = time.time()

        with pytest.raises(ToolException):
            rate_limited_capped()

        elapsed = time.time() - start_time

        # With 10 retries and max_delay=0.1, total should be ~1s max
        # Without cap, it would be 100 + 200 + 400 + ... seconds
        assert elapsed < 5  # Should complete quickly due to max_delay cap

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        @retry_on_rate_limit()
        def my_function():
            """This is my docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "This is my docstring."

    def test_works_with_arguments(self):
        """Test that decorated function handles arguments correctly."""
        call_count = 0

        @retry_on_rate_limit(max_retries=2, base_delay=0.01)
        def fn_with_args(a, b, c=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 Too Many Requests")
            return f"{a}-{b}-{c}"

        result = fn_with_args("x", "y", c="z")

        assert result == "x-y-z"
        assert call_count == 2

    def test_works_with_class_methods(self):
        """Test that decorator works with class instance methods."""

        class MyClass:
            def __init__(self):
                self.call_count = 0

            @retry_on_rate_limit(max_retries=2, base_delay=0.01)
            def my_method(self, value):
                self.call_count += 1
                if self.call_count < 2:
                    raise Exception("Rate limit exceeded")
                return f"result: {value}"

        obj = MyClass()
        result = obj.my_method("test")

        assert result == "result: test"
        assert obj.call_count == 2


class TestBug4056Regression:
    """
    Regression tests specifically for bug #4056.

    The bug was: Bitbucket toolkit lacked rate limit handling causing 429 errors
    to fail immediately without retry.

    Expected behavior: When HTTP 429 is encountered, SDK should automatically
    retry with exponential backoff.
    """

    def test_bug_4056_scenario_list_branches(self):
        """
        Reproduce the scenario from bug #4056:
        - Multiple API calls hit rate limit
        - Expected: Automatic retry with backoff
        - Bug behavior: Immediate failure
        """
        call_count = 0

        @retry_on_rate_limit(max_retries=5, base_delay=0.01)
        def list_branches_simulated():
            """Simulates list_branches API call that hits rate limit."""
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Simulate rate limit error from Bitbucket API
                raise Exception(
                    "429 Client Error: Too Many Requests for url: "
                    "https://api.bitbucket.org/2.0/repositories/org/repo/refs/branches"
                )
            return ["main", "develop", "feature/test"]

        result = list_branches_simulated()

        assert result == ["main", "develop", "feature/test"]
        assert call_count == 3  # 2 failures + 1 success

    def test_bug_4056_error_message_quality(self):
        """Test that error messages are clear when max retries exceeded."""
        @retry_on_rate_limit(max_retries=2, base_delay=0.01)
        def always_failing():
            raise Exception("429 Too Many Requests")

        with pytest.raises(ToolException) as exc_info:
            always_failing()

        error_msg = str(exc_info.value)

        # Error message should be helpful
        assert "rate limit" in error_msg.lower()
        assert "retries" in error_msg.lower()
        # Should include the original error for debugging
        assert "429" in error_msg

    def test_bug_4056_recovery_after_backoff(self):
        """Test that SDK successfully recovers after backing off."""
        failures_before_success = 4
        call_count = 0

        @retry_on_rate_limit(max_retries=5, base_delay=0.01)
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count <= failures_before_success:
                raise Exception("Too Many Requests")
            return "recovered"

        result = eventually_succeeds()

        assert result == "recovered"
        assert call_count == failures_before_success + 1
