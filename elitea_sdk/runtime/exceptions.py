class PipelineConfigurationError(Exception):
    """Raised when pipeline configuration is invalid.

    This exception is caught by the indexer and its message is displayed
    to the user, so messages should be user-friendly.
    """

    pass


# Error shape the platform's LLM proxy returns when a cost budget blocks a request.
# Scope tells the caller which budget was reached, which drives the message shown.
BUDGET_ERROR_TYPE = "budget_exceeded"
DEFAULT_BUDGET_SCOPE = "project_budget_exceeded"
BUDGET_SCOPES = (DEFAULT_BUDGET_SCOPE, "member_budget_exceeded")


class BudgetExceededError(Exception):
    """Raised when a cost budget blocks a request.

    Must never be swallowed into message content: there is no recovery from an
    exhausted budget, so continuing would feed a policy rejection back into the
    model as if it were data. Handlers let this through the way they already let
    McpAuthorizationRequired and GraphBubbleUp through.

    Subclasses Exception rather than ToolException on purpose - several call sites
    treat ToolException specially, and this must not interact with any of that.
    """

    def __init__(self, message, scope=DEFAULT_BUDGET_SCOPE):
        super().__init__(message)
        self.scope = scope if scope in BUDGET_SCOPES else DEFAULT_BUDGET_SCOPE


def budget_exceeded_from(exc):
    """Return a BudgetExceededError if exc is a budget rejection, else None.

    The single place the SDK knows the platform proxy's error contract. Kept to one
    function so the coupling stays contained, and so the SDK is unaffected when run
    without that proxy - nothing matches and every error keeps its current handling.
    """
    if isinstance(exc, BudgetExceededError):
        return exc
    #
    body = getattr(exc, "body", None)
    #
    if isinstance(body, dict):
        # The OpenAI client strips the "error" wrapper before storing body; the
        # Anthropic one keeps it, so read through either shape
        detail = body.get("error") if isinstance(body.get("error"), dict) else body
        #
        if detail.get("type") == BUDGET_ERROR_TYPE:
            return BudgetExceededError(
                detail.get("message") or str(exc), detail.get("code"),
            )
        #
        return None
    #
    # Paths that lose the structured body still carry the type in the message text
    if BUDGET_ERROR_TYPE in str(exc):
        scope = next((s for s in BUDGET_SCOPES if s in str(exc)), DEFAULT_BUDGET_SCOPE)
        return BudgetExceededError(str(exc), scope)
    #
    return None
