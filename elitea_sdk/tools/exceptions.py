class ToolkitConfigurationError(Exception):
    """Raised when toolkit configuration validation fails.

    Attributes:
        user_message: Human-friendly error message for UI display.
    """

    def __init__(self, user_message: str, cause: Exception):
        self.user_message = user_message
        super().__init__(str(cause))
