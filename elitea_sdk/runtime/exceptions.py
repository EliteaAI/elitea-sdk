class PipelineConfigurationError(Exception):
    """Raised when pipeline configuration is invalid.

    This exception is caught by the indexer and its message is displayed
    to the user, so messages should be user-friendly.
    """

    pass
