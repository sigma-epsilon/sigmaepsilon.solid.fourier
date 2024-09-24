class NavierLoadError(Exception):
    """
    Exception raised for invalid load inputs.
    """

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "Invalid input for loads. "
                "It must be an instance of `sigmaepsilon.solid.fourier.LoadGroup`."
            )
        super().__init__(message)
