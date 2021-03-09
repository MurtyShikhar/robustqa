class FeatureFunction:
    """This is an abstract base class for an object that evaluates context paragraphs."""

    def __init__(self):
        pass

    def evaluate(self, context: str) -> float:
        raise NotImplementedError()
