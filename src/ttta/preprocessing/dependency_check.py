class DependencyChecker:
    """Centralized checker for optional dependencies."""

    _wordcloud_available = None
    _embeddings_available = None

    @classmethod
    def check_wordcloud(cls):
        """Check if wordcloud dependencies are installed."""
        if cls._wordcloud_available is None:
            try:
                import wordcloud
                cls._wordcloud_available = True
            except ImportError:
                cls._wordcloud_available = False
        return cls._wordcloud_available

    @classmethod
    def check_embeddings(cls):
        """Check if embeddings dependencies are installed."""
        if cls._embeddings_available is None:
            try:
                import transformers
                import gensim
                import torch
                import wasabi
                cls._embeddings_available = True
            except ImportError:
                cls._embeddings_available = False
        return cls._embeddings_available

    @classmethod
    def require_wordcloud(cls):
        """Raise ImportError if wordcloud dependencies are not installed."""
        if not cls.check_wordcloud():
            raise ImportError(
                "Wordcloud functionality requires additional dependencies.\n"
                "Install with: pip install ttta[wordcloud]"
                )

    @classmethod
    def require_embeddings(cls):
        """Raise ImportError if embeddings dependencies are not installed."""
        if not cls.check_embeddings():
            raise ImportError(
                "Embeddings functionality requires additional dependencies.\n"
                "Install with: pip install ttta[embeddings]"
                )
