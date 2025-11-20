"""Synthetic movie items with metadata and embeddings."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Valid genres for movies
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Sport",
    "Thriller",
    "War",
    "Western",
]


@dataclass
class Movie:
    """
    Synthetic movie item with metadata and embedding vector.

    Represents a movie in the AgentBricks simulation with title, genres,
    release year, duration, popularity, and a 50-dimensional embedding vector
    that captures semantic features for recommendation matching.

    Attributes:
        movie_id: Unique identifier for the movie
        title: Movie title
        genres: List of genre tags (validated against GENRES)
        release_year: Year the movie was released
        duration_minutes: Movie duration in minutes
        embedding: 50-dimensional embedding vector
        popularity_score: Popularity score normalized to [0, 1]

    Example:
        >>> # Create movie with specified attributes
        >>> movie = Movie(
        ...     movie_id="movie_123",
        ...     title="The Matrix",
        ...     genres=["Action", "Sci-Fi"],
        ...     release_year=1999,
        ...     duration_minutes=136,
        ...     popularity_score=0.85
        ... )
        >>>
        >>> # Generate random movie
        >>> random_movie = Movie.generate_random("movie_456")
        >>>
        >>> # Calculate similarity with user preferences
        >>> user_prefs = np.random.rand(50)
        >>> similarity = movie.calculate_similarity(user_prefs)
        >>> print(f"Similarity: {similarity:.3f}")
        Similarity: 0.234
    """

    movie_id: str
    title: str
    genres: List[str]
    release_year: int
    duration_minutes: int
    embedding: npt.NDArray[np.float64] = field(init=False)
    popularity_score: float = 0.5

    def __post_init__(self) -> None:
        """
        Initialize movie attributes after dataclass initialization.

        Validates inputs and generates embedding if not provided.

        Raises:
            ValueError: If genres are invalid
            ValueError: If release_year is out of valid range
            ValueError: If duration_minutes is invalid
            ValueError: If popularity_score is out of [0, 1] range
        """
        # Validate genres
        self._validate_genres(self.genres)

        # Validate release year (reasonable range: 1900-2030)
        if not (1900 <= self.release_year <= 2030):
            raise ValueError(
                f"release_year must be between 1900 and 2030, got {self.release_year}"
            )

        # Validate duration (reasonable range: 5 minutes to 4 hours)
        if not (5 <= self.duration_minutes <= 240):
            raise ValueError(
                f"duration_minutes must be between 5 and 240, got {self.duration_minutes}"
            )

        # Validate popularity score
        if not (0.0 <= self.popularity_score <= 1.0):
            raise ValueError(
                f"popularity_score must be between 0.0 and 1.0, got {self.popularity_score}"
            )

        # Generate embedding based on genres
        self.embedding = self.generate_embedding()

        logger.debug(
            f"Created movie",
            extra={
                "movie_id": self.movie_id,
                "title": self.title,
                "genres": self.genres,
                "release_year": self.release_year,
            }
        )

    def _validate_genres(self, genres: List[str]) -> None:
        """
        Validate that all genres are in the known list.

        Args:
            genres: List of genre strings to validate

        Raises:
            ValueError: If any genre is not in the known GENRES list
        """
        if not genres:
            raise ValueError("genres list cannot be empty")

        invalid_genres = [g for g in genres if g not in GENRES]
        if invalid_genres:
            raise ValueError(
                f"Invalid genres: {invalid_genres}. "
                f"Valid genres are: {GENRES}"
            )

    def generate_embedding(
        self,
        genre_weights: Optional[Dict[str, float]] = None,
        n_dims: int = 50,
        seed: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        """
        Generate movie embedding vector based on genres.

        Creates a 50-dimensional embedding vector that encodes genre information
        and other semantic features. The embedding is normalized to unit length
        for cosine similarity calculations.

        Args:
            genre_weights: Optional dictionary mapping genres to weights.
                          If None, uses uniform weights for movie's genres.
            n_dims: Dimension of embedding vector (default: 50)
            seed: Optional random seed for reproducibility

        Returns:
            Normalized embedding vector of shape (n_dims,)

        Example:
            >>> movie = Movie(
            ...     movie_id="m1",
            ...     title="Test Movie",
            ...     genres=["Action", "Sci-Fi"],
            ...     release_year=2020,
            ...     duration_minutes=120
            ... )
            >>> embedding = movie.generate_embedding()
            >>> print(embedding.shape)
            (50,)
            >>> print(np.linalg.norm(embedding))
            1.0
        """
        if n_dims <= 0:
            raise ValueError(f"n_dims must be positive, got {n_dims}")

        # Use seeded random if provided (for reproducibility)
        rng = np.random.default_rng(seed)

        # Initialize embedding vector
        embedding = np.zeros(n_dims, dtype=np.float64)

        # Generate genre-based components
        if genre_weights is None:
            # Uniform weights for all genres in the movie
            genre_weights = {genre: 1.0 / len(self.genres) for genre in self.genres}

        # Create genre embeddings (deterministic hash-based)
        for genre, weight in genre_weights.items():
            if genre in GENRES:
                # Create deterministic vector for this genre
                genre_idx = GENRES.index(genre)
                genre_seed = hash(genre) % (2**31)  # Deterministic seed from genre
                genre_rng = np.random.default_rng(genre_seed)

                # Generate genre-specific component
                genre_component = genre_rng.normal(loc=0.0, scale=1.0, size=n_dims)
                embedding += weight * genre_component

        # Add popularity-based component
        popularity_component = rng.normal(loc=0.0, scale=0.5, size=n_dims)
        embedding += self.popularity_score * 0.3 * popularity_component

        # Add temporal component (based on release year)
        year_normalized = (self.release_year - 1900) / 130.0  # Normalize to [0, 1]
        temporal_component = rng.normal(loc=0.0, scale=0.3, size=n_dims)
        embedding += year_normalized * 0.2 * temporal_component

        # Add duration component
        duration_normalized = (self.duration_minutes - 5) / 235.0  # Normalize to [0, 1]
        duration_component = rng.normal(loc=0.0, scale=0.2, size=n_dims)
        embedding += duration_normalized * 0.1 * duration_component

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            # Fallback: use uniform distribution if norm is zero (rare)
            logger.warning(
                f"Embedding norm was zero for movie {self.movie_id}, "
                "using uniform distribution fallback"
            )
            embedding = rng.uniform(low=-1.0, high=1.0, size=n_dims)
            embedding = embedding / np.linalg.norm(embedding)

        logger.debug(
            f"Generated embedding",
            extra={
                "movie_id": self.movie_id,
                "n_dims": n_dims,
                "norm": float(np.linalg.norm(embedding)),
            }
        )

        return embedding.astype(np.float64)

    def calculate_similarity(
        self,
        user_preferences: npt.NDArray[np.float64]
    ) -> float:
        """
        Calculate cosine similarity between movie embedding and user preferences.

        Computes the cosine similarity (dot product of normalized vectors)
        between the movie's embedding and user's preference vector. This
        similarity score indicates how well the movie matches the user's
        preferences.

        Args:
            user_preferences: User's preference vector (should match embedding
                            dimension)

        Returns:
            Cosine similarity score in range [-1, 1], where:
            - 1.0: Perfect match
            - 0.0: No correlation
            - -1.0: Opposite preferences

        Raises:
            ValueError: If user_preferences dimension doesn't match embedding
            ValueError: If user_preferences is empty

        Example:
            >>> movie = Movie(
            ...     movie_id="m1",
            ...     title="Action Movie",
            ...     genres=["Action"],
            ...     release_year=2020,
            ...     duration_minutes=120
            ... )
            >>> user_prefs = np.random.rand(50)
            >>> similarity = movie.calculate_similarity(user_prefs)
            >>> print(f"Similarity: {similarity:.3f}")
            Similarity: 0.234
        """
        if user_preferences.size == 0:
            raise ValueError("user_preferences cannot be empty")

        if user_preferences.shape[0] != self.embedding.shape[0]:
            raise ValueError(
                f"user_preferences dimension {user_preferences.shape[0]} "
                f"does not match embedding dimension {self.embedding.shape[0]}"
            )

        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = np.dot(self.embedding, user_preferences)

        logger.debug(
            f"Calculated similarity",
            extra={
                "movie_id": self.movie_id,
                "similarity": float(similarity),
            }
        )

        return float(similarity)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert movie to dictionary for serialization.

        Serializes the movie to a dictionary format that can be easily stored
        in databases or JSON files. Embedding vector is converted to a list.

        Returns:
            Dictionary representation of the movie

        Example:
            >>> movie = Movie(
            ...     movie_id="m1",
            ...     title="Test Movie",
            ...     genres=["Action"],
            ...     release_year=2020,
            ...     duration_minutes=120
            ... )
            >>> movie_dict = movie.to_dict()
            >>> print(movie_dict["movie_id"])
            'm1'
            >>> print(len(movie_dict["embedding"]))
            50
        """
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "release_year": self.release_year,
            "duration_minutes": self.duration_minutes,
            "embedding": self.embedding.tolist(),
            "popularity_score": self.popularity_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Movie":
        """
        Create Movie instance from dictionary.

        Deserializes a movie from a dictionary format (e.g., from database
        or JSON file). Reconstructs the embedding vector from list.

        Args:
            data: Dictionary with movie data. Must contain:
                - movie_id: str
                - title: str
                - genres: List[str]
                - release_year: int
                - duration_minutes: int
                - embedding: List[float] (optional, will be regenerated if missing)
                - popularity_score: float (optional, default: 0.5)

        Returns:
            Movie instance created from dictionary data

        Raises:
            KeyError: If required keys are missing
            ValueError: If data validation fails

        Example:
            >>> data = {
            ...     "movie_id": "m1",
            ...     "title": "Test Movie",
            ...     "genres": ["Action", "Sci-Fi"],
            ...     "release_year": 2020,
            ...     "duration_minutes": 120,
            ...     "embedding": [0.1] * 50,
            ...     "popularity_score": 0.8
            ... }
            >>> movie = Movie.from_dict(data)
            >>> print(movie.title)
            'Test Movie'
        """
        # Extract required fields
        movie_id = data["movie_id"]
        title = data["title"]
        genres = data["genres"]
        release_year = data["release_year"]
        duration_minutes = data["duration_minutes"]
        popularity_score = data.get("popularity_score", 0.5)

        # Create movie instance (will generate embedding in __post_init__)
        movie = cls(
            movie_id=movie_id,
            title=title,
            genres=genres,
            release_year=release_year,
            duration_minutes=duration_minutes,
            popularity_score=popularity_score,
        )

        # Override embedding if provided in data
        if "embedding" in data and data["embedding"]:
            embedding_array = np.array(data["embedding"], dtype=np.float64)
            if embedding_array.shape[0] == movie.embedding.shape[0]:
                # Normalize if not already normalized
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    movie.embedding = (embedding_array / norm).astype(np.float64)
                else:
                    logger.warning(
                        f"Embedding norm was zero for movie {movie_id}, "
                        "using generated embedding"
                    )
            else:
                logger.warning(
                    f"Embedding dimension mismatch for movie {movie_id}, "
                    f"expected {movie.embedding.shape[0]}, got {embedding_array.shape[0]}. "
                    "Using generated embedding"
                )

        logger.debug(
            f"Created movie from dict",
            extra={"movie_id": movie_id}
        )

        return movie

    @classmethod
    def generate_random(
        cls,
        movie_id: str,
        seed: Optional[int] = None
    ) -> "Movie":
        """
        Generate a random movie with realistic attributes.

        Creates a movie with randomly selected title, genres, release year,
        duration, and popularity score. Useful for generating synthetic datasets
        for testing and training.

        Args:
            movie_id: Unique identifier for the movie
            seed: Optional random seed for reproducibility

        Returns:
            Movie instance with randomly generated attributes

        Example:
            >>> movie = Movie.generate_random("movie_123")
            >>> print(movie.title)
            'Random Movie Title'
            >>> print(movie.genres)
            ['Action', 'Drama']
            >>> print(movie.release_year)
            2015
        """
        rng = np.random.default_rng(seed)

        # Generate random title
        title_prefixes = [
            "The", "A", "An", "My", "Your", "Our", "Their"
        ]
        title_nouns = [
            "Adventure", "Journey", "Quest", "Story", "Tale", "Legend",
            "Mystery", "Secret", "Truth", "Promise", "Dream", "Hope",
            "Warrior", "Hero", "Legend", "Master", "Guardian", "Protector"
        ]
        title_suffixes = [
            "", " Returns", " Rises", " Falls", " Begins", " Ends",
            " Redemption", " Awakening", " Discovery"
        ]

        prefix = rng.choice(title_prefixes)
        noun = rng.choice(title_nouns)
        suffix = rng.choice(title_suffixes)
        title = f"{prefix} {noun}{suffix}".strip()

        # Generate random genres (1-3 genres per movie)
        num_genres = rng.integers(1, 4)
        genres = rng.choice(GENRES, size=num_genres, replace=False).tolist()

        # Generate release year (weighted towards recent years)
        # Most movies from 1990-2020, some older
        year_weights = np.concatenate([
            np.ones(10) * 0.5,  # 1900-1909: 0.5 weight
            np.ones(80) * 1.0,  # 1910-1989: 1.0 weight
            np.ones(30) * 2.0,  # 1990-2019: 2.0 weight (more common)
            np.ones(11) * 1.5,  # 2020-2030: 1.5 weight
        ])
        years = np.arange(1900, 2031)
        release_year = int(rng.choice(years, p=year_weights / year_weights.sum()))

        # Generate duration (most movies 90-150 minutes)
        # Use normal distribution centered at 120 minutes
        duration_minutes = int(rng.normal(loc=120, scale=30))
        duration_minutes = np.clip(duration_minutes, a_min=60, a_max=180)

        # Generate popularity score (skewed towards lower scores)
        # Most movies are less popular, few are very popular
        popularity_score = float(rng.beta(a=2, b=5))  # Skewed right distribution

        movie = cls(
            movie_id=movie_id,
            title=title,
            genres=genres,
            release_year=release_year,
            duration_minutes=duration_minutes,
            popularity_score=popularity_score,
        )

        logger.info(
            f"Generated random movie",
            extra={
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
            }
        )

        return movie
