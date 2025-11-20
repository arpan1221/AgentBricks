"""CLI tool for generating synthetic user, movie, and interaction data."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sim.agents.user import UserAgent
from sim.items.movie import Movie
from sim.interactions.rules import (
    calculate_watch_probability,
    simulate_watch_time,
    simulate_rating,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging level based on verbose flag.

    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


@click.group()
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run without saving files (for testing)"
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, seed: Optional[int], dry_run: bool) -> None:
    """
    AgentBricks synthetic data generation tool.

    Generate synthetic users, movies, and interactions for training
    recommendation systems.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["seed"] = seed
    ctx.obj["dry_run"] = dry_run

    setup_logging(verbose)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")

    if dry_run:
        logger.info("DRY RUN MODE: No files will be saved")


@cli.command()
@click.option(
    "--count",
    type=int,
    default=10000,
    help="Number of users to generate"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="users.parquet",
    help="Output file path (Parquet format)"
)
@click.pass_context
def generate_users(
    ctx: click.Context,
    count: int,
    output: Path
) -> None:
    """
    Generate synthetic user agents.

    Creates UserAgent instances with random preferences, age groups,
    regions, and activity patterns. Saves to Parquet format.

    Example:
        $ python -m sim.generate generate-users --count 5000 --output data/users.parquet
    """
    if count <= 0:
        raise click.BadParameter(f"count must be positive, got {count}")

    dry_run = ctx.obj.get("dry_run", False)

    logger.info(f"Generating {count} users...")

    users = []

    # Generate users with progress bar
    with tqdm(total=count, desc="Generating users", unit="user") as pbar:
        for i in range(count):
            try:
                user_id = f"user_{i:08d}"
                user = UserAgent(user_id=user_id)
                users.append(user.to_dict())
                pbar.update(1)
            except Exception as e:
                logger.error(f"Failed to generate user {user_id}: {e}", exc_info=True)
                raise click.Abort(f"Failed to generate user: {e}")

    # Convert to DataFrame
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(users)

    # Calculate summary statistics
    age_group_counts = df["age_group"].value_counts().to_dict()
    region_counts = df["region"].value_counts().to_dict()

    logger.info(f"Generated {len(df)} users")
    logger.info(f"Age group distribution: {age_group_counts}")
    logger.info(f"Region distribution: {region_counts}")

    # Save to Parquet
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, engine="pyarrow", compression="snappy")
        logger.info(f"Saved users to {output}")
    else:
        logger.info(f"[DRY RUN] Would save {len(df)} users to {output}")
        logger.info(f"Preview (first 5 rows):\n{df.head()}")


@cli.command()
@click.option(
    "--count",
    type=int,
    default=5000,
    help="Number of movies to generate"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="movies.parquet",
    help="Output file path (Parquet format)"
)
@click.pass_context
def generate_movies(
    ctx: click.Context,
    count: int,
    output: Path
) -> None:
    """
    Generate synthetic movies with diverse attributes.

    Creates Movie instances with random titles, genres, release years,
    durations, and popularity scores. Ensures genre diversity.

    Example:
        $ python -m sim.generate generate-movies --count 3000 --output data/movies.parquet
    """
    if count <= 0:
        raise click.BadParameter(f"count must be positive, got {count}")

    dry_run = ctx.obj.get("dry_run", False)
    seed = ctx.obj.get("seed")

    logger.info(f"Generating {count} movies...")

    movies = []

    # Generate movies with progress bar
    with tqdm(total=count, desc="Generating movies", unit="movie") as pbar:
        for i in range(count):
            try:
                movie_id = f"movie_{i:06d}"
                # Use movie index as part of seed for reproducibility
                movie_seed = (seed + i) if seed is not None else None
                movie = Movie.generate_random(movie_id=movie_id, seed=movie_seed)
                movies.append(movie.to_dict())
                pbar.update(1)
            except Exception as e:
                logger.error(f"Failed to generate movie {movie_id}: {e}", exc_info=True)
                raise click.Abort(f"Failed to generate movie: {e}")

    # Convert to DataFrame
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(movies)

    # Calculate summary statistics
    genre_counts = {}
    for genres_list in df["genres"]:
        for genre in genres_list:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    year_stats = {
        "min": int(df["release_year"].min()),
        "max": int(df["release_year"].max()),
        "mean": float(df["release_year"].mean()),
    }

    duration_stats = {
        "min": int(df["duration_minutes"].min()),
        "max": int(df["duration_minutes"].max()),
        "mean": float(df["duration_minutes"].mean()),
    }

    logger.info(f"Generated {len(df)} movies")
    logger.info(f"Genre distribution (top 10): {dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
    logger.info(f"Release year stats: {year_stats}")
    logger.info(f"Duration stats (minutes): {duration_stats}")

    # Save to Parquet
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, engine="pyarrow", compression="snappy")
        logger.info(f"Saved movies to {output}")
    else:
        logger.info(f"[DRY RUN] Would save {len(df)} movies to {output}")
        logger.info(f"Preview (first 5 rows):\n{df.head()}")


@cli.command()
@click.option(
    "--users",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to users Parquet file"
)
@click.option(
    "--movies",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to movies Parquet file"
)
@click.option(
    "--days",
    type=int,
    default=30,
    help="Number of days to simulate interactions over"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="interactions.parquet",
    help="Output file path (Parquet format)"
)
@click.option(
    "--max-daily-interactions",
    type=int,
    default=10,
    help="Maximum interactions per user per day"
)
@click.pass_context
def generate_interactions(
    ctx: click.Context,
    users: Path,
    movies: Path,
    days: int,
    output: Path,
    max_daily_interactions: int
) -> None:
    """
    Generate synthetic user-movie interactions.

    Simulates users watching movies over a time period using interaction
    rules. Generates events: views, ratings, and skips.

    Example:
        $ python -m sim.generate generate-interactions \\
            --users data/users.parquet \\
            --movies data/movies.parquet \\
            --days 30 \\
            --output data/interactions.parquet
    """
    if days <= 0:
        raise click.BadParameter(f"days must be positive, got {days}")

    if max_daily_interactions <= 0:
        raise click.BadParameter(
            f"max-daily-interactions must be positive, got {max_daily_interactions}"
        )

    dry_run = ctx.obj.get("dry_run", False)

    logger.info(f"Loading users from {users}...")
    try:
        users_df = pd.read_parquet(users, engine="pyarrow")
        logger.info(f"Loaded {len(users_df)} users")
    except Exception as e:
        logger.error(f"Failed to load users: {e}", exc_info=True)
        raise click.Abort(f"Failed to load users: {e}")

    logger.info(f"Loading movies from {movies}...")
    try:
        movies_df = pd.read_parquet(movies, engine="pyarrow")
        logger.info(f"Loaded {len(movies_df)} movies")
    except Exception as e:
        logger.error(f"Failed to load movies: {e}", exc_info=True)
        raise click.Abort(f"Failed to load movies: {e}")

    # Convert DataFrames to objects
    logger.info("Converting to UserAgent and Movie objects...")
    user_agents = []
    for _, row in users_df.iterrows():
        try:
            user = UserAgent(
                user_id=row["user_id"],
                age_group=row["age_group"],
                region=row["region"]
            )
            # Override with saved preferences
            user.preference_vector = np.array(row["preference_vector"])
            user.activity_pattern = row["activity_pattern"]
            user_agents.append(user)
        except Exception as e:
            logger.warning(f"Failed to create UserAgent for {row['user_id']}: {e}")
            continue

    movie_objects = []
    for _, row in movies_df.iterrows():
        try:
            movie = Movie.from_dict(row.to_dict())
            movie_objects.append(movie)
        except Exception as e:
            logger.warning(f"Failed to create Movie for {row['movie_id']}: {e}")
            continue

    logger.info(f"Created {len(user_agents)} user agents and {len(movie_objects)} movies")

    # Generate interactions
    logger.info(f"Simulating interactions over {days} days...")
    interactions = []

    # Time range
    start_date = datetime.now() - timedelta(days=days)

    # Estimate total iterations for progress bar (users * days * avg interactions)
    total_iterations = len(user_agents) * days
    with tqdm(total=total_iterations, desc="Processing user-days", unit="user-day") as pbar:
        for user in user_agents:
            for day_offset in range(days):
                current_date = start_date + timedelta(days=day_offset)
                day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
                is_weekend = day_of_week >= 5

                # Sample number of interactions for this day
                num_interactions = np.random.poisson(lam=max_daily_interactions * 0.5)
                num_interactions = min(num_interactions, max_daily_interactions)

                # Select random movies to consider
                if num_interactions > 0 and len(movie_objects) > 0:
                    sample_size = min(num_interactions, len(movie_objects))
                    movie_indices = np.random.choice(
                        len(movie_objects),
                        size=sample_size,
                        replace=False
                    )

                    for movie_idx in movie_indices:
                        movie = movie_objects[movie_idx]

                        # Random hour of day (weighted towards evening)
                        hour = np.random.choice(
                            24,
                            p=[0.02] * 6 + [0.03] * 6 + [0.05] * 6 + [0.1] * 6  # Evening higher
                        )

                        # Create context
                        context = {
                            "hour": int(hour),
                            "day_of_week": int(day_of_week),
                            "is_weekend": is_weekend,
                        }

                        # Calculate watch probability
                        try:
                            watch_prob = calculate_watch_probability(user, movie, context)
                            did_watch = np.random.random() < watch_prob

                            if did_watch:
                                # Simulate watch time
                                watch_time = simulate_watch_time(user, movie, did_watch=True)

                                # Simulate rating
                                rating = simulate_rating(user, movie, watch_time)

                                # Create interaction record
                                interaction = {
                                    "user_id": user.user_id,
                                    "movie_id": movie.movie_id,
                                    "timestamp": current_date.replace(hour=hour, minute=0, second=0),
                                    "event_type": "view",
                                    "watch_time_seconds": watch_time,
                                    "rating": rating,
                                }
                                interactions.append(interaction)
                            else:
                                # Record skip
                                interaction = {
                                    "user_id": user.user_id,
                                    "movie_id": movie.movie_id,
                                    "timestamp": current_date.replace(hour=hour, minute=0, second=0),
                                    "event_type": "skip",
                                    "watch_time_seconds": 0,
                                    "rating": None,
                                }
                                interactions.append(interaction)
                        except Exception as e:
                            logger.warning(
                                f"Failed to generate interaction for {user.user_id} "
                                f"and {movie.movie_id}: {e}"
                            )

                pbar.update(1)
                # Update progress description with interaction count
                pbar.set_postfix(interactions=len(interactions))

    # Convert to DataFrame
    logger.info("Converting interactions to DataFrame...")
    df = pd.DataFrame(interactions)

    if len(df) == 0:
        logger.warning("No interactions generated!")
        return

    # Calculate summary statistics
    event_type_counts = df["event_type"].value_counts().to_dict()
    total_views = event_type_counts.get("view", 0)
    total_ratings = df["rating"].notna().sum()
    avg_watch_time = df[df["event_type"] == "view"]["watch_time_seconds"].mean()
    avg_rating = df["rating"].mean()

    logger.info(f"Generated {len(df)} interactions")
    logger.info(f"Event type distribution: {event_type_counts}")
    logger.info(f"Total views: {total_views}")
    logger.info(f"Total ratings: {total_ratings}")
    logger.info(f"Average watch time (seconds): {avg_watch_time:.1f}")
    logger.info(f"Average rating: {avg_rating:.2f}" if not pd.isna(avg_rating) else "No ratings")

    # Save to Parquet
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, engine="pyarrow", compression="snappy")
        logger.info(f"Saved interactions to {output}")
    else:
        logger.info(f"[DRY RUN] Would save {len(df)} interactions to {output}")
        logger.info(f"Preview (first 10 rows):\n{df.head(10)}")


@cli.command()
@click.option(
    "--user-count",
    type=int,
    default=10000,
    help="Number of users to generate"
)
@click.option(
    "--movie-count",
    type=int,
    default=5000,
    help="Number of movies to generate"
)
@click.option(
    "--days",
    type=int,
    default=30,
    help="Number of days to simulate interactions over"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="data",
    help="Output directory for all files"
)
@click.pass_context
def generate_all(
    ctx: click.Context,
    user_count: int,
    movie_count: int,
    days: int,
    output_dir: Path
) -> None:
    """
    Generate all synthetic data (users, movies, interactions).

    Runs all three generation commands in sequence and saves all files
    to the specified output directory.

    Example:
        $ python -m sim.generate generate-all \\
            --user-count 10000 \\
            --movie-count 5000 \\
            --days 30 \\
            --output-dir data/
    """
    logger.info("Starting full data generation pipeline...")

    # Create output directory
    if not ctx.obj.get("dry_run", False):
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Generate users
    logger.info("=" * 60)
    logger.info("Step 1: Generating users")
    logger.info("=" * 60)
    users_path = output_dir / "users.parquet"
    try:
        ctx.invoke(
            generate_users,
            count=user_count,
            output=users_path
        )
    except Exception as e:
        logger.error(f"Failed to generate users: {e}", exc_info=True)
        raise click.Abort(f"Data generation failed at user generation step: {e}")

    # Generate movies
    logger.info("=" * 60)
    logger.info("Step 2: Generating movies")
    logger.info("=" * 60)
    movies_path = output_dir / "movies.parquet"
    try:
        ctx.invoke(
            generate_movies,
            count=movie_count,
            output=movies_path
        )
    except Exception as e:
        logger.error(f"Failed to generate movies: {e}", exc_info=True)
        raise click.Abort(f"Data generation failed at movie generation step: {e}")

    # Generate interactions
    logger.info("=" * 60)
    logger.info("Step 3: Generating interactions")
    logger.info("=" * 60)
    interactions_path = output_dir / "interactions.parquet"
    try:
        ctx.invoke(
            generate_interactions,
            users=users_path,
            movies=movies_path,
            days=days,
            output=interactions_path
        )
    except Exception as e:
        logger.error(f"Failed to generate interactions: {e}", exc_info=True)
        raise click.Abort(f"Data generation failed at interaction generation step: {e}")

    logger.info("=" * 60)
    logger.info("Data generation completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Generated files:")
    logger.info(f"  - Users: {users_path}")
    logger.info(f"  - Movies: {movies_path}")
    logger.info(f"  - Interactions: {interactions_path}")


if __name__ == "__main__":
    cli()
