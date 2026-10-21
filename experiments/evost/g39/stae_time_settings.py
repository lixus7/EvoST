"""Settings of g39 fixed in PRESPEC.md from development data (read by train_stae.py and the package builder)."""
TIME_INIT_STD = 0.02          # the time-of-day / day-of-week embeddings start N(0, 0.02^2); the original's N(0, 1) diverged under the repository's recipe (PRESPEC.md)
