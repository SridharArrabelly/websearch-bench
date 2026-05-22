"""Allow ``python -m websearch_bench`` to run the comparison harness."""

from .compare import cli

if __name__ == "__main__":
    cli()
