"""Allow running as 'python -m mindforge'."""
import sys
from mindforge.cli import main

if __name__ == "__main__":
    sys.exit(main())
