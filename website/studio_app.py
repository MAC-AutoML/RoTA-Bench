"""Launch the existing portal on the ModelScope CPU Studio port."""

import os

from app import main

if __name__ == "__main__":
    os.environ["PORT"] = "7860"
    main()
