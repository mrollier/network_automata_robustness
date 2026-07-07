import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR / "characterization"))

FIXTURES = TESTS_DIR / "characterization" / "fixtures"


@pytest.fixture(scope="session")
def fx():
    """Load all characterization fixture archives once per session."""
    return {p.stem: np.load(p) for p in sorted(FIXTURES.glob("*.npz"))}
