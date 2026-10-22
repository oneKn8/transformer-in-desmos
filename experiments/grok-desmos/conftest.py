"""Tests import the shared passes from language-model/ (two levels up)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
