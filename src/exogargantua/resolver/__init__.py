"""Calibrated period-alias resolver (brief §5). See docs/resolver_design.md."""

from .combine import DEFAULT_ABSTAIN, LearnedCombiner, alias_rows, principled  # noqa: F401
from .core import FEATURES, Star, resolve  # noqa: F401
from .hypotheses import ALIAS_SET  # noqa: F401
