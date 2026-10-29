"""Argument validators shared by the tool-use test case files."""

import re
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class ArgSpec:
    validator: Callable[[Any], bool]
    description: str = ""

def exact(value):
    return ArgSpec(lambda v: v == value, f"== {value!r}")

def within(lo, hi):
    return ArgSpec(lambda v: lo <= v <= hi, f"in [{lo}, {hi}]")

def one_of(*values):
    return ArgSpec(lambda v: v in values, f"in {values}")

def matches(pattern):
    return ArgSpec(lambda v: bool(re.match(pattern, str(v))), f"matches {pattern}")

def any_value():
    return ArgSpec(lambda v: True, "unconstrained")
