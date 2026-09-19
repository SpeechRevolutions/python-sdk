"""The User-Agent must track the shipped version.

HttpClient-style clients that send no User-Agent get a bare 403 from the edge, so
this header is load-bearing. It was previously a literal and drifted: PyPI served
0.2.2 while the wire said 0.2.0. These pin it to the distribution metadata.
"""

from __future__ import annotations

import re

from speechrevolutions._config import USER_AGENT, _sdk_version


def test_user_agent_matches_the_installed_distribution() -> None:
    assert USER_AGENT == f"speechrevolutions-python/{_sdk_version()}"


def test_user_agent_is_well_formed() -> None:
    assert USER_AGENT.startswith("speechrevolutions-python/")
    # A literal that someone forgot to bump is the failure this guards against,
    # so require something version-shaped rather than any non-empty string.
    version = USER_AGENT.split("/", 1)[1]
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", version), version
