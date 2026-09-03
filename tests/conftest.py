"""Pytest fixtures wrapping the builders in :mod:`scenes`."""

import pytest

from scenes import make_scene, make_swath_scene, make_varying_scene


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def varying_scene():
    return make_varying_scene()


@pytest.fixture
def swath_scene():
    return make_swath_scene()
