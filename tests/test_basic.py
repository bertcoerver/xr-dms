"""Basic import and version tests."""

import xr_dms


def test_version():
    """Test that version is defined."""
    assert hasattr(xr_dms, "__version__")
    assert isinstance(xr_dms.__version__, str)
