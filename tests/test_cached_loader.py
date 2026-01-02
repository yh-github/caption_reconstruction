import pytest
from unittest.mock import Mock
from data.data_loaders import CachedDataLoader, BaseDataLoader

@pytest.fixture
def mock_loader():
    loader = Mock(spec=BaseDataLoader)
    loader.get_data_type_name.return_value = "mock_data"
    # Use mocks that have video_id attribute
    v1 = Mock(video_id="video1")
    v2 = Mock(video_id="video2")
    loader.load.return_value = [v1, v2]
    loader.count.return_value = 2
    return loader

@pytest.fixture
def cached_loader(mock_loader):
    return CachedDataLoader(mock_loader)

def test_cache_hit(cached_loader, mock_loader):
    """Verify that subsequent calls with same parameters use the cache."""
    # First call - should trigger load
    data1 = cached_loader.load(limit=10)
    assert len(data1) == 2
    assert data1[0].video_id == "video1"
    mock_loader.load.assert_called_once_with(10)
    
    # Second call - should use cache
    data2 = cached_loader.load(limit=10)
    assert data2 == data1 # Exact same list object
    mock_loader.load.assert_called_once_with(10) # Call count should still be 1

def test_cache_miss_on_limit_change(cached_loader, mock_loader):
    """Verify that changing the limit triggers a reload."""
    # First call
    cached_loader.load(limit=10)
    mock_loader.load.assert_called_with(10)
    
    # Second call with different limit
    cached_loader.load(limit=20)
    mock_loader.load.assert_called_with(20)
    assert mock_loader.load.call_count == 2

def test_cache_miss_on_none_limit(cached_loader, mock_loader):
    """Verify handling of None limit."""
    # First call with None
    cached_loader.load(limit=None)
    mock_loader.load.assert_called_with(None)
    
    # Second call with None - should hit cache
    cached_loader.load(limit=None)
    assert mock_loader.load.call_count == 1

def test_find_uses_cache(cached_loader, mock_loader):
    """Verify find usages cache if available."""
    # Populate cache
    cached_loader.load()
    
    # Mock finding something
    # Note: The real CachedDataLoader.find iterates over the list.
    found = cached_loader.find("video1")
    assert found.video_id == "video1"
    
    # Ensure inner loader was NOT called
    mock_loader.find.assert_not_called()

def test_passthrough_methods(cached_loader):
    """Verify that get_data_type_name delegates correctly."""
    assert cached_loader.get_data_type_name() == "mock_data"
