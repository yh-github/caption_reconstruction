from textwrap import dedent
import json
import pytest
from pydantic import BaseModel
from common_utils.jsonables import dump_model_compact_json, CompactJSONEncoder


class MockModel(BaseModel):
    """A simple Pydantic model for testing."""
    name: str
    value: int
    tags: list[str]


@pytest.fixture
def short_data():
    """Test data where all sub-lists are short enough to be compact."""
    return {
        "id": "item-1",
        "params": {"a": 1, "b": 2},
        "scores": [10, 20, 30]
    }


@pytest.fixture
def long_data():
    """Test data with a long sub-list that should not be compact."""
    return {
        "id": "item-2",
        "description": "A very long description that will force the parent dict to expand.",
        "params": {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4},
        "scores": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }


# --- Pytest Tests ---

def test_short_list_is_compact():
    """Tests that a short list is formatted on a single line."""
    data = [1, 2, 3, 4, 5]
    expected = "[1, 2, 3, 4, 5]"
    actual = json.dumps(data, cls=CompactJSONEncoder, compact_width=20)
    assert actual == expected


def test_long_list_is_expanded():
    """Tests that a long list is formatted on multiple lines."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expected = ("[\n"
                "  1,\n"
                "  2,\n"
                "  3,\n"
                "  4,\n"
                "  5,\n"
                "  6,\n"
                "  7,\n"
                "  8,\n"
                "  9,\n"
                "  10\n"
                "]")
    actual = json.dumps(data, cls=CompactJSONEncoder, indent=2, compact_width=20)
    assert actual == expected


def test_nested_compactable_list(long_data):
    """
    Tests that a short sub-list inside an expanded dictionary remains on one line.
    """
    # Make the 'params' short enough to be compact, but the parent is still long.
    long_data["params"] = {"a": 1}

    # We expect the 'params' dict to be on a single line.
    expected_line = '  "params": {"a": 1},'

    actual_json = json.dumps(long_data, cls=CompactJSONEncoder, indent=2, compact_width=40)

    assert expected_line in actual_json
    # Verify the parent is still expanded by checking for newlines
    assert "\n" in actual_json, f"{len(actual_json.splitlines())=}"


def test_parent_dict_is_compact(short_data):
    """
    Tests that if the parent dictionary is compactable, the whole thing is one line.
    """
    expected = '{"id": "item-1", "params": {"a": 1, "b": 2}, "scores": [10, 20, 30]}'
    actual = json.dumps(short_data, cls=CompactJSONEncoder, indent=2, compact_width=100)
    assert actual == expected


def test_dump_model_compact_json_with_single_model():
    """Tests the wrapper function with a single Pydantic model."""
    model = MockModel(name="test", value=1, tags=["a", "b"])
    # The model dump is short enough to be on one line
    expected = '{"name": "test", "value": 1, "tags": ["a", "b"]}'
    actual = dump_model_compact_json(model, width=100)
    assert actual == expected


def test_dump_model_compact_json_with_list_of_models():
    """Tests the wrapper function with a list of Pydantic models."""
    models = [
        MockModel(name="test1", value=1, tags=["a", "b"]),
        MockModel(name="test2", value=2, tags=["c", "d"]),
    ]
    # The list of models should be expanded, but each model within it should be compact
    expected = ('[\n'
                '  {"name": "test1", "value": 1, "tags": ["a", "b"]},\n'
                '  {"name": "test2", "value": 2, "tags": ["c", "d"]}\n'
                ']')
    actual = dump_model_compact_json(models, width=100)
    assert actual == expected


def test_dump_model_with_code_block():
    """Tests that the code_block=True flag works correctly."""
    model = MockModel(name="test", value=1, tags=["a", "b"])
    result = dump_model_compact_json(model, width=100, code_block=True)
    assert result.startswith("")
    assert result.endswith("\n```")
    assert '```json\n{"name": "test", "value": 1, "tags": ["a", "b"]}\n```' == result


def test_dump_model_with_code_block_over():
    """Tests that the code_block=True flag works correctly."""
    model = MockModel(name="test", value=1, tags=["a", "b"])
    result = dump_model_compact_json(model, width=15, code_block=True)
    assert result.startswith("")
    assert result.endswith("\n```")
    expected = dedent("""\
    ```json
    {
      "name": "test",
      "value": 1,
      "tags": ["a", "b"]
    }
    ```""")
    assert result == expected