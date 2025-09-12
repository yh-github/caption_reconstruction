import re
from pathlib import Path
import pytest
from llm.prompting import simple_safe_format
from common_utils.jsonables import build_safe_dict, flat_dict


def test_build_safe_dict_successful_merge():
    """
    Tests that the function correctly merges multiple lists of items.
    """
    # Arrange
    list1 = [('a', 1), ('b', 2)]
    list2 = [('c', 3), ('d', 4)]
    
    # Act
    result = build_safe_dict(list1, list2)
    
    # Assert
    assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

def test_build_safe_dict_handles_identical_duplicates():
    """
    Tests that the function handles cases where a key is duplicated
    but has the same value, which should not raise an error.
    """
    # Arrange
    list1 = [('a', 1), ('b', 2)]
    list2 = [('b', 2), ('c', 3)] # ('b', 2) is duplicated
    
    # Act
    result = build_safe_dict(list1, list2)
    
    # Assert
    assert result == {'a': 1, 'b': 2, 'c': 3}

def test_build_safe_dict_raises_error_on_conflicting_duplicates():
    """
    Tests that a ValueError is raised if a key exists with different values.
    """
    # Arrange
    list1 = [('a', 1), ('b', 2)]
    list2 = [('b', 99), ('c', 3)] # ('b', 2) conflicts with ('b', 99)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Duplicate key: b v1=2 v2=99"):
        build_safe_dict(list1, list2)

def test_build_safe_dict_with_single_list():
    """
    Tests the function with just one list of items.
    """
    # Arrange
    list1 = [('a', 1), ('b', 2)]
    
    # Act
    result = build_safe_dict(list1)
    
    # Assert
    assert result == {'a': 1, 'b': 2}

def test_build_safe_dict_with_no_arguments():
    """
    Tests that the function returns an empty dictionary when called with no arguments.
    """
    # Act
    result = build_safe_dict()
    
    # Assert
    assert result == {}

def test_flat_dict():
    result = flat_dict({
        '': {
            'master_seed': 0x5EED
        },
        'data_config': {
            'name': 'toy_data',
            'path': 'datasets/toy_dataset/data.json'
        },
        'masking': {
            'seed': 43,
            'width': 2
        },
        'recon_strategy': {
            'name': 'llm_zero_shot_v2',
            'type': 'llm',
            'llm': {
                'model_name': 'gemini-2.0-flash-exp',
                'temperature': 0.6,
                'prompt_template': 'prompts/dense_zero_shot_v2.txt'
            }
        }
    })

    assert result == {
        'master_seed': 0x5EED,

        'data_config.name': 'toy_data',
        'data_config.path': 'datasets/toy_dataset/data.json',

        'masking.seed': 43,
        'masking.width': 2,

        'recon_strategy.name': 'llm_zero_shot_v2',
        'recon_strategy.type': 'llm',
        'recon_strategy.llm.model_name': 'gemini-2.0-flash-exp',
        'recon_strategy.llm.temperature': 0.6,
        'recon_strategy.llm.prompt_template': 'prompts/dense_zero_shot_v2.txt'
    }

def test_simple_safe_format():
    # Arrange
    template = "Hello, {USER_NAME}! Welcome to {LOCATION}. Your support contact is {CONTACT_PERSON}."
    substitution_data = {
        "USER_NAME": "Alex",
        "LOCATION": "the main server"
    }

    formatted_string, missing = simple_safe_format(template, substitution_data)

    assert formatted_string == "Hello, Alex! Welcome to the main server. Your support contact is {CONTACT_PERSON}."
    assert missing == ["CONTACT_PERSON"]


def test_simple_safe_format_empty_template():
    formatted_string, missing = simple_safe_format("", {})
    assert formatted_string == ""
    assert missing == []


def test_simple_safe_format_no_placeholders():
    formatted_string, missing = simple_safe_format("Hello World!", {"KEY": "value"})
    assert formatted_string == "Hello World!"
    assert missing == []


def test_simple_safe_format_empty_data():
    template = "Hello, {NAME}!"
    formatted_string, missing = simple_safe_format(template, {})
    assert formatted_string == "Hello, {NAME}!"
    assert missing == ["NAME"]


def test_simple_safe_format_repeated_placeholder():
    template = "{WORD}, {WORD}! {WORD}..."
    substitution_data = {"WORD": "hello"}
    formatted_string, missing = simple_safe_format(template, substitution_data)
    assert formatted_string == "hello, hello! hello..."
    assert missing == []


def test_simple_safe_format_special_chars():
    substitution_data = {
        "SPECIAL_1@#$": "value1",
        "NOT_SPECIAL_2___": "value2"
    }
    with pytest.raises(AssertionError, match=re.escape(r"Bad keys: ['SPECIAL_1@#$']")):
        simple_safe_format("", substitution_data)

def test_simple_safe_format_special_and_lower():
    substitution_data = {
        "SPECIAL_1@#$": "value1",
        "NOT_SPECIAL_2___": "value2",
        "lower": "value3"
    }
    with pytest.raises(AssertionError, match=re.escape(r"Bad keys: ['SPECIAL_1@#$', 'lower']")):
        simple_safe_format("", substitution_data)

def test_component_simple_safe_format():
    prompt_path = Path("../prompts")
    text_files = prompt_path.glob('**/*.txt')
    print()
    print('===========')
    for file_path in text_files:
        with open(file_path) as f:
            text = f.read()
            formatted_string1, missing1 = simple_safe_format(text, {})
            print(f"{file_path} - {missing1}")
            rep_dict = {k:"replaced {k}" for k in missing1}
            formatted_string2, missing2 = simple_safe_format(text, rep_dict)
            assert missing2 == []
            formatted_string3, missing3 = simple_safe_format(text, rep_dict)
            assert missing3 == []
            assert formatted_string2 == formatted_string3
            if '```json' in text or '{"' in text:
                with pytest.raises(KeyError):
                    text.format_map(rep_dict)
                assert formatted_string2 == text.replace('{', '{{').replace('}', '}}').format_map(rep_dict)
            else:
                assert text.format_map(rep_dict) == formatted_string2

    print('===========')
