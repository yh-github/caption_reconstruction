import pytest
from utils import build_safe_dict, flat_dict


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