from pydantic import BaseModel, ValidationError
import pytest

from experiment_executor.cli_parser import parse_args_to_model


class AppConfig(BaseModel):
    user_name: str
    retries: int = 3
    learning_rate: float = 0.01
    is_production: bool = False


def test_successful_parsing():
    """
    Tests the ideal case where all arguments are provided correctly and
    are successfully parsed and type-coerced.
    """
    cli_args = [
        "user_name=Alice",
        "retries=5",
        "learning_rate=0.005",
        "is_production=true"
    ]
    config = parse_args_to_model(AppConfig, cli_args)

    assert isinstance(config, AppConfig)
    assert config.user_name == "Alice"
    assert config.retries == 5
    assert isinstance(config.retries, int)
    assert config.learning_rate == 0.005
    assert isinstance(config.learning_rate, float)
    assert config.is_production is True
    assert isinstance(config.is_production, bool)


def test_missing_required_field_raises_validation_error():
    """
    Tests that a ValidationError is raised if a required field (user_name)
    is not provided in the arguments.
    """
    cli_args = ["retries=10"]

    # pytest.raises is a context manager that checks for the expected exception.
    # The test passes if the exception is raised, and fails otherwise.
    with pytest.raises(ValidationError) as exc_info:
        parse_args_to_model(AppConfig, cli_args)

    # Optionally, inspect the exception to ensure it's the one we expect.
    # This checks that the error is specifically about the 'user_name' field missing.
    assert "user_name" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)


def test_incorrect_type_raises_validation_error():
    """
    Tests that a ValidationError is raised if a value cannot be coerced
    to the correct type (e.g., a string for an int field).
    """
    cli_args = ["user_name=Bob", "retries=not-a-number"]

    with pytest.raises(ValidationError) as exc_info:
        parse_args_to_model(AppConfig, cli_args)

    # Check that the error message is about parsing an integer.
    assert "retries" in str(exc_info.value)
    assert "Input should be a valid integer" in str(exc_info.value)


def test_invalid_argument_format_raises_value_error():
    """
    Tests that a ValueError is raised if an argument does not contain '='.
    """
    cli_args = ["user_name:Charlie"]  # Using ':' instead of '='

    with pytest.raises(ValueError, match="Invalid argument format"):
        parse_args_to_model(AppConfig, cli_args)

