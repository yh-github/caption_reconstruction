import sys
from typing import Type, TypeVar, Any
from pydantic import BaseModel, ValidationError

# Create a Generic TypeVar. This allows our function to have a precise
# return type hint. If you pass in `MyConfig`, the IDE will know the
# function returns an instance of `MyConfig`.
T = TypeVar('T', bound=BaseModel)


def parse_args_to_model(model_class: Type[T], args_list: list[str]) -> T:
    """
    Parses a list of "KEY=VALUE" strings into an instance of a Pydantic model.

    This function builds a dictionary from the command-line arguments and then
    uses it to instantiate the provided Pydantic model class. Pydantic
    handles the validation and type coercion from string to the correct field type.

    Args:
        model_class: The Pydantic BaseModel class to instantiate.
        args_list: A list of strings, typically from sys.argv[1:].

    Returns:
        A validated instance of the provided model_class.

    Raises:
        ValueError: If an argument is not in the "KEY=VALUE" format.
        ValidationError: If the provided values fail Pydantic's validation
                         (e.g., wrong type, missing required fields).
    """
    data: dict[str, Any] = {}
    for arg in args_list:
        if '=' not in arg:
            raise ValueError(
                f"Invalid argument format: '{arg}'. "
                "All arguments must be in 'KEY=VALUE' format."
            )
        # Split only on the first '=' to allow for values that contain '='
        key, value = arg.split('=', 1)
        data[key] = value

    try:
        return model_class(**data)
    except ValidationError:
        # possible handling here
        raise


# --- Example Usage & Verification ---

if __name__ == "__main__":

    # Define a sample Pydantic model for configuration.
    class AppConfig(BaseModel):
        # This field is required.
        user_name: str
        # These fields have default values and specific types.
        retries: int = 3
        learning_rate: float = 0.01
        is_production: bool = False


    # --- Test 1: Successful Case ---
    print("--- 1. Successful Case: All arguments provided correctly ---")
    cli_args_success = [
        "user_name=Alice",
        "retries=5",
        "learning_rate=0.005",
        "is_production=true"  # Pydantic correctly handles string booleans
    ]
    print(f"Input arguments: {cli_args_success}")
    try:
        config = parse_args_to_model(AppConfig, cli_args_success)
        print("Successfully created config object:")
        print(config.model_dump_json(indent=2))
        assert config.retries == 5
        assert config.is_production is True
        print("✅ Test 1 Passed")
    except (ValueError, ValidationError) as e:
        print(f"❌ Test 1 FAILED: Caught unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50 + "\n")

    # --- Test 2: Missing a required field ---
    print("--- 2. Error Case: Missing a required field ---")
    cli_args_missing = ["retries=10"]
    print(f"Input arguments: {cli_args_missing}")
    try:
        parse_args_to_model(AppConfig, cli_args_missing)
        print("❌ Test 2 FAILED: Did not raise ValidationError as expected.", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print("✅ Correctly caught ValidationError for missing 'user_name'.")
        # Print a concise version of the error for clarity.
        print(f"   Details: {e.errors()[0]['msg']}")
        print("✅ Test 2 Passed")
    except ValueError as e:
        print(f"❌ Test 2 FAILED: Caught unexpected ValueError: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50 + "\n")

    # --- Test 3: Incorrect type for a field ---
    print("--- 3. Error Case: Incorrect type for a field ---")
    cli_args_bad_type = ["user_name=Bob", "retries=not-a-number"]
    print(f"Input arguments: {cli_args_bad_type}")
    try:
        parse_args_to_model(AppConfig, cli_args_bad_type)
        print("❌ Test 3 FAILED: Did not raise ValidationError as expected.", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print("✅ Correctly caught ValidationError for bad type in 'retries'.")
        print(f"   Details: {e.errors()[0]['msg']}")
        print("✅ Test 3 Passed")
    except ValueError as e:
        print(f"❌ Test 3 FAILED: Caught unexpected ValueError: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n🎉 All self-tests passed successfully!")

