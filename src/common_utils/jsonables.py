from typing import Any, TypeVar, Union, get_origin, get_args
from pydantic import BaseModel


def flatten_dict(d: dict[str, Any], parent_key: str = '', sep: str = '.') -> list[tuple[str, Any]]:
    """
    Flattens a nested dictionary.
    Assumes all keys are strings.

    Args:
        d: The dictionary to flatten.
        parent_key: The base key to use for a nested key.
        sep: The separator to use between nested keys.

    Returns:
        A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep))
        else:
            items.append((new_key, v))
    return items

K = TypeVar('K')
V = TypeVar('V')
def build_safe_dict(*lists_of_items: list[tuple[K, V]]) -> dict[K, V]:
    """
    Safely builds a single dictionary from multiple lists of (key, value) tuples.

    Args:
        *lists_of_items: A variable number of lists, where each list
                         contains (key, value) tuples.

    Returns:
        A  dictionary, where keys are unique across all lists.

    Raises:
        ValueError: If a key is found in multiple lists with a different value.
    """
    d = {}
    for items in lists_of_items:
        for k,v in items:
            if k in d and d[k] != v:
                raise ValueError(f"Duplicate key: {k} v1={d[k]} v2={v}")
            d[k]=v
    return d

def flat_dict(d:dict[str, dict[str, Any]]) -> dict[str, Any]:
    return build_safe_dict(flatten_dict(d))

####

def get_clean_type_name(type_hint: Any) -> str:
    """Generates a clean, readable name for a type hint."""
    origin = get_origin(type_hint)
    if origin is None:
        # For simple types like int, str, or a Pydantic model
        return getattr(type_hint, '__name__', str(type_hint))

    # For generic types like List[User] or Optional[str]
    args = get_args(type_hint)
    # Special case for Optional[T], which is Union[T, None]
    if origin is Union and len(args) == 2 and args[1] is type(None):
        return f"Optional[{get_clean_type_name(args[0])}]"

    arg_names = [get_clean_type_name(arg) for arg in args]
    return f"{origin.__name__}[{', '.join(arg_names)}]"


def get_model_schema_lines(model: type[BaseModel], level: int = 0) -> list[str]:
    """
    Recursively generates a list of strings representing the schema of a
    Pydantic model in a YAML-like format.

    Args:
        model (type[BaseModel]): The Pydantic model class to inspect.
        level (int): The current indentation level for recursive calls.

    Returns:
        list[str]: A list of formatted strings describing the model schema.
    """
    indent = "  " * level
    fields = model.model_fields
    schema_lines = []

    for field_name, field_info in fields.items():
        # Get the description, defaulting to an empty string if not provided
        description = field_info.description or ""

        # Get a clean name for the type annotation
        type_name = get_clean_type_name(field_info.annotation)

        # Append the current field's details to the list
        schema_lines.append(f"{indent}{field_name} ({type_name}): {description}")

        # --- Recursion Logic ---
        # Find the potential Pydantic models to recurse into,
        # even if they are inside List[Model] or Optional[Model].
        types_to_check = get_args(field_info.annotation) or [field_info.annotation]

        for sub_type in types_to_check:
            # Check if the subtype is itself a Pydantic model
            if isinstance(sub_type, type) and issubclass(sub_type, BaseModel):
                # If it is, extend the list with the lines from the recursive call
                schema_lines.extend(get_model_schema_lines(sub_type, level + 1))

    return schema_lines

import json, uuid
class CompactJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that formats objects/lists on a single line if they
    fit within a specified character width.
    """

    def __init__(self, *args, compact_width:int=100, separators:tuple[str,str]|None, **kwargs):
        super().__init__(*args, **kwargs)
        self._compact_width = compact_width
        self._placeholders = {}
        # Use separators=(',', ':') for the most compact representation
        self._separators=separators or (', ', ': ')

    def _is_compactable(self, obj):
        """Check if an object's compact representation is within the width limit."""
        if isinstance(obj, (dict, list)):
            compact_repr = json.dumps(obj, separators=self._separators)
            return len(compact_repr) <= self._compact_width
        return False

    def _find_and_replace(self, obj):
        """Recursively find compactable objects and replace them with a placeholder."""
        if self._is_compactable(obj):
            marker = f"__COMPACT_OBJ_{uuid.uuid4()}__"
            self._placeholders[f'"{marker}"'] = json.dumps(obj, separators=self._separators)
            return marker

        if isinstance(obj, dict):
            return {k: self._find_and_replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._find_and_replace(item) for item in obj]

        return obj

    def encode(self, obj):
        # Pre-process the object to replace compact lists/dicts with placeholders
        processed_obj = self._find_and_replace(obj)

        # Use the parent encoder to format the structure with placeholders
        formatted_json = super().encode(processed_obj)

        # Post-process the resulting string to substitute the placeholders back
        for marker, compact_obj in self._placeholders.items():
            formatted_json = formatted_json.replace(marker, compact_obj)

        return formatted_json


def dump_model_compact_json(model_or_list:BaseModel|list[BaseModel], width:int = 100, code_block:bool=False) -> str:
    """
    Dumps a Pydantic model or a list of models to a compact JSON string.

    - Any object/list that can be represented in a single line under `width`
      characters will be.
    - Otherwise, each field is on its own line (indent=2).
    """
    if isinstance(model_or_list, list):
        data_to_dump = [model.model_dump() for model in model_or_list]
    else:
        data_to_dump = model_or_list.model_dump()

    j = json.dumps(data_to_dump, cls=CompactJSONEncoder, indent=2, compact_width=width)
    if code_block:
        return f"```json\n{j}\n```"
    return j
