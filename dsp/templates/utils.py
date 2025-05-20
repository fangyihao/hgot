from typing import Union, Optional


def passages2text(passages: Union[str, list, tuple]) -> str:
    """Formats the given one or more passages into a single structured string."""
    if isinstance(passages, str):
        return passages

    assert type(passages) in [list, tuple]

    # if len(passages) == 1:
    #    return f"«{passages[0]}»"

    return "\n".join([f"[{idx+1}] «{txt}»" for idx, txt in enumerate(passages)])

def hints2text(hints: Union[str, list, tuple]) -> str:
    """Formats the given one or more hints into a single structured string."""
    if isinstance(hints, str):
        return hints

    assert type(hints) in [list, tuple]

    if len(hints) == 1:
        return f"{hints[0]}"

    return "\n".join([f"{txt}" for txt in hints])


def format_answers(answers: Union[str, list]) -> Optional[str]:
    """Parses the given answers and returns the appropriate answer string.

    Args:
        answers (Union[str, list]): The answers to parse.

    Raises:
        ValueError: when instance is of type list and has no answers
        ValueError: when is not of type list or str

    Returns:
        _type_: Optiona[str]
    """
    if isinstance(answers, list):
        if len(answers) >= 1:
            return str(answers[0]).strip()
        if len(answers) == 0:
            raise ValueError("No answers found")
    elif isinstance(answers, str):
        return answers
    else:
        raise ValueError(f"Unable to parse answers of type {type(answers)}")
