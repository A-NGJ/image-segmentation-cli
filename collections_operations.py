from typing import (
    List,
)


def check_strings_in_list(list1: List[str], list2: List[str]) -> bool:
    """
    Check if all strings from list1 are contained in at least one string in list2.
    """
    return all(any(s in elem for elem in list2) for s in list1)
