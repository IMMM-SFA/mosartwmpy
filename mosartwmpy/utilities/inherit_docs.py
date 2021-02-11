import types

def inherit_docs(cls: object) -> object:
    """A class decorator to automatically inherit docstrings from parents.

    Args:
        cls (object): The class that needs docstrings

    Returns:
        object: The class with docstrings inherited from parents
    """

    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls