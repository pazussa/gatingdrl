import argparse
import inspect
from typing import Any, Callable, List, Type, get_origin, get_args


def custom_list_type(item_type: Type) -> Callable[[str], List[Any]]:
    """
    Generates a function to convert a comma-separated string into a list of a specific type.

    Parameters
    ----------
    item_type : Type
        The type of the items in the resulting list.

    Returns
    -------
    Callable[[str], List[Any]]
        A function that converts a string to a list of the specified item type.
    """
    def convert(s: str) -> List[Any]:
        return [item_type(item) for item in s.split(',')]
    return convert




def execute_from_command_line(func: Callable):
    """
    Executes a given function using arguments provided from the command line.

    This function uses the inspect module to determine the required and optional
    arguments of the `func` function and their annotations. It then creates an
    ArgumentParser object and adds the function's arguments to it.

    The command-line arguments are expected to be provided in the format
    `--arg_name arg_value`. The function arguments can be either required or
    optional. If an optional argument is not provided, its default value from
    the function definition is used.

    After parsing the command-line arguments, this function calls `func` with
    the parsed arguments.

    Parameters
    ----------
    func : Callable
        The function to be executed. This function can have any number of
        required or optional arguments.

    Raises
    ------
    argparse.ArgumentError
        If a required argument is not provided in the command line.
    """
    # Get the signature of the function
    sig = inspect.signature(func)

    # Create the argument parser
    parser = argparse.ArgumentParser(description=func.__doc__)

    # Add arguments to the parser
    for name, param in sig.parameters.items():
        # Determine the type of the argument
        if param.annotation is not param.empty:
            arg_type = param.annotation
            # Check if the annotation is a generic type
            origin_type = get_origin(arg_type)
            if origin_type is list:
                # Get the inner type of the list
                inner_type = get_args(arg_type)[0]
                arg_type = custom_list_type(inner_type)
        else:
            arg_type = str

        if param.default is param.empty:  # it's a required argument
            parser.add_argument('--' + name, required=True, type=arg_type)
        else:  # it's an optional argument, use default value from function definition
            parser.add_argument('--' + name, default=param.default, type=arg_type)

    args = parser.parse_args()

    # Convert args to a dictionary
    args_dict = vars(args)

    # Call the function with the arguments
    return func(**args_dict)
