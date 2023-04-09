from typing import Optional, Any
import pprint
from pprint import pformat


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Symbol:
    DATA = "📂"
    CONFIG = "📝"


def _print(obj: Any, symbol: str, color: str, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param symbol: Symbol to print.
    :param color: Color to print.
    :param title: Optional title.
    """
    obj_str = pformat(obj, depth=3).replace('\n', f'\n{symbol}\t')
    txt = f"{symbol}\t{color}{Color.BOLD}{title}{Color.END}{color}{obj_str}{Color.END}"
    print(txt)


def _print_info(obj: Any, symbol: str, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param symbol: Symbol to print.
    :param title: Optional title.
    """
    _print(obj, symbol, Color.YELLOW, title)


def print_info_data(obj: Any, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param title: Optional title.
    """
    _print_info(obj, Symbol.DATA, title)

def print_info_config(obj: Any, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param title: Optional title.
    """
    _print_info(obj, Symbol.DATA, title)