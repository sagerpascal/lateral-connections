from typing import Optional
import pprint


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
    DATA = "📄"


pp = pprint.PrettyPrinter(indent=4, depth=6)


def _print(str_: str, symbol: str, color: str, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param str_: Text to print.
    :param symbol: Symbol to print.
    :param color: Color to print.
    :param title: Optional title.
    """
    txt = f"{symbol}\t{color}{Color.BOLD}{title}{Color.END}{color}{str_}{Color.END}"
    pp.pprint(txt)


def _print_info(str_: str, symbol: str, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param str_: Text to print.
    :param symbol: Symbol to print.
    :param title: Optional title.
    """
    _print(str_, symbol, Color.YELLOW, title)


def print_info_data(str_: str, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param str_: Text to print.
    :param title: Optional title.
    """
    _print_info(str_, Symbol.DATA, title)
