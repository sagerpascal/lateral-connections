from typing import Optional, Any
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
    WARNING = "⚠️"
    EXCEPTION = "🚨"
    START = "💥"
    LOGS = "📊"


def _print(
        obj: Any,
        symbol: str,
        color: str,
        title: Optional[str] = None,
        symbol_border: bool = False,
        pretty_format: bool = True
):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param symbol: Symbol to print.
    :param color: Color to print.
    :param title: Optional title.
    :param symbol_border: If True, print symbol border.
    :param pretty_format: If True, use pprint.pformat to print the object.
    """
    if pretty_format:
        obj = pformat(obj, depth=3)
    if title is not None:
        title = f"{title}\n"
    txt = f"{symbol}\t{color}{Color.BOLD}{title if title is not None else ''}{Color.END}{color}{obj}{Color.END}"
    txt = txt.replace('\n', f'\n{symbol}\t')
    if symbol_border:
        symbol_border = symbol * 50
        print(f"{symbol_border}\n{txt}\n{symbol_border}")
    else:
        print(txt)


def _print_info(obj: Any, symbol: str, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param symbol: Symbol to print.
    :param title: Optional title.
    """
    _print(obj, symbol, Color.YELLOW, title)


def print_start(obj: Any, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param obj: Text to print.
    :param title: Optional title.
    """
    _print(obj, Symbol.START, Color.BLUE, title, symbol_border=True)

def print_logs(logs: Any, title: Optional[str] = None):
    """
    Print info text in yellow.
    :param logs: Text to print.
    :param title: Optional title.
    """
    res = ""
    for k, v in logs.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        res += f"\t{k:15s}:\t{v}\n"
    _print(res, Symbol.LOGS, Color.BLUE, title, pretty_format=False)

def print_exception(obj: Exception):
    """
    Print warning text in red.
    :param obj: Text to print.
    :param title: Optional title.
    """
    _print(obj, Symbol.WARNING, Color.RED, "EXCEPTION:\n", symbol_border=True)


def print_warn(obj: Any, title: Optional[str] = None):
    """
    Print warning text in red.
    :param obj: Text to print.
    :param title: Optional title.
    """
    _print(obj, Symbol.WARNING, Color.RED, title)


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
