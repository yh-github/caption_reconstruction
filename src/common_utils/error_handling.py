import traceback
from typing import Callable

from pydantic import BaseModel, ConfigDict, ValidationError


class UserFacingError(Exception):
    """
    A base class for all exceptions that are considered user-correctable
    and should not produce a full stack trace.
    """
    pass


def raise_if(condition, message:str|None=None, exception_builder:Callable[[], Exception]|None=None) -> None:
    if not condition:
        return

    if exception_builder is not None:
        raise exception_builder()

    raise AssertionError(message or "Assertion failed")


class ExceptionStr(BaseModel):
    model_config = ConfigDict(frozen=True, extra='allow')

    type: str
    message: str
    traceback: list[str]

    def __init__(self, e: Exception):
        if isinstance(e, ValidationError):
            super().__init__(
                type=type(e).__name__,
                message=f"{e.error_count()} error(s) when parsing {e.title}",
                traceback=traceback.format_tb(e.__traceback__),
                errors=e.errors(include_url=False, include_input=False, include_context=False)
            )
        else:
            super().__init__(
                type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_tb(e.__traceback__)
            )


######

from types import FrameType
import signal

def handle_ctrl_c():
    """
    Factory function to create a signal handler. This allows us to keep track
    of how many times Ctrl+C has been pressed.
    """
    press_count = [0]

    def signal_handler(sig:int, frame:FrameType|None):
        """
        This function is called when a SIGINT signal (Ctrl+C) is received.
        """
        press_count[0] += 1

        # On the first press, ask for confirmation.
        if press_count[0] != 1:
            raise KeyboardInterrupt

        if frame:
            print("\n" + "-"*40)
            print(f"Signal {signal.Signals(sig).name} received.")
            print(f"Interrupted at: File '{frame.f_code.co_filename}', Line {frame.f_lineno}")
            print("-" * 40)

        response = ''
        try:
            # handle interruptions during the prompt itself.
            response = input("\n\nAre you sure you want to exit? (y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Handle cases where the user presses Ctrl+D or Ctrl+C at the prompt.
            print("\nExit prompt cancelled. Continuing program.")
            press_count[0] = 0
            return # Exit the handler to continue the main loop

        # The logic to handle the response is now outside the try/except block.
        if response == 'y':
            print("Exiting program.")
            # This raise is no longer caught by the except block above.
            raise KeyboardInterrupt
        else:
            print("Continuing program...")
            press_count[0] = 0

    signal.signal(signal.SIGINT, signal_handler)
