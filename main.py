# ---------------------------- Approach 1: Single hardcoded question ---------------------------- #
# Uncomment this block (and comment out Approach 2 below) to invoke the data agent once
# with a hardcoded HumanMessage.

# from agents.data_agent import data_agent
# from langchain_core.messages import HumanMessage
#
# if __name__ == "__main__":
#     response = data_agent.invoke(
#         {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
#          "route_response": ""}
#     )
#
#     print(response)


# ---------------------------- Approach 2: Interactive terminal ---------------------------- #
# Run `python main.py` and chat with the data agent. Type 'q' to quit.

import sys
import threading
import itertools
import time

# Colors only when printing to a real terminal (not when output is piped to a file)
USE_COLOR = sys.stdout.isatty()

def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

CYAN = "38;5;45"
ORANGE = "38;5;209"
GREY = "38;5;245"
GREEN = "38;5;114"
RED = "38;5;203"
BOLD = "1"

# "PIPEQL" on top, "AGENT" below (printed in a different color in print_banner)
LOGO = r"""
 ██████╗ ██╗██████╗ ███████╗ ██████╗ ██╗
 ██╔══██╗██║██╔══██╗██╔════╝██╔═══██╗██║
 ██████╔╝██║██████╔╝█████╗  ██║   ██║██║
 ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║▄▄ ██║██║
 ██║     ██║██║     ███████╗╚██████╔╝███████╗
 ╚═╝     ╚═╝╚═╝     ╚══════╝ ╚══▀▀═╝ ╚══════╝"""

LOGO_AGENT = r"""
  █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
 ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
 ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
"""

TAGLINE = "Ask in plain English: AI agents build your ETL pipelines and query your data with SQL."

QUIT_COMMANDS = {"q", "quit", "exit"}


def print_banner():
    print(color(LOGO, CYAN))
    print(color(LOGO_AGENT, ORANGE))
    print(color(" PipeQL Agent", BOLD) + color("  ·  " + TAGLINE, GREY))
    print()
    # Each line is a list of (text, color) parts. Padding is calculated on the plain
    # text, so the right border lines up no matter which parts are colored.
    lines = [
        [("Try:", None)],
        [("  • Which 3 users have taken the most rides?", GREY)],
        [("  • Extract https://pokeapi.co/api/v2/pokemon", GREY)],
        [("    and save it to data/extract as csv", GREY)],
        [],
        [("Type your question and press Enter.  ", None), ("q", ORANGE), (" to quit.", None)],
    ]
    width = 62
    print(color(" ╭" + "─" * (width + 2) + "╮", GREY))
    for parts in lines:
        plain = "".join(text for text, _ in parts)
        styled = "".join(color(text, c) if c else text for text, c in parts)
        print(color(" │ ", GREY) + styled + " " * (width - len(plain)) + color(" │", GREY))
    print(color(" ╰" + "─" * (width + 2) + "╯", GREY))
    print()


class Spinner:
    """
    Shows an animated 'Thinking...' line while the agent is working.

    Why threading is needed here:

    Python runs code one line at a time. When we call `data_agent.invoke(...)`, the
    program waits on that line until the agent is done (it calls the LLM, runs SQL,
    etc.), which can take 10-30 seconds. While it waits, nothing else in the program runs.

    WITHOUT a thread: there is no way to update the spinner during that wait. We could
    draw it once before the call and once after, but it would not move in between. The
    screen would look frozen, and the user couldn't tell whether the program is working
    or stuck. We also can't put the animation loop before or after the agent call. The
    loop would either block the agent from starting, or start only after the agent has
    already finished.

    WITH a thread: the spinner animation runs in a second, separate line of execution
    (a "background thread"). The main thread does the slow agent call while the background
    thread redraws the spinner every 0.1 seconds. Both make progress at the same time.
    This works even though Python allows only one thread to run Python code at a time (the GIL):
    while the main thread waits for the network (LLM API, database), it gives up its turn,
    so the spinner thread gets to run. Even during pure Python work (like importing the
    agents), Python switches between threads every few milliseconds, so the spinner keeps moving.

    How the two threads talk to each other:
    - `threading.Event` (`self._stop`) is a simple on/off flag that both threads can safely
      read and write. When the agent is done, the main thread sets it (`self._stop.set()`),
      and the spinner loop sees it and stops.
    - `join()` makes the main thread wait until the spinner thread has really finished and
      cleared its line. Without it, the spinner could print one last frame on top of the answer.
    - `daemon=True` means this thread will never keep the program alive by itself. If the
      program exits (e.g. on an error), the spinner thread is stopped automatically
      instead of spinning forever.

    It is used as a context manager (`with Spinner(): ...`), so the spinner starts when
    the `with` block begins and stops when it ends, even if an error happens inside it.
    """

    def __init__(self, text: str = "Thinking..."):
        self.text = text
        self._stop = threading.Event()  # On/off flag: "stop spinning now"
        self._thread = threading.Thread(target=self._spin, daemon=True)  # Runs _spin() in the background

    def _spin(self):
        # This method runs in the background thread. It keeps drawing the next frame
        # until the main thread sets the stop flag.
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r {color(frame, ORANGE)} {color(self.text, GREY)}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        # Animate only in a real terminal; piped output would fill with junk frames.
        if USE_COLOR:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        # The `with` block is done (the agent answered, or an error happened).
        self._stop.set()             # Tell the spinner thread to stop
        if self._thread.is_alive():
            self._thread.join()      # Wait until it has stopped and cleared its line


def ask_agent(data_agent, question: str):
    """Sends one question to the data agent and returns (route, answer)."""
    from langchain_core.messages import HumanMessage
    from utils.logger import get_logger, block

    logger = get_logger("main")

    # A clear divider in the log file, so each question is easy to find
    logger.info("=" * 90)
    logger.info(f"NEW REQUEST: {question}")
    start = time.perf_counter()

    response = data_agent.invoke(
        {"messages": [HumanMessage(content=question)],
         "route_response": ""}
    )

    answer = response["messages"][-1].text
    logger.info(f"ANSWER (took {time.perf_counter() - start:.2f}s):{block(answer)}")
    return response["route_response"], answer


def run_interactive():
    print_banner()

    # Imported after the banner because loading the agents takes a few seconds
    with Spinner("Loading agents..."):
        from agents.data_agent import data_agent

    while True:
        try:
            question = input(color(" ❯ ", ORANGE)).strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print()
            break

        if not question:
            continue
        if question.lower() in QUIT_COMMANDS:
            break

        try:
            with Spinner():
                route, answer = ask_agent(data_agent, question)
        except KeyboardInterrupt:
            print(color("\n Cancelled.\n", GREY))
            continue
        except Exception as e:  # Keep the session alive if one question fails
            print(color(f"\n ✗ Error: {e}\n", RED))
            continue

        agent_name = "SQL Analyst" if route == "sql" else "ETL Agent"
        print(color(f"\n ● {agent_name}", GREEN))
        for line in answer.splitlines():
            print(f"   {line}")
        print()

    print(color(" Bye! 👋", GREY))


if __name__ == "__main__":
    run_interactive()
