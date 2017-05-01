import atexit
import os
import readline
import rlcompleter

# Register this only once
historyPath = os.path.join(os.getcwd(), '.pyhistory')
if os.path.exists(historyPath):
    readline.read_history_file(historyPath)

atexit.register(lambda: readline.write_history_file(historyPath))

# Generator to get text from the user. Also does some nice things like use
# a readline prompt and save user's history.
def get_input(prompt='> '):
    from six.moves import input

    try:
        while True:
            text = input(prompt)
            yield text
    except (KeyboardInterrupt, EOFError):
        print
        raise StopIteration()
    # We need to add a newline after the prompt when exiting.
    print
