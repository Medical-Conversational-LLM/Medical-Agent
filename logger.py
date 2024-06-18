import logging
import datetime


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fmt = '%(asctime)s | %(levelname)s | %(message)s'

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(CustomFormatter(fmt))

    logger.addHandler(stdout_handler)

    return logger


def get_profiler(logger):
    return Profiler(logger)


class Profiler:

    last_profile = 0
    profile_table = dict()
    profile_memory = []
    last_profile_id = "start"

    def __init__(self, logger):
        self.logger = logger

    def start(self):
        self.last_profile = datetime.datetime.now()
        self.last_profile_id = "start"

    def add(self, name):
        self.profile_table[name] = datetime.datetime.now()

    def reset(self):
        self.last_profile = 0
        self.profile_table = dict()
        self.profile_memory = []
        self.last_profile_id = "start"

        self.start()
        
    def profile(self, name, profile_id=None):
        now = datetime.datetime.now()

        since = self.last_profile
        if profile_id in self.profile_table and self.profile_table[profile_id] is not None:
            since = self.profile_table[profile_id]

        diff = now - since

        self.logger.info("PROFILE {name}: {diff}".format(name=name, diff=diff))

        self.profile_memory.append(
            [name, diff, profile_id if profile_id is not None else self.last_profile_id])

        self.last_profile = now
        self.last_profile_id = name

    def dump(self):
        self.pretty_print_table(self.profile_memory)

    # @url https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    def pretty_print_table(self, rows, line_between_rows=True):
        """
        Example Output
        ┌──────┬─────────────┬────┬───────┐
        │ True │ short       │ 77 │ catty │
        ├──────┼─────────────┼────┼───────┤
        │ 36   │ long phrase │ 9  │ dog   │
        ├──────┼─────────────┼────┼───────┤
        │ 8    │ medium      │ 3  │ zebra │
        └──────┴─────────────┴────┴───────┘
        """

        # find the max length of each column
        max_col_lens = list(
            map(max, zip(*[(len(str(cell)) for cell in row) for row in rows])))

        # print the table's top border
        print('┌' + '┬'.join('─' * (n + 2) for n in max_col_lens) + '┐')

        rows_separator = '├' + '┼'.join('─' * (n + 2)
                                        for n in max_col_lens) + '┤'

        row_fstring = ' │ '.join("{: <%s}" % n for n in max_col_lens)

        for i, row in enumerate(rows):
            print('│', row_fstring.format(*map(str, row)), '│')

            if line_between_rows and i < len(rows) - 1:
                print(rows_separator)

        print('└' + '┴'.join('─' * (n + 2) for n in max_col_lens) + '┘')


logger = get_logger()
profiler = get_profiler(logger)
profiler.start()
logger.info("...START...")
