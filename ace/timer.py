import logging
from datetime import datetime


def str_of_timedelta(delta):
    mm, ss = divmod(delta.seconds, 60)
    hh, mm = divmod(mm, 60)
    hh += delta.days * 24
    return '{}:{:02}:{:02}'.format(hh, mm, ss)


class _HandlerEnabler:
    def __init__(self, handler: logging.FileHandler):
        self.value = False
        handler.addFilter(lambda record: self.value)

    def __enter__(self):
        self.value = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.value = False


class Timer:
    def __init__(self, logfile_path, section_name):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logfile_path)
        self.logger.addHandler(handler)
        self.log_to_file = _HandlerEnabler(handler)
        self.timer_start = self.section_start = datetime.now()
        self.section_name = section_name

        self._section_start()

    def __call__(self, section_name):
        self._section_end(section_name)
        self._section_start()

    def close(self):
        self._section_end('')

    def _section_start(self):
        self.logger.info(self.section_name)

    def _section_end(self, next_section_name):
        now = datetime.now()
        with self.log_to_file:
            self.logger.info('{} {} {}'.format(
                str_of_timedelta(now - self.timer_start),
                str_of_timedelta(now - self.section_start),
                self.section_name))
        self.section_start = now
        self.section_name = next_section_name
