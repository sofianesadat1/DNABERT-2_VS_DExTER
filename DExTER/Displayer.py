import os

from datetime import datetime, timedelta
import logging
import time

from threading import Thread

import sys


class Displayer(Thread):

    def __init__(self, broker, terminated):
        super().__init__()
        self.broker = broker
        self.terminated = terminated
        # self.progress = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃']
        self.progress = ['┤', '┘', '┴', '└', '├', '┌', '┬', '┐']
        self.start_time = datetime.now()
        self.datetime_zero = self.start_time - self.start_time
        self.pid = None

    def stop(self):
        self.terminated = True

    def run(self):
        ts = 1
        i = 0
        count = 0
        self.pid = int(os.popen('echo $PPID', 'r').read())
        columns = 80
        endl = '\n'
        if sys.stdout.isatty():
            rows, columns = os.popen('stty size', 'r').read().split()
            endl = '\r'
        while not self.terminated:
            nb_finished = len(self.broker.domain_obj_done)
            nb_remaining = len(self.broker.domain_obj_waiting)
            current_time = datetime.now()
            delta = current_time - self.start_time
            if nb_remaining == 0:
                count += 1
                if count > 5:
                    self.terminated = True
            else:
                count = 0
            msg_progress = '\033[1;31m' + self.progress[i] + '\033[0m'
            msg = msg_progress + ' domains remaining: ' + str(nb_remaining) + ' ' + msg_progress + ' domains done: ' + str(nb_finished) + ' ' + msg_progress + ' elapsed time: ' + '{}'.format(delta) + ' ' + msg_progress
            msg = msg.center(int(columns))
            print(msg, end=endl)
            i += 1
            i %= len(self.progress)
            time.sleep(ts)
        self.pid = None
