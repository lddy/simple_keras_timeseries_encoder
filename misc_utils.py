from datetime import datetime
import os, pathlib
class bclr:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

leadpadding = ' '*17

class simple_echo_logger:
    def __init__(self, file, append=False):
        self.file = file
        self.append = append
        self.last_msg = None
        self.started = False
        self.started_date = None
        self.started_time = None
        self.ended_time = None

    def start(self):
        s = datetime.now()
        self.started_date = s.strftime('%Y-%m-%d')
        self.started_time = s.strftime('%H:%M:%S.%f')
        msg = f'{self.started_time}: Started logging, date: {self.started_date}'
        self.last_msg = msg
        print(f'{msg}, to: {self.file}')
        if self.append:
            with open(self.file, 'a') as f:
                f.write(f'\n{msg}\n\n')
        else:
            with open(self.file, 'w') as f:
                f.write(f'{msg}\n\n')
        self.started = True

    def out(self, m:str, level = -1, style = ''):
        if not self.started:
            self.start()
        et = datetime.now().strftime('%H:%M:%S.%f')
        clro = ''
        clrc = ''
        match level:
            case 0:
                clro = bclr.OKGREEN
            case 1:
                clro = bclr.WARNING
            case 2:
                clro = bclr.FAIL
        match style:
            case 'header':
                clro += bclr.HEADER
            case 'bold':
                clro = bclr.BOLD
            case 'underline':
                clro = bclr.UNDERLINE
        if clro != '':
            clrc = bclr.ENDC

        m_padded = f'\n{leadpadding}'.join(m.split('\n')) if  m[-1] != '\n'\
                                                          else f'\n{leadpadding}'.join(m.split('\n')[:-1]) + '\n'
        print(f'{et}: {clro}{m_padded}{clrc}')
        with open(self.file, 'a') as f:
            f.write(f'{et}: {m_padded}\n')

    def stop(self):
        if not self.started:
            return
        et = datetime.now().strftime('%H:%M:%S.%f')
        message = f'\n{et}: Ended logging\n'
        print(print(f'{message} to: {self.file}'))
        with open(self.file, 'a') as f:
            f.write(message)

        self.started = False


