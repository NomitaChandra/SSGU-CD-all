# Copyright (c) 2018 Borna Bešić
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import subprocess
import re
import os

class GENIATagger:
    '''GENIA tagger wrapper
    
    Spawns a tagger subprocess in the background.
    The communication is done through stdin & stdout pipes.
    Messages that executable writes to stderr are ignored.
    '''

    def __init__(self, executable_path):
        '''
        Constructs a GENIA tagger wrapper object

        Arguments:
            - executable_path : str
              Path to the compiled GENIA tagger executable
        '''

        self.executable_path = os.path.abspath(executable_path)
        directory, executable = os.path.split(self.executable_path)
        self.process = subprocess.Popen(
            os.path.join(".", executable),
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            cwd = directory
        )

    def tag(self, text):
        '''
        A generator function that tags the specified text.

        Arguments:
            - text : str
              Text that will be tagged
        Yields:
            (word, base form, POS tag, chunk, named entity)
            tuple for each word in the specified text.
        '''

        text_lf = text + os.linesep

        self.process.stdin.write(text_lf.encode("utf-8"))
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline().decode("utf-8").strip()
            if line == "":
                break

            row = tuple(re.split("\s+", line))
            if len(row) == 5: # word, base form, POS tag, chunk, named entity
                yield row

    def stop(self):
        ''' Terminates the subprocess running in the backgroud. '''

        self.process.terminate()
        return self.process.wait()

    def __enter__(self):
        ''' Enables the object to be used with the 'with' statement '''

        return self

    def __exit__(self, type, value, traceback):
        ''' Calls self.stop() after exiting the 'with' block '''

        self.stop()
