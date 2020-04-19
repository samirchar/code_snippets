import subprocess
import time
import os
from nltk.parse.corenlp import CoreNLPParser
from retry import retry


class serverConnection:
    
    def __init__(self):
        self.command = ["java", "-mx4g","-cp",'"*"', "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", \
                        "-preload","tokenize,ssplit,pos,lemma,parse,depparse",\
                        "-status_port","9000","-port","9000","-timeout","15000"]
        self.cwd = os.path.dirname(__file__)
    
    
    def start_server(self):
        self.process = subprocess.Popen(self.command,cwd = self.cwd)
        self.test_connection()

    @retry(delay = 3)
    def test_connection(self):
        st = CoreNLPParser()
        print('testing connection...')
        list(st.tokenize("test"))
        print('Server ready!')

    def kill_server(self):
        self.process.kill()

    

