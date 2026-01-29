import os
import sys



class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(root_dir, pipeline_id):
    log_dir = os.path.join(root_dir, "results/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{pipeline_id}.log")
    
    
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout 
    
    print(f"save to: {log_file}")
    return log_file