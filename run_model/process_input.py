from params_init import *
import sys
import os

def processing(params):
    cmd = 'python ' + params.process_input_script
    print(cmd)
    os.system(cmd)



if __name__ == '__main__':
    section = sys.argv[1]
    param = params(section)
    processing(param)
