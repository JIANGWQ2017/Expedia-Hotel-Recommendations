from params_init import *
import sys
import os

def processing(params):
    cmd = 'python %s %s %s' % (params.process_input_script, params.RAW_DATA_PATH, params.RAW_DATA_FILENAME)
    print(cmd)
    os.system(cmd)



if __name__ == '__main__':
    section = sys.argv[1]
    param = params(section)
    processing(param)
