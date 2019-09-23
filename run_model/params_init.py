import configparser


class params:
    def __init__(self, section):
        self.section = section
        self.cf = configparser.ConfigParser()
        self.cf.read('resource/config.conf')
        self.RAW_DATA_PATH = self.get_param('RAW_DATA_PATH')
        self.RAW_DATA_FILENAME = self.get_param('RAW_DATA_FILENAME')


        self.process_input_script = self.get_param('process_input_script')



    def get_param(self, p_name):
        return self.cf.get(self.section, p_name)
