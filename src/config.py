"""

"""
from ConfigParser import NoOptionError

import os
import ConfigParser

class Config(object):
    def __init__(self):
        self.CONFIG_FILENAME = 'pic4turtle.cfg'
        self.rawConfig = ConfigParser.RawConfigParser()

        if not os.path.exists(self.CONFIG_FILENAME):
            # set config
            self.rawConfig.add_section('GlobalConf')
            self.rawConfig.set('GlobalConf', 'AppPath', '/home/ubuntu/pic4turtle')
            self.rawConfig.set('GlobalConf', 'ModelPath', '.')
            self.rawConfig.set('GlobalConf', 'DataPath', 'data')
            self.rawConfig.set('GlobalConf', 'UseGPU', 'true')

            self.rawConfig.set('GlobalConf', 'RPCHost', 'localhost')
            self.rawConfig.set('GlobalConf', 'RPCPort', '8888')

            self.rawConfig.set('GlobalConf', 'Unclassified', -1)

            # Writing our configuration file
            with open('pic4turtle.cfg', 'wb') as configfile:
                self.rawConfig.write(configfile)
        else:
            self.rawConfig.read(self.CONFIG_FILENAME)

    def __getattr__(self, item):
        try:
            return self.rawConfig.get('GlobalConf', item)
        except NoOptionError:
            raise AttributeError('Wrong config property')

conf = Config()
