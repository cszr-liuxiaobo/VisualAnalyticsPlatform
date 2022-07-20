import glob
import os
import logging
from logging.handlers import TimedRotatingFileHandler

path = os.path.split(os.path.realpath(__file__))[0]
log_path = os.path.join(path, 'logs')  # 存放log文件的路径
class Logger(object):
    def __init__(self,filename,logger_name=None):
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)

        log_names = glob.glob(log_path+'*.*')
        if str(filename+'.log') not in log_names:
            self.log_file_name=str(filename+'.log')
        # self.log_file_name = 'test.log'  # 日志文件的名称
        self.backup_count = 10000  # 最多存放日志的数量
        # 日志输出级别
        self.console_output_level = 'WARNING'
        self.file_output_level = 'INFO'
        # 日志输出格式
        # self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def get_logger(self):
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        """在logger中添加日志句柄并返回，如果logger已有句柄，则直接返回"""
        if not self.logger.handlers:  # 避免重复日志
            console_handler = logging.StreamHandler()
            # console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            self.logger.addHandler(console_handler)

            # # 每天重新创建一个日志文件，最多保留backup_count份
            # file_handler = TimedRotatingFileHandler(filename=os.path.join(log_path, self.log_file_name), when='D',
            #                                         interval=1, backupCount=self.backup_count, delay=True,
            #                                         encoding='utf-8')
            file_handler = TimedRotatingFileHandler(filename=os.path.join(log_path, self.log_file_name), when='W6',
                                                    interval=60 * 60 * 24 * 7, backupCount=self.backup_count, delay=True,
                                                    encoding='utf-8')
            # file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self.logger.addHandler(file_handler)
        return self.logger



# 日志的调用格式
# import os
# from logprocess import log
#
# path = os.path.split(os.path.realpath(__file__))
#
# file_path = os.path.join(path[0], path[1])
# print(path)
# print(path[0])
# print(path[1])
# print(file_path)
#必须制定日志的名字，然后向此文件中导入信息
# logger = log.Logger(filename="wdqwde")
# logger.get_logger().info('eeecqwd')