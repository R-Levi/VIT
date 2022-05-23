"""LOGGER"""
import os
class logger():
    def __init__(self,path,log_name):
        self.path = path
        self.log_name = log_name
        self.log_dir = os.path.join(self.path,self.log_name)
        if(os.path.exists(self.log_dir)==False):
            os.makedirs(self.log_dir)
    def write(self,text):
        with open(os.path.join(self.log_dir,f'{self.log_name}'), "a") as f:
            f.write(text+'\n')
            f.close()
    def print(self,text):
        print(text)
