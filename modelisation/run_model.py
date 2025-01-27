import sys
import os


from src.data.data import Data
from src.models.ModelManager import ModelRunner



class Model:

    def __init__(self, data_file_location:str, config_location:str, model_pkl_file:str):
        self.data_file_location = data_file_location
        self.config_location    = config_location
        self.model_pkl_file     = model_pkl_file


    def run(self):

        self.data_obj = Data(data_file_location = self.data_file_location, 
                     config_location = self.config_location)
        
        self.data_obj.ReadModelingOptions()
        self.data_obj.ReadData()


        self.model_manager_obj = ModelRunner(data_obj=self.data_obj)
        self.model_manager_obj.BuildAndTrain()
        self.model_manager_obj.model_obj.ExportToPkl(model_pkl_file = self.model_pkl_file)