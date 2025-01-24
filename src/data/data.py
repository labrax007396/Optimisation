
import pandas as pd
from datetime import datetime

class Data:


    def __init__(self, data_file_location:str, 
                 config_location:str):

        
        self.data_file_location = data_file_location
        self.config_location = config_location 
        self.data = pd.DataFrame
        self.options = dict()



    def ReadModelingOptions(self):

        """
            Lecture du fichier .json contenant les options
        
        """

        import commentjson

        with open(self.config_location, encoding='utf-8') as file:
        
            self.options = commentjson.load(file)


    def ReadCSVData(self):

        """
            Lecture du fichier de données
        
        """

        # Lecture de l'entête

        var_modelisee = self.options['y_config']['name']
        facteurs      = [d for d in self.options['f_configs'].keys()]
        index_name    = self.options['dataformat']['index_name']
        col_to_import = [index_name] + [var_modelisee] + facteurs

        self.data = pd.read_csv(self.data_file_location,
                              sep     = self.options['dataformat']['sep'],
                              decimal = self.options['dataformat']['decimal'],
                              usecols = col_to_import
                              )
        
        if self.options['dataformat']['index_type'] == "date":
            date_format = self.options['dataformat']['date_format']
            self.data[index_name] = self.data[index_name].apply(lambda x: datetime.strptime(str(x), date_format))
            self.data.set_index(index_name,inplace=True)
            debut = self.options['debut']
            fin   = self.options['fin']
            self.data = self.data[(self.data.index>=debut) & (self.data.index<=fin)]


        for col in self.options['f_configs'].keys():
            type_c = self.options['f_configs'][col]['type']
            if type_c == 'num':
                self.data[col] = self.data[col].astype(float)
            elif type_c == 'disc':
                self.data[col] = self.data[col].astype(str)
        
        self.DeleteOutliers()

    def DeleteOutliers(self):

        for col in self.options['f_configs'].keys():
            type_c = self.options['f_configs'][col]['type']
            if type_c == 'num':
                vmin = self.options['f_configs'][col]['vmin']
                vmax = self.options['f_configs'][col]['vmax']
                self.data = self.data[ (self.data[col]>=vmin) & (self.data[col]<=vmax) ]
            if type_c == 'cat':
                modalites = self.options['f_configs'][col]['modalites']
                if type(modalites) == list:
                    self.data[self.data[col].isin(modalites)]


    def GetModelOptions(self):
        return self.options
    

    def GetData(self):
        return self.data