import pandas as pd
from ..ModeleGenerique import GenericModel

class RegressionOLS(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression

        super().BuildModel()

        self.BuilPrepro()
        self.ModelSklearn = Pipeline([("preprocessor",self.prepro),("model", LinearRegression())])
        self.IsBuild = True

        

    def Learn(self, model_options=dict, data=pd.DataFrame):
        
        from sklearn.model_selection import train_test_split
        data.dropna(inplace=True)
        target = data.columns[0]

        model_options = self.data_obj.GetModelOptions()

        if model_options['used_part_of_data_for_validation']:
            self.train_data_set, self.test_data_set = train_test_split(data, test_size=0.33, random_state=42)
            train_x = self.train_data_set.drop(target,axis=1)
            train_y = self.train_data_set[target]
            self.ModelSklearn.fit(train_x,train_y)
            self.IsLearned = True
        else:
            self.train_data_set = data.copy()
            train_x = self.train_data_set.drop(target,axis=1)
            train_y = self.train_data_set[target]
            self.ModelSklearn.fit(train_x,train_y)
            self.IsLearned = True

    def GetTrainData(self):
        return self.data_obj.GetData()
    