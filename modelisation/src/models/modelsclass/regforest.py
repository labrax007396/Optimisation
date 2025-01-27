import sys
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn
sys.path.append('../models')
from sklearn.ensemble import RandomForestRegressor


from ..ModeleGenerique import GenericModel


class RegressionRandForestReg(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):
        from sklearn.pipeline import Pipeline

        super().BuildModel()


        self.BuilPrepro()
        self.ModelSklearn = Pipeline([("preprocessor",self.prepro),("model", RandomForestRegressor())])
        self.IsBuild = True


    def Learn(self, model_options=dict, data=pd.DataFrame):

        self.FindHyperParams(data=data)
        super().LearnWithHyperParams()
  

    def FindHyperParams(self,data=pd.DataFrame):

        import optuna
        import warnings
        from tqdm import TqdmExperimentalWarning

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        data.dropna(inplace=True)
        Y  = data[data.columns[0]]
        X  = data.drop(columns=data.columns[0])

    
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X, Y), n_trials=30)
        params  = study.best_trial.params
        self.best_params = {k.replace("model__",""): v for k,v in params.items()}


    def objective(self,trial,X,Y):

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import warnings
        warnings.simplefilter('ignore')

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2,random_state=42)

            
        param = { 
            'model__n_estimators':trial.suggest_int('model__n_estimators',10,500,10),
            'model__max_depth': trial.suggest_categorical('model__max_depth', [4,5,6]),
            'model__min_samples_split': trial.suggest_categorical('model__min_samples_split', [2, 5, 10]),
            'model__min_samples_leaf': trial.suggest_categorical('model__min_samples_leaf', [1, 2, 4]),
        }

        model = self.ModelSklearn

        model.set_params(**param)

        model.fit(train_x,train_y)

        preds_train = model.predict(train_x)  
        rmse_train = mean_squared_error(train_y, preds_train,squared=False)
        preds_test = model.predict(test_x)
        rmse_test = mean_squared_error(test_y, preds_test,squared=False)

        alpha_overfit = 0.4
        score_final = alpha_overfit*rmse_train + (1-alpha_overfit)*np.abs(rmse_train-rmse_test)
        
        return score_final


    def GetTrainData(self):
        return self.data_obj.GetData()
    
    def CreateJsonResults(self):
        super().CreateJsonResults()


    def ConvertModelToOnnx(self):

        ''' Conversion du modèle au format onnx '''


        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        from skl2onnx.common.data_types import Int64TensorType


        data = self.data_obj.GetData()
        target = data.columns[0]
        X = data.drop(target,axis=1)

        inputs = []
        for k, v in zip(X.columns, X.dtypes):
            if v == 'int64':
                t = Int64TensorType([None, 1])
            elif v == 'float64':
                t = FloatTensorType([None, 1])
            else:
                t = StringTensorType([None, 1])
            inputs.append((k, t))

        output = [('target',FloatTensorType([None, 1]))]


        

        self.ModelOnnx = to_onnx(self.ModelSklearn, initial_types=inputs,final_types=output,
                            target_opset={'': 13, 'ai.onnx.ml': 2})

        super().ChangeOnnxInputNames()
                

    @classmethod
    def ConvertModelToOnnxCls(cls,ModelSklearn,data_obj):

        ''' Conversion du modèle au format onnx '''


        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        from skl2onnx.common.data_types import Int64TensorType

        data = data_obj.GetData()
        target = data.columns[0]
        X = data.drop(target,axis=1)

        inputs = []
        for k, v in zip(X.columns, X.dtypes):
            if v == 'int64':
                t = Int64TensorType([None, 1])
            elif v == 'float64':
                t = FloatTensorType([None, 1])
            else:
                t = StringTensorType([None, 1])
            inputs.append((k, t))

        output = [('target',FloatTensorType([None, 1]))]

        ModelOnnx = to_onnx(ModelSklearn, initial_types=inputs,final_types=output,
                            target_opset={'': 13, 'ai.onnx.ml': 2})
        
        ModelOnnx = GenericModel.ChangeOnnxInputNamesCls(ModelOnnx,data_obj)
                
        
        return ModelOnnx