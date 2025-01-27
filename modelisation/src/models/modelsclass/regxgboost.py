import sys
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn
sys.path.append('../models')
from xgboost import XGBRegressor
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster

from ..ModeleGenerique import GenericModel


class RegressionXGBoost(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):
        from sklearn.pipeline import Pipeline

        super().BuildModel()

        self.BuilPrepro()
        self.ModelSklearn = Pipeline([("preprocessor",self.prepro),("model", XGBRegressor())])
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

        self.MessageLogger.write_msg('info',"identification des hypers paramètres")
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
            'model__max_depth': trial.suggest_int('model__max_depth', 1, 10),
            'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 1.0),
            'model__n_estimators': trial.suggest_int('model__n_estimators', 50, 1000),
            'model__min_child_weight': trial.suggest_int('model__min_child_weight', 1, 10),
            'model__gamma': trial.suggest_float('model__gamma', 0.01, 1.0),
            'model__subsample': trial.suggest_float('model__subsample', 0.01, 1.0),
            'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.01, 1.0),
            'model__reg_alpha': trial.suggest_float('model__reg_alpha', 0.01, 1.0),
            'model__reg_lambda': trial.suggest_float('model__reg_lambda', 0.01, 1.0),
            'model__random_state': trial.suggest_int('model__random_state', 1, 1000)
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

        update_registered_converter(
        XGBRegressor,
        "XGBoostXGBRegressor",
        calculate_linear_regressor_output_shapes,
        convert_xgboost,
        options={'split': None})

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
                
        
        self.ModelOnnxCreated = True