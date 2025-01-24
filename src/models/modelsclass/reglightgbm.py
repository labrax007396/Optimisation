import pandas as pd
from lightgbm import LGBMRegressor,Dataset
import lightgbm
from onnxmltools import convert_lightgbm
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes  # noqa
from onnxmltools import __version__ as oml_version
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
from skl2onnx import convert_sklearn, update_registered_converter

from ..ModeleGenerique import GenericModel

class RegressionLGBM(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):
        from sklearn.pipeline import Pipeline

        super().BuildModel()

        self.BuilPrepro()
        self.ModelSklearn = Pipeline([("preprocessor",self.prepro),("model", LGBMRegressor())])
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

        X_trs = self.prepro.fit_transform(X)
        
        study = optuna.create_study(direction='minimize')


        # calcul du nombre trials

        Nb_point_min = 5000
        Nb_point_max = 500000

        Nb_trial_min = 5
        Nb_trial_max = 50


        alpha_2 = (Nb_trial_max-Nb_trial_min)/(Nb_point_min-Nb_point_max)
        beta_2  = Nb_trial_max - alpha_2*Nb_point_min

        NBP = len(Y)

        Ntrial = round(alpha_2*NBP + beta_2)
        Ntrial = Nb_trial_min if Ntrial<Nb_trial_min else Ntrial
        Ntrial = Nb_trial_max if Ntrial>Nb_trial_max else Ntrial

        Ntrial = 100


        study.optimize(lambda trial: self.objective(trial, X_trs, Y), n_trials = Ntrial)
        self.best_params = study.best_trial.params


    def objective(self,trial,X_trs,Y):

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import warnings
        warnings.simplefilter('ignore')

        train_x, test_x, train_y, test_y = train_test_split(X_trs, Y, test_size=0.2,random_state=42)

        callbacks = [lightgbm.early_stopping(100, verbose=0), lightgbm.log_evaluation(period=0)]

        model = LGBMRegressor(verbosity = -1)


        # calcul du nombre d'estimateurs min

        Nb_point_min = 5000
        Nb_point_max = 500000
        Nest_min_au_max = 300
        Nest_min_au_min = 30

        alpha_1 = (Nest_min_au_max-Nest_min_au_min)/(Nb_point_max-Nb_point_min)
        beta_1  = Nest_min_au_max-alpha_1*Nb_point_max

        NBP = len(Y)

        Nest_min = round(alpha_1*NBP + beta_1)
        Nest_min = Nest_min_au_min if Nest_min<Nest_min_au_min else Nest_min
        Nest_min = Nest_min_au_max if Nest_min>Nest_min_au_max else Nest_min

        Nest_min = 20

        
        param = { 
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5,0.6,0.7]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.02,0.04,0.08,0.12]),
            'max_depth': trial.suggest_categorical('max_depth', [4,5,6]),
            'n_estimators':trial.suggest_int('n_estimators',Nest_min,500,10),
            'num_leaves' : trial.suggest_int('num_leaves',100,200,20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_categorical('subsample', [0.7,0.8,0.9])
        }

        fixed_hp =   {
                'metric': 'rmse', 
                'random_state': 48,
                'verbose': -1
            }

        for p, pv in fixed_hp.items():
            param[p] = pv

        model = LGBMRegressor(**param)

        

        model.fit(train_x,train_y,eval_set=[(test_x,test_y)],callbacks=callbacks)

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

    def skl2onnx_convert_lightgbm(self,scope, operator, container):
        options = scope.get_options(operator.raw_operator)
        if 'split' in options:
            if pv.Version(oml_version) < pv.Version('1.9.2'):
                warnings.warn(
                    "Option split was released in version 1.9.2 but %s is "
                    "installed. It will be ignored." % oml_version)
            operator.split = options['split']
        else:
            operator.split = None
        convert_lightgbm(scope, operator, container)
    

    def ConvertModelToOnnx(self):

        ''' Conversion du modèle au format onnx '''
        from sklearn.preprocessing import StandardScaler


        update_registered_converter(
            LGBMRegressor, 'LightGbmLGBMRegressor',
            calculate_linear_regressor_output_shapes,
            self.skl2onnx_convert_lightgbm,
            options={'split': None})

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

        self.ModelOnnx = to_onnx(self.ModelSklearn, initial_types=inputs,final_types=output,options={"split": 100},
                            target_opset={'': 13, 'ai.onnx.ml': 2})
        
        super().ChangeOnnxInputNames()


        self.IsLearned = True
        self.ModelOnnxCreated = True



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

        ModelOnnx = to_onnx(ModelSklearn, initial_types=inputs,final_types=output,options={"split": 100},
                            target_opset={'': 13, 'ai.onnx.ml': 2})
        
        ModelOnnx = GenericModel.ChangeOnnxInputNamesCls(ModelOnnx,data_obj)
        
        return ModelOnnx