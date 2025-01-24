import pandas as pd

import os,sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)

from data.data import Data
from reporting.RapportJson import RapportJson



class GenericModel:

    def __init__(self, data_obj:Data):
        self.data_obj = data_obj
        self.test_data_set  = None
        self.train_data_set = None
        self.ModelSklearn   = None
        self.ModelOnnx      = None
        self.resu_model     = dict()
        self.ResuModelCreated = True
        self.ModelOnnxCreated = True
        self.FileSaveCorrectly = True


    def BuildModel(self):
        pass

    def BuilPrepro(self):

        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        data   = self.data_obj.GetData()
        target = data.columns[0]

        num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
        cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

        if len(cat_feat)>0:
            num_prepro  = StandardScaler()
            cat_prepro  = OneHotEncoder(handle_unknown='ignore')
            self.prepro =  ColumnTransformer([('num',num_prepro,num_feat),('cat',cat_prepro, cat_feat)])
        else:
            num_prepro = StandardScaler()
            self.prepro =  ColumnTransformer([('num',num_prepro,num_feat)])




    def LearnModel(self):
        ''' Apprentissage do modèle '''

        model_options = self.data_obj.GetModelOptions()
        data          = self.data_obj.GetData()

        print("apprentissage du modèle")

        self.Learn(model_options=model_options, data=data)  
        self.IsLearned = True
               

    def Learn(self, model_options=dict, data=pd.DataFrame):
        pass

    def LearnWithHyperParams(self):

        from sklearn.model_selection import train_test_split

        if self.best_params:
            hp_param_set = {f"model__{key}": value for key, value in self.best_params.items()}
            self.ModelSklearn.set_params(**hp_param_set)
            
            data = self.data_obj.GetData()
            data.dropna(inplace=True)
            target = data.columns[0]
            model_options = self.data_obj.GetModelOptions()

            if model_options['used_part_of_data_for_validation']:
                self.train_data_set, self.test_data_set = train_test_split(data, test_size=0.33, random_state=42)
                train_x = self.train_data_set.drop(target,axis=1)
                train_y = self.train_data_set[target]
                self.ModelSklearn.fit(train_x,train_y)
            else:
                self.train_data_set = data.copy()
                train_x = self.train_data_set.drop(target,axis=1)
                train_y = self.train_data_set[target]
                self.ModelSklearn.fit(train_x,train_y)
        

    def CreateModelFormula(self):
        self.formula = ""

    def CreateJsonResults(self):

        import datetime
        from sklearn.metrics import r2_score

        print("Création des résultats au format json")


        data = self.data_obj.GetData()
        self.CreateFormula()
        model_options = self.data_obj.GetModelOptions()

        y = data[data.columns[0]]
        X = data.drop(columns=data.columns[0])
        y_pred = self.ModelSklearn.predict(X)

        resu_model = dict()

        resu_model['Date']   = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        resu_model['Nombre de points']   = data.shape[0]
        resu_model['Variable modélisée'] = data.columns[0]
        resu_model['Type de modèle'] = model_options['model_type']
        resu_model['formula_string'] = self.formula
        resu_model['Statistiques'] = dict()
        resu_model['Statistiques']['R2'] = r2_score(y.values,y_pred)
        #resu_model['TimeStamps'] = [dt.strftime("%d/%m/%Y %H:%M:%S")  for dt in data.index]
        #resu_model['Mesure'] = y.values.tolist()
        #resu_model['Modele'] = y_pred.tolist()
        self.resu_model = resu_model



    def ConvertModelToOnnx(self):

        ''' Conversion du modèle au format onnx '''


        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        from skl2onnx.common.data_types import Int64TensorType

        model_options = self.data_obj.GetModelOptions()


        data = self.data_obj.GetData()

        if model_options['model_type'] == "RegressionLineaire":
            list_fact_cat = [t for t in data.dtypes if t == 'object']
            if len(list_fact_cat) == 0:
                self.ModelOnnx = None
            else:

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
                
                self.ChangeOnnxInputNames()
                
        elif model_options['model_type'] == "RegressionExp":
            self.ModelOnnx = None

        elif model_options['model_type'] == "RegressionPuissance":
            self.ModelOnnx = None

        else:

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

            self.ChangeOnnxInputNames()


    def SaveModelingResults(self):

        import json
        from onnx.onnx_pb import StringStringEntryProto
        import os

        
        with open("resu_modele.json",'w', encoding='utf-8') as fi:
            json.dump(self.resu_model,fi,indent=4, ensure_ascii=False)



        self.rapport_obj = RapportJson(model_obj=self,data_obj=self.data_obj)
        self.rapport_obj.CreateReport()
        modelreport_json = self.rapport_obj.GetJsonReportAsString()
        modelreport_dict = json.loads(modelreport_json)

        # Création des résultats des contributions des facteurs


        #interpreteur_obj = Interpreteur(model_obj = self,
        #                                data_obj  = self.data_obj,
        #                                rapport_obj = self.rapport_obj,
        #                                mgl_interp = "ParentMglInterp")


        #interpreteur_obj.Run()
        #print(interpreteur_obj.dico_contrib_formula)


        if self.ModelOnnx:

            header = self.data_obj.GetHeader()
            target = header['Tagname'][0]
            mgltarget = header['ParentScopeMangling'][0]
            frequency = header['TagInfoFrequency'][0]
            onnx_model_name = mgltarget+"."+target+"_"+frequency+".onnx"
            self.ModelOnnx.metadata_props.append(StringStringEntryProto(key="ReportModel", value = modelreport_json))
            
            with open(onnx_model_name, "wb") as f:
                f.write(self.ModelOnnx.SerializeToString()) 


        else:
            onnx_model_name = ""

        modelreport_dict['onnx_model_name'] = onnx_model_name
        
        with open("rapport_modelisation.json",'w', encoding='utf-8') as fi:
            json.dump(modelreport_dict,fi,indent=4, ensure_ascii=False)


    def ChangeOnnxInputNames(self):

        dict_map_tagname_to_init = self.data_obj.dict_map_tagname_to_init

        for i in range(len(self.ModelOnnx.graph.node)):
            for j in range(len(self.ModelOnnx.graph.node[i].input)):
                node_name = self.ModelOnnx.graph.node[i].input[j]
                if node_name in dict_map_tagname_to_init.keys():
                    self.ModelOnnx.graph.node[i].input[j] = dict_map_tagname_to_init[node_name]

        for i in range(len(self.ModelOnnx.graph.input)):
            input_name = self.ModelOnnx.graph.input[i].name
            if input_name in dict_map_tagname_to_init.keys():
                self.ModelOnnx.graph.input[i].name = dict_map_tagname_to_init[input_name]
                


    @classmethod
    def ChangeOnnxInputNamesCls(cls,ModelOnnx,data_obj):

        dict_map_tagname_to_init = data_obj.dict_map_tagname_to_init

        for i in range(len(ModelOnnx.graph.node)):
            for j in range(len(ModelOnnx.graph.node[i].input)):
                node_name = ModelOnnx.graph.node[i].input[j]
                if node_name in dict_map_tagname_to_init.keys():
                    ModelOnnx.graph.node[i].input[j] = dict_map_tagname_to_init[node_name]

        for i in range(len(ModelOnnx.graph.input)):
            input_name = ModelOnnx.graph.input[i].name
            if input_name in dict_map_tagname_to_init.keys():
                ModelOnnx.graph.input[i].name = dict_map_tagname_to_init[input_name]

        return ModelOnnx




    def GetModel(self):
        return self.ModelSklearn
    
    def GetTrainData(self):
        return self.train_data_set

    def GetTestData(self):
        return self.test_data_set

    def GetResuModel(self):
        return self.resu_model

    def CreateFormula(self):
        self.formula = ""

    def Export_Prediction_To_Phd(self,filename:str):


        from onnxruntime import InferenceSession
        import pandas as pd
        import numpy as np

        data   = self.data_obj.GetData()
        target = data.columns[0]

        X = data.drop(columns=[target])

        sess = InferenceSession(self.ModelOnnx.SerializeToString())

            
        input_dict = dict()

        for f in X.columns:
            if X[f].dtypes == 'int64' or X[f].dtypes == 'float64':
                input_dict[f] = X[f].astype(np.float32).values.reshape(-1, 1)
            else:
                input_dict[f] = X[f].values.astype(str).reshape(-1, 1)

        pred_onnx = sess.run(None, input_dict)[0].flatten()
        pred_pkl  = self.ModelSklearn.predict(X)


        predictions = pd.DataFrame(data={'Date':X.index,'python_onnx':pred_onnx,'python_pkl':pred_pkl})
        predictions['Date'] = predictions['Date'].dt.strftime('%Y/%m/%d %H:%M:%S')                           

        predictions.to_csv(filename,index=False)

        
    def ReduceCorpusSize(self):
        from random import seed
        from random import sample

        NMAX = 100000
        # seed random number generator
        seed(1)

        train_data = self.GetTrainData()
        test_data  = self.GetTestData()


        if train_data.shape[0] > NMAX:
            # prepare a sequence
            sequence = [i for i in range(train_data.shape[0])]
            # select a subset without replacement
            subset = sample(sequence, NMAX)
            self.train_data_set = train_data.iloc[subset]


        if test_data is not None:
            if test_data.shape[0] > NMAX:
                # prepare a sequence
                sequence = [i for i in range(test_data.shape[0])]
                # select a subset without replacement
                subset = sample(sequence, NMAX)
                self.test_data_set = test_data.iloc[subset]
