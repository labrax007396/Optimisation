import pandas as pd
import onnx
import os,sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)

from data.data import Data



class InferModel:

    def __init__(self, model_onnx_location:onnx.onnx_ml_pb2.ModelProto,
                       data_file_location:str, 
                       model_file_options_location:str):
        

        self.model_onnx_location = model_onnx_location
        self.data_file_location  = data_file_location
        self.model_file_options_location = model_file_options_location



    def PredictWithOnnxModel(self):

        from onnxruntime import InferenceSession
        import numpy as np

        data_obj = Data(data_file_location = self.data_file_location, 
                            model_file_options_location = self.model_file_options_location)

        data_obj.ReadModelingOptions()
        data_obj.ReadCSVData()

        data   = data_obj.GetData()
        target = data.columns[0]

        X = data.drop(columns=[target])

        sess = InferenceSession(self.model_onnx_location)

            
        input_dict = dict()

        for f in X.columns:
            if X[f].dtypes == 'int64' or X[f].dtypes == 'float64':
                input_dict[f] = X[f].astype(np.float32).values.reshape(-1, 1)
            else:
                input_dict[f] = X[f].values.astype(str).reshape(-1, 1)

        pred_onnx = sess.run(None, input_dict)[0].flatten()
        self.pred_df   = pd.DataFrame(data={'Date':X.index,'python_onnx':pred_onnx})


    def SavePredictionToTSV(self,dir_resu:str,mangling:str):
        import os      
        self.pred_df['Date'] = self.pred_df['Date'].dt.strftime('%Y%m%d_%H%M%S') 
        file_resu = os.path.join(dir_resu, mangling+".tsv")
        self.pred_df.to_csv(file_resu,index=False,header=False,sep=" ")



    def SavePredictionToCSV(self,file_resu:str):
        import os      
        self.pred_df.to_csv(file_resu,index=False,header=False,sep=" ")
        self.pred_df['Date'] = self.pred_df['Date'] .dt.strftime('%Y/%m/%d %H:%M:%S')                           
        self.pred_df.to_csv(file_resu,index=False)