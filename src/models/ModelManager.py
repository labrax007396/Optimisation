import sys
import os

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)

from data.data import Data
from figure.GenererFigure import CreateFigure
from models.modelsclass.regresslin import RegressionOLS
from models.modelsclass.reglightgbm import RegressionLGBM
from models.modelsclass.regforest import RegressionRandForestReg
from models.modelsclass.regpoly import RegressionPoly
from models.modelsclass.regxgboost import RegressionXGBoost
from models.modelsclass.regexponentielle import RegressionExp
from models.modelsclass.regpuissance import RegressionPuissance
from reporting.RapportPdf import RapportPdf
from interpretation.shapeley import Interpreteur



class ModelRunner:

    def __init__(self, data_obj:Data, pdfreport:str):
        self.data_obj = data_obj
        self.Status = True
        self.pdfreport = pdfreport

       

    def Build(self):

        model_options = self.data_obj.GetModelOptions()

        if model_options['model_type'] == "RegressionLineaire":
            self.model_obj = RegressionOLS(data_obj=self.data_obj)

        elif model_options['model_type'] == "RegressionLgbm":
            self.model_obj = RegressionLGBM(data_obj=self.data_obj)

        elif model_options['model_type'] == "RegressionRandomForest":
            self.model_obj = RegressionRandForestReg(data_obj=self.data_obj)   

        elif model_options['model_type'] == "RegressionXGBoost":
            self.model_obj = RegressionXGBoost(data_obj=self.data_obj)   

        elif model_options['model_type'] == "RegressionExp":
            self.model_obj = RegressionExp(data_obj=self.data_obj)    

        elif model_options['model_type'] == "RegressionPuissance":
            self.model_obj = RegressionExp(data_obj=self.data_obj)            
        else:
            assert False, model_options['model_type']+ " Non pris en compte"


    def BuildAndTrain(self):

        model_options = self.data_obj.GetModelOptions()

        if model_options['model_type'] == "RegressionLineaire":
            self.model_obj = RegressionOLS(data_obj=self.data_obj)

        elif model_options['model_type'] == "RegressionLgbm":
            self.model_obj = RegressionLGBM(data_obj=self.data_obj)

        elif model_options['model_type'] == "RegressionRandomForest":
            self.model_obj = RegressionRandForestReg(data_obj=self.data_obj)   

        elif model_options['model_type'] == "RegressionPolynomiale2":
            self.model_obj = RegressionPoly(data_obj=self.data_obj)   

        elif model_options['model_type'] == "RegressionXGBoost":
            self.model_obj = RegressionXGBoost(data_obj=self.data_obj)  

        elif model_options['model_type'] == "RegressionExp":
            self.model_obj = RegressionExp(data_obj=self.data_obj)  
           
        elif model_options['model_type'] == "RegressionPuissance":
            self.model_obj = RegressionPuissance(data_obj=self.data_obj)            

  
        else:
            assert False, model_options['model_type']+ " Non pris en compte"

        
        if self.Status:  # Si le type de modèle est pris en charge on continu le processus

            # Construction du modèle
            self.model_obj.BuildModel()
            if self.model_obj.IsBuild == False:
                self.Status = False
        
        if self.Status:        
            # Apprentissage du modèle
            self.model_obj.LearnModel()
            if self.model_obj.IsLearned == False:
                self.Status = False                

        if self.Status:
            # Creation des résultats au format json
            self.model_obj.CreateJsonResults()
            if not self.model_obj.ResuModelCreated:
                self.Status = False   

        if self.Status:
            # Convertion du modèle au format Onnx
            self.model_obj.ConvertModelToOnnx()         
            if not self.model_obj.ModelOnnxCreated:  
                self.Status = False  

        if self.Status:            
            # Sauvegarde des résultats
            self.model_obj.SaveModelingResults()
            if not self.model_obj.FileSaveCorrectly:
                self.Status = False  

        
        self.interpreteur_obj = Interpreteur(model_obj   = self.model_obj,
                                             data_obj    = self.data_obj,
                                             rapport_obj = self.model_obj.rapport_obj)

        self.interpreteur_obj.Run()


        if self.pdfreport=='Yes':
            # Creation des figures
            if self.Status:                
                fig_obj = CreateFigure(model_obj = self.model_obj)
                fig_obj.CreateAllFigures()
                if not fig_obj.FigureCreated:
                    self.Status = False  
            
                # Creation du rapport modélisation
            if self.Status: 
                rapport_pdf = RapportPdf(model_obj = self.model_obj, figures_dir = fig_obj.fig_rep)
                rapport_pdf.CreateReport()
                if not rapport_pdf.RepportCreated:
                    self.Status = False  
        

