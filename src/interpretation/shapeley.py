import os,sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)

from data.data import Data
from models.ModeleGenerique import GenericModel
from models.modelsclass.reglightgbm import RegressionLGBM
from models.modelsclass.regforest import RegressionRandForestReg
from reporting.RapportJson import RapportJson

class Interpreteur:

    def __init__(self, model_obj:GenericModel,data_obj:Data, rapport_obj:RapportJson) -> None:
        self.model_obj = model_obj
        self.data_obj  = data_obj
        self.rapport_obj  = rapport_obj

        self.PARENT_MGL_INTERPRETATION = "mgl_interp"
        self.OFFSET = "offset"

    def Run(self):

        
        if self.data_obj.GetModelOptions()['model_type'] == 'RegressionLgbm':
            self.ComputeShapeleyValues()
            self.SelectFeatures()
            self.LearnSklearnShapeModels()
            self.ConvertModelsToOnnx()
            self.SaveOnnxModelsToFile()
            #self.CreateDicoInterpretation()
        

        if self.data_obj.GetModelOptions()['model_type'] == 'RegressionLineaire':
            self.ComputeShapeleyValues()
            #self.CreateDicoFormuleRegLinFrequence()
            #self.CreateDicoFormuleOthersFrequences()
            #self.ExportToFormulaFile()

    def RunAndCreateFormulaInterp(self):


        if self.data_obj.GetModelOptions()['model_type'] == 'RegressionLineaire':
            self.ComputeShapeleyValues()
            self.CreateDicoFormuleRegLinFrequence()
            self.CreateDicoFormuleOthersFrequences()
            self.ExportToFormulaFile()        


    def ComputeShapeleyValues(self):

        import scipy
        import numpy as np
        import pandas as pd
        import shap

        data = self.data_obj.GetData()
        model = self.model_obj.GetModel()

        target = data.columns[0]
        train_x = data.drop(target,axis=1)
        train_y = data[target]


        x_transformed = model.steps[0][1].transform(train_x)
        if type(x_transformed) == scipy.sparse._csr.csr_matrix:
            x_transformed = np.asarray(x_transformed.todense())


        f_name = model.steps[0][1].get_feature_names_out()
        f_name = [f.replace('num__','').replace('cat__','') for f in f_name]

        regressor = model.steps[1][1]

        model_option = self.data_obj.GetModelOptions()

        if model_option['model_type'] == 'RegressionLineaire':
            delta_x = x_transformed - x_transformed.mean(axis=0)
            self.svals = delta_x*regressor.coef_
            self.df_x = pd.DataFrame(index=train_x.index,data=x_transformed,columns=f_name)


        elif model_option['model_type'] == 'RegressionLgbm':


            self.df_x = pd.DataFrame(index=train_x.index,data=x_transformed,columns=f_name)

            explainer = shap.TreeExplainer(regressor, 
                                        model_output='raw', 
                                        feature_perturbation='interventional' 
                                        )


            self.svals = explainer.shap_values(self.df_x, y=train_y)


        elif model_option['model_type'] == 'RegressionRandomForest':



            self.df_x = pd.DataFrame(index=train_x.index,data=x_transformed,columns=f_name)

            explainer = shap.TreeExplainer(regressor, 
                                        model_output='raw', 
                                        feature_perturbation='interventional' 
                                        )


            self.svals = explainer.shap_values(self.df_x, y=train_y)

        

        else:

            self.svals = None

    def SelectFeatures(self):

        import pandas as pd
        df_svals = pd.DataFrame(index=self.df_x.index,data=self.svals,columns=self.df_x.columns)
        serie_sommeshape = df_svals.apply(abs).sum(axis=0)
        serie_poids = serie_sommeshape/serie_sommeshape.sum()
        serie_poids.sort_values(ascending=False,inplace=True)
        SEUIL_FEAT_KEPT = 0.005

        serie_poids_kept = serie_poids[serie_poids>SEUIL_FEAT_KEPT]
        self.feat_kept = serie_poids_kept.index.to_list()

    def LearnSklearnShapeModels(self):

        from sklearn.base import clone
        import pandas as pd

        self.dico_model_shape_sklearn = dict()
        data = self.data_obj.GetData().copy(deep=True)
        model = self.model_obj.GetModel()
        target = data.columns[0]
        df_svals = pd.DataFrame(index=self.df_x.index,data=self.svals,columns=self.df_x.columns)

        self.df_svals = df_svals

        for s_v in self.feat_kept:
            model_shape_sklearn = clone(model)
            data[target] = df_svals[s_v]
            X = data.drop(columns=[target])
            y = data[target]
            model_shape_sklearn.fit(X,y)
            self.dico_model_shape_sklearn[s_v] = model_shape_sklearn

    def ConvertModelsToOnnx(self):

        self.dico_model_shape_onnx = dict()
        data = self.data_obj.GetData()

        model_option = self.data_obj.GetModelOptions()

        if model_option['model_type'] == 'RegressionLgbm':
            for s_v, models in self.dico_model_shape_sklearn.items():
                self.dico_model_shape_onnx[s_v] = RegressionLGBM.ConvertModelToOnnxCls(models,self.data_obj)

        if model_option['model_type'] == 'RegressionRandomForest':
            for s_v, models in self.dico_model_shape_sklearn.items():
                self.dico_model_shape_onnx[s_v] = RegressionRandForestReg.ConvertModelToOnnxCls(models,data)





    def SaveOnnxModelsToFile(self):

        import tempfile

        self.dico_model_shape_onnx_name = dict()

        #self.formula_rep = tempfile.mkdtemp(prefix="models_",suffix="_onnx")
        #print(self.formula_rep)

        
        for s_v, onnx_model in self.dico_model_shape_onnx.items():
            nom_contrib = 'Contrib_'+s_v
            nom_model_onnx = '.'.join([self.PARENT_MGL_INTERPRETATION,nom_contrib,'onnx'])
            self.dico_model_shape_onnx_name[s_v] = nom_model_onnx
            #loc_model_onnx = os.path.join(self.formula_rep ,nom_model_onnx)
            with open(nom_model_onnx, "wb") as f:
                f.write(onnx_model.SerializeToString())



    def ReadInterpConfig(self,file_config:str):

        import commentjson
    
        with open(file_config, encoding='utf-8') as file:
            self.config = commentjson.load(file)



    def CreateDicoFormuleOthersFrequences(self):

        if self.config["IsConsoSpec"]:
            self.CreateDicoFormuleConsoSpecOthersFrequences()
        else:
            self.CreateDicoFormuleConsoOthersFrequences()


    def CreateDicoFormuleConsoSpecOthersFrequences(self):

        header = self.data_obj.GetHeader()
        IPE_freq = header['TagInfoFrequency'][0]
        IPE_unit = header['Unit'][0]
        IPE_name = header['Tagname'][0]
        IPE_desc = header['Description'][0]
        IPE_ShortMangling = ".".join([header['ParentScopeMangling'][0],header['Tagname'][0]])

        model_options = self.data_obj.GetModelOptions()
        ParentMglInterp = self.config['ParentMglInterp']
        offset = self.config['offset']
   

        #### Calcul de la Production hors et dans le domaine de l'IPE à la fréquence du modèle ####


        IPE_Model_FullMangling = "["+ ".".join([model_options['ipe_config']['tag_mgl'],'NormalizationRecalc']) + "]"

        Desc_Prod_HorsDomaine     = self.config["ShortMglProd"].split(".")[-1] + "_Hors_Domaine"
        TagName_Prod_HorsDomaine  = Desc_Prod_HorsDomaine
        Desc_Prod_Domaine         = self.config["ShortMglProd"].split(".")[-1] + "_Domaine"
        TagName_Prod_Domaine      = Desc_Prod_Domaine

        Prod_FullMangling         = "["+ ".".join([self.config["ShortMglProd"],IPE_freq]) + "]"
        Prod_HD_ShortMangling     = "["+ ".".join([self.config['ParentMglInterp'],TagName_Prod_HorsDomaine]) + "]"
        Prod_HD_FullMangling      = "["+ ".".join([self.config['ParentMglInterp'],TagName_Prod_HorsDomaine,IPE_freq]) + "]"
        Prod_D_FullMangling       = "["+ ".".join([self.config['ParentMglInterp'],TagName_Prod_Domaine,IPE_freq]) + "]"

        
        formule = Prod_FullMangling + '.KeepIf(' + IPE_Model_FullMangling + ').FillWith("' + self.dico_map_freq_long_to_short[IPE_freq] + '","' + offset + '",0.0)'

        formula_prod_d = { 'ScopeMangling': ParentMglInterp,
                    'Description': Desc_Prod_Domaine,
                    'Identifiant': TagName_Prod_Domaine,
                    'Frequency':IPE_freq,
                    'Role':'Data',
                    'DataType':'Numeric',
                    'Unit':'',
                    'Formule': formule
            }
        
        full_mgl    = ".".join([formula_prod_d['ScopeMangling'],formula_prod_d['Identifiant'],formula_prod_d['Frequency']])
        self.dico_contrib_formula[full_mgl] = formula_prod_d

         
        formule = Prod_FullMangling + "-" + Prod_D_FullMangling 

        formula_prod_hd = { 'ScopeMangling': ParentMglInterp,
                    'Description': Desc_Prod_HorsDomaine,
                    'Identifiant': TagName_Prod_HorsDomaine,
                    'Frequency':IPE_freq,
                    'Role':'Data',
                    'DataType':'Numeric',
                    'Unit':'',
                    'Formule': formule
            }

        
        full_mgl    = ".".join([formula_prod_hd['ScopeMangling'],formula_prod_hd['Identifiant'],formula_prod_hd['Frequency']])
        self.dico_contrib_formula[full_mgl] = formula_prod_hd   
   

        #### Calcul de la consommation hors du domaine de l'IPE à la fréquence du modèle ####

        FullMglConso = "["+ ".".join([self.config['ShortMglConso'],IPE_freq])+ "]"

        formule_conso_hd = FullMglConso + "-" + FullMglConso + '.KeepIf(' + IPE_Model_FullMangling + ').FillWith("' + self.dico_map_freq_long_to_short[IPE_freq] + '","' + offset + '",0.0)'

        formula_conso_hd = { 'ScopeMangling': ParentMglInterp,
                    'Description': 'Conso Hors Domaine',
                    'Identifiant': 'Conso_Hors_Domaine',
                    'Frequency':IPE_freq,
                    'Role':'Data',
                    'DataType':'Numeric',
                    'Unit':'',
                    'Formule': formule_conso_hd
            }
        
        full_mgl    = ".".join([formula_conso_hd['ScopeMangling'],formula_conso_hd['Identifiant'],formula_conso_hd['Frequency']])
        Conso_HD_FullMangling = "[" + full_mgl + "]"
        self.dico_contrib_formula[full_mgl] = formula_conso_hd

        #### Calcul des formules pour les fréquences > ####

        for freq_long in self.list_freq_long:
            if self.dico_map_freq_to_int[freq_long] > self.dico_map_freq_to_int[IPE_freq]:
                freq_short = self.dico_map_freq_long_to_short[freq_long]

                ### IPE moyenne de référence ###

                ref_fbase_full_mgl = "[" + ".".join([ParentMglInterp,IPE_name + "_VAL_MOY",IPE_freq]) + "]"
                ref_full_mgl       = ".".join([ParentMglInterp,IPE_name + "_VAL_MOY",freq_long])
                formule = ref_fbase_full_mgl + '.Average(' + '"' + freq_short + '","' + offset + '")'

                formula = {'ScopeMangling': ParentMglInterp,
                        'Description': IPE_desc + " Valeur Moyenne",
                        'Identifiant': IPE_name + "_VAL_MOY",
                        'Frequency':freq_long,
                        'Role':'Data',
                        'DataType':'Numeric',
                        'Unit':IPE_unit,
                        'Formule': formule
                        }     
                self.dico_contrib_formula[ref_full_mgl] = formula

                ### Facteurs ###

                list_contrib_full_mgl = list()

                for fact in self.list_fact:

                    Prod_FullMangling     = '['+'.'.join([self.config["ShortMglProd"],freq_long])+']'
                    Contrib_FullMangling  = '['+'.'.join([ParentMglInterp,'Contrib_'+fact,freq_long])+']'
                    Contrib_BaseFullMangling  = '['+'.'.join([ParentMglInterp,'Contrib_'+fact,IPE_freq])+']'

                    somme_prod_contrib = '(' + Prod_D_FullMangling + '*' + Contrib_BaseFullMangling + ').Sum(' + '"' + freq_short + '","' + offset + '")'

                    formule = somme_prod_contrib + "/" +  Prod_FullMangling


                    formula = { 'ScopeMangling': ParentMglInterp,
                                'Description': 'Contrib '+fact,
                                'Identifiant': 'Contrib_'+fact,
                                'Frequency':freq_long,
                                'Role':'Data',
                                'DataType':'Numeric',
                                'Unit':IPE_unit,
                                'Formule': formule
                        }
                    

                    self.dico_contrib_formula[Contrib_FullMangling] = formula
                    list_contrib_full_mgl.append(Contrib_FullMangling)

                ### contribution à production nulle ###
                
                Prod_FullMangling     = '['+'.'.join([self.config["ShortMglProd"],freq_long])+']'
            
                formule = Conso_HD_FullMangling + '.Sum(' + '"' + freq_short + '","' + offset + '")/' + Prod_FullMangling

                formula = { 'ScopeMangling': ParentMglInterp,
                            'Description': 'Contrib Prod Hors Domaine',
                            'Identifiant': 'Contrib_Prod_Hors_Domaine',
                            'Frequency':freq_long,
                            'Role':'Data',
                            'DataType':'Numeric',
                            'Unit':IPE_unit,
                            'Formule': formule
                    }
                
                full_mgl    = ".".join([formula['ScopeMangling'],formula['Identifiant'],formula['Frequency']])

                self.dico_contrib_formula[full_mgl] = formula
                list_contrib_full_mgl.append('['+full_mgl+']')


                ### Autres contributions ###

                IPE_FullMangling = "["+ ".".join([IPE_ShortMangling,freq_long]) + "]"
                formule = IPE_FullMangling + "-(" + "+".join(['['+ref_full_mgl+']'] + list_contrib_full_mgl) + ")"
                formula = { 'ScopeMangling': ParentMglInterp,
                            'Description': 'Contrib Autre',
                            'Identifiant': 'Contrib_Autre',
                            'Frequency':freq_long,
                            'Role':'Data',
                            'DataType':'Numeric',
                            'Unit':IPE_unit,
                            'Formule': formule
                          }
                full_mgl    = ".".join([formula['ScopeMangling'],formula['Identifiant'],formula['Frequency']])

                self.dico_contrib_formula[full_mgl] = formula




    def CreateDicoFormuleConsoOthersFrequences(self):
        pass

    def ExportToFormulaFile(self):
        import pandas as pd
        import csv


        dico_formula = self.dico_contrib_formula
        ScopeMangling = [d['ScopeMangling'] for d in dico_formula.values()]
        Description = [d['Description'] for d in dico_formula.values()]
        Identifiant = [d['Identifiant'] for d in dico_formula.values()]
        Frequency = [d['Frequency'] for d in dico_formula.values()]
        Role = [d['Role'] for d in dico_formula.values()]
        DataType = [d['DataType'] for d in dico_formula.values()]
        Unit = [d['Unit'] for d in dico_formula.values()]
        Formule = [d['Formule'] for d in dico_formula.values()]

        df_formula = pd.DataFrame(

        data = {'ScopeMangling':ScopeMangling,
                'Description':Description,
                'Identifiant':Identifiant,
                'Fréquence':Frequency,
                'Rôles':Role,
                'Type de données':DataType,
                'Unités':Unit,
                'Formule':Formule
                }

        )
        


        df_formula.to_csv("contributions.formula",sep=';',index=False, quoting=csv.QUOTE_NONE)



    def CreateDicoFormuleRegLinFrequence(self):


        import json
        import tempfile
        import pandas as pd
        import csv
        import commentjson
        import os

        self.list_freq_int   = [0,1,2,3,4,5,6,7]
        self.list_freq_long  = ['Second','ProductUnit','Minute','Hour','ShiftWork','Day','Week','Month']
        self.list_freq_short = ["S","ProductUnit","m","h","8h","1d","1w","1M"]
        self.dico_map_freq_to_int = {fl:fi for fl,fi in zip(self.list_freq_long,self.list_freq_int)}
        self.dico_map_freq_long_to_short = {fl:fs for fl,fs in zip(self.list_freq_long,self.list_freq_short)}


        ParentMglInterp = self.config['ParentMglInterp']
                
        model_options = self.data_obj.GetModelOptions()
        header = self.data_obj.GetHeader()
        list_tag_mgl = [fc['tag_mgl'] for fc in model_options['factor_configs']]
        dico_map_fact_mgl = dict()
        df_data = self.data_obj.GetData()

    
        fact_num = [f for f in df_data.columns[1:] if df_data[f].dtype=='float64']
        fact_cat = [f for f in df_data.columns[1:] if df_data[f].dtype=='object']
        self.list_fact = fact_num + fact_cat
     

        serie_mean_fact = df_data[fact_num].mean()
        dico_coef = self.model_obj.dico_coef
        for fn in fact_num:
            dico_map_fact_mgl[fn] = '[' + [mg for mg in list_tag_mgl if fn in mg][0] + ']'
          
        
        ###### dictionnaire des formules ######

        self.dico_contrib_formula = dict()
        list_tag_mgl_contrib = list()


            # Formule valeur moyenne de l'IPE sur référence

        formula_ref = self.FormuleIPERef()
        full_mgl    = ".".join([formula_ref['ScopeMangling'],formula_ref['Identifiant'],formula_ref['Frequency']])
        self.dico_contrib_formula[full_mgl] = formula_ref

            # Formule des contributions

        IPE_freq = header['TagInfoFrequency'][0]
        IPE_unit = header['Unit'][0]
        IPE_name = header['Tagname'][0]




        for fn in fact_num:

            formule = str(dico_coef[fn]) + '*(' + dico_map_fact_mgl[fn] + '-' + str(serie_mean_fact.loc[fn]) + ')'        
            formula = { 'ScopeMangling': ParentMglInterp,
                        'Description': 'Contrib '+fn,
                        'Identifiant': 'Contrib_'+fn,
                        'Frequency':IPE_freq,
                        'Role':'Data',
                        'DataType':'Numeric',
                        'Unit':IPE_unit,
                        'Formule': formule
                }
            
            tag_mgl_contrib = '['+'.'.join([formula['ScopeMangling'],formula['Identifiant'],formula['Frequency']])+']'

            full_mgl    = ".".join([formula['ScopeMangling'],formula['Identifiant'],formula['Frequency']])
            self.dico_contrib_formula[full_mgl] = formula
            list_tag_mgl_contrib.append(tag_mgl_contrib)
         
        

        dico_formula_cat = self.Formula_Var_Cat()
        if dico_formula_cat:   
            for formula in dico_formula_cat.values():
                tag_mgl_contrib = '['+'.'.join([formula['ScopeMangling'],formula['Identifiant'],formula['Frequency']])+']'
                nom_contrib = formula['Identifiant']
                full_mgl    = ".".join([formula['ScopeMangling'],formula['Identifiant'],formula['Frequency']])
                self.dico_contrib_formula[full_mgl] = formula
                list_tag_mgl_contrib.append(tag_mgl_contrib)
            

        ipe_tag_mgl_moyenne = '['+'.'.join([ParentMglInterp,IPE_name + "_VAL_MOY",IPE_freq])+']' 



        formula_autre = self.FormuleAutresContrib(ipe_tag_mgl_moyenne,model_options['ipe_config']['tag_mgl'],list_tag_mgl_contrib,IPE_freq,IPE_unit)
        full_mgl    = ".".join([formula_autre['ScopeMangling'],formula_autre['Identifiant'],formula_autre['Frequency']])
        self.dico_contrib_formula[full_mgl] = formula_autre
      
        dico_formula = self.dico_contrib_formula
        ScopeMangling = [d['ScopeMangling'] for d in dico_formula.values()]
        Description = [d['Description'] for d in dico_formula.values()]
        Identifiant = [d['Identifiant'] for d in dico_formula.values()]
        Frequency = [d['Frequency'] for d in dico_formula.values()]
        Role = [d['Role'] for d in dico_formula.values()]
        DataType = [d['DataType'] for d in dico_formula.values()]
        Unit = [d['Unit'] for d in dico_formula.values()]
        Formule = [d['Formule'] for d in dico_formula.values()]

        df_formula = pd.DataFrame(

        data = {'ScopeMangling':ScopeMangling,
                'Description':Description,
                'Identifiant':Identifiant,
                'Fréquence':Frequency,
                'Rôles':Role,
                'Type de données':DataType,
                'Unités':Unit,
                'Formule':Formule
                }

        )
        

            #loc_formula = os.path.join(self.formula_rep ,'Formules.formula')

            #df_formula.to_csv("contributions.formula",sep=';',index=False, quoting=csv.QUOTE_NONE)



            #formula_aggregats = self.FormuleAggregats(IPE_freq, formula_ref, self.dico_contrib_formula, formula_autre)
            #for f in formula_aggregats:
                #id_formule+=1
                #dico_formula[id_formule] = f         


            #self.DicoInterp['Formules'] = dico_formula

            #loc_contribution_file = os.path.join(self.formula_rep ,'contribution.json')


            #with open(loc_contribution_file, 'w', encoding='Latin1') as f:
            #    json.dump(self.DicoInterp, f,indent=4,ensure_ascii=False)

            
            #self.ConvertToFormulaFile(IPE_freq,IPE_unit)


    def Formula_Var_Cat(self):

        model_options = self.data_obj.GetModelOptions()
        header = self.data_obj.GetHeader()
        list_feat_cat = [tagname  for tagname,mdtype in zip(header['Tagname'],header['MeasureDataType']) if mdtype in ['Discrete','Blob'] ]
        dico_formula_cat = dict()

        if len(list_feat_cat)>0:

            model = self.model_obj.GetModel()
            regressor = model.steps[1][1]

            dico_tag_mgl_cat = dict()
            for fc in list_feat_cat:
                for d in model_options['factor_configs']:
                    if fc in d['tag_mgl']:
                        dico_tag_mgl_cat[fc] = d['tag_mgl']

            dico_coef_cat = dict()

            for fc in list_feat_cat:
                dico_coef_cat[fc] = dict()
                for coef, col in zip(regressor.coef_,self.df_x.columns):
                    if fc in col:
                        dico_coef_cat[fc][col] = coef

            list_formula_modalite = list()

            dico_formula_cat = dict()

            for fc in list_feat_cat:
                mgl_cat = "["+dico_tag_mgl_cat[fc]+"]"
                for fc_modalite, coef in dico_coef_cat[fc].items():
                    valeur_moyenne = self.df_x[fc_modalite].mean()
                    modalite = fc_modalite.split("_")[-1]
                    formula_modalite = mgl_cat + '.Condition(Equal("' + modalite + '"),1.0,0.0)'
                    formula_modalite = '(' + str(coef) + "*(" + formula_modalite + '-(' + str(valeur_moyenne) + ')))'
                    list_formula_modalite.append(formula_modalite)

                formula_cat = '+'.join(list_formula_modalite)

                formula = { 'ScopeMangling': self.config['ParentMglInterp'],
                            'Description': 'Contrib_'+fc,
                            'Identifiant': 'Contrib_'+fc,
                            'Frequency':header['TagInfoFrequency'][0],
                            'Role':'Data',
                            'DataType':'Numeric',
                            'Unit':header['Unit'][0],
                            'Formule':formula_cat
                    }
                


                dico_formula_cat[fc] = formula

        

        return dico_formula_cat

    def CreateDicoInterpretation(self):
        import json
        
        # Formule du modèle onnx (qui est la même pour les contributions)
        uv_formula = self.rapport_obj.GetJsonReportAsDict()['uv_formula'] 

        # dictionnaire des modèles onnx 
        dico_model_shape_onnx_formula = dict()
        for tagname, nom_model_onnx in self.dico_model_shape_onnx_name.items():
            dico_model = {"uv_formula":uv_formula,"onnx_model_name":nom_model_onnx}
            dico_model_shape_onnx_formula["Contrib_"+tagname] = dico_model

        self.DicoInterp = dict()
        self.DicoInterp['ParentMangling'] = self.PARENT_MGL_INTERPRETATION
        self.DicoInterp['onnx_models']    = dico_model_shape_onnx_formula
        
        # dictionnaire des formules
        id_formule = 1
        dico_formula = dict()

        model_options = self.data_obj.GetModelOptions()
        header = self.data_obj.GetHeader()
        
        IPE_freq = header['TagInfoFrequency'][0]
        IPE_unit = header['Unit'][0]
        IPE_name = header['Tagname'][0]



        ipe_tag_mgl_model = '['+'.'.join([model_options['ipe_config']['tag_mgl'],'NormalizationRecalc'])+']'

        ipe_tag_mgl_moyenne = '['+'.'.join([self.PARENT_MGL_INTERPRETATION,IPE_name + "_MoyRef",IPE_freq])+']'

    
        dico_contrib_tag_mgl = dict()

        for tagname_contrib in dico_model_shape_onnx_formula.keys():
            mgl = '['+'.'.join([self.PARENT_MGL_INTERPRETATION,tagname_contrib,IPE_freq,'NormalizationRecalc'])+']'
            dico_contrib_tag_mgl[tagname_contrib] = mgl


        # Formule valeur moyenne de l'IPE sur référence

        formula_ref = self.FormuleIPERef()
        dico_formula[id_formule] = formula_ref


        # Formule de l'erreur d'ajustement

        formula_erreur_ajust = self.FormuleErreurAjustement(ipe_tag_mgl_model,
                                                            ipe_tag_mgl_moyenne,
                                                            dico_contrib_tag_mgl,
                                                            IPE_freq,
                                                            IPE_unit)
        id_formule+=1
        dico_formula[id_formule] = formula_erreur_ajust

        # Formule somme des valeurs absolues des contributions

        formula_somme_contrib_abs = self.FormuleSommeContribAbs(dico_contrib_tag_mgl,IPE_freq,IPE_unit)
        id_formule+=1
        dico_formula[id_formule] = formula_somme_contrib_abs

        # Formule des contributions ajustées

        tag_mgl_erreur_ajust      = '['+'.'.join([formula_erreur_ajust['ScopeMangling'],formula_erreur_ajust['Identifiant'],formula_erreur_ajust['Frequency']])+']'
        tag_mgl_somme_contrib_abs = '['+'.'.join([formula_somme_contrib_abs['ScopeMangling'],formula_somme_contrib_abs['Identifiant'],formula_somme_contrib_abs['Frequency']])+']'

        list_tag_mgl_contrib_ajustees = list()
        dico_contrib_ajustees_formula = dict()

        for contrib_name, mgl_contrib in dico_contrib_tag_mgl.items():
            formula_contrib_ajuste = self.FormuleContribAjustee(tag_mgl_erreur_ajust,tag_mgl_somme_contrib_abs,contrib_name,mgl_contrib,IPE_freq,IPE_unit)
            id_formule+=1
            dico_formula[id_formule] = formula_contrib_ajuste

            list_tag_mgl_contrib_ajustee = '['+'.'.join([formula_contrib_ajuste['ScopeMangling'],formula_contrib_ajuste['Identifiant'],formula_contrib_ajuste['Frequency'],formula_contrib_ajuste['Role']])+']'
            list_tag_mgl_contrib_ajustees.append(list_tag_mgl_contrib_ajustee)
            dico_contrib_ajustees_formula[contrib_name] = formula_contrib_ajuste




        formula_autre = self.FormuleAutresContrib(ipe_tag_mgl_moyenne,model_options['ipe_config']['tag_mgl'],list_tag_mgl_contrib_ajustees,IPE_freq,IPE_unit)
        id_formule+=1
        dico_formula[id_formule] = formula_autre
        #tag_mgl_autres = '['+'.'.join([formula_autre['ScopeMangling'],formula_autre['Identifiant'],formula_autre['Frequency']])+']'


        formula_aggregats = self.FormuleAggregats(IPE_freq, formula_ref, dico_contrib_ajustees_formula, formula_autre)
        for f in formula_aggregats:
            id_formule+=1
            dico_formula[id_formule] = f         


        self.DicoInterp['Formules'] = dico_formula


        loc_contribution_file = os.path.join(self.formula_rep ,'contribution.json')


        with open(loc_contribution_file, 'w', encoding='Latin1') as f:
            json.dump(self.DicoInterp, f,indent=4,ensure_ascii=False)

        self.ConvertToFormulaFile(IPE_freq,IPE_unit)


    def ConvertToFormulaFile(self,IPE_freq,IPE_unit):

        import pandas as pd
        import csv

        dico_formula = self.DicoInterp['Formules']
        ScopeMangling = [d['ScopeMangling'] for d in dico_formula.values()]
        Description = [d['Description'] for d in dico_formula.values()]
        Identifiant = [d['Identifiant'] for d in dico_formula.values()]
        Frequency = [d['Frequency'] for d in dico_formula.values()]
        Role = [d['Role'] for d in dico_formula.values()]
        DataType = [d['DataType'] for d in dico_formula.values()]
        Unit = [d['Unit'] for d in dico_formula.values()]
        Formule = [d['Formule'] for d in dico_formula.values()]

        df_formula = pd.DataFrame(

        data = {'ScopeMangling':ScopeMangling,
                'Description':Description,
                'Identifiant':Identifiant,
                'Fréquence':Frequency,
                'Rôles':Role,
                'Type de données':DataType,
                'Unités':Unit,
                'Formule':Formule
                }

        )

        loc_formula = os.path.join(self.formula_rep ,'Formules.formula')
        df_formula.to_csv(loc_formula,sep=';',index=False, quoting=csv.QUOTE_NONE)

        if 'onnx_models' in self.DicoInterp.keys():

            dico_onnx = self.DicoInterp['onnx_models']
            nbcontrib = len(dico_onnx)

                
            df_formula_onnx = pd.DataFrame(

            data = {'ScopeMangling':[self.PARENT_MGL_INTERPRETATION]*nbcontrib,
                    'Description':list(dico_onnx.keys()),
                    'Identifiant':list(dico_onnx.keys()),
                    'Fréquence':[IPE_freq]*nbcontrib,
                    'Rôles':['NormalizationRecalc']*nbcontrib,
                    'Type de données':['Numeric']*nbcontrib,
                    'Unités':[IPE_unit]*nbcontrib,
                    'Formule':[0.0]*nbcontrib
                    }

            )

            loc_formula_onnx = os.path.join(self.formula_rep ,'Formules_onnx.formula')
            df_formula_onnx.to_csv(loc_formula_onnx,sep=';',index=False, quoting=csv.QUOTE_NONE)


    def FormuleAggregats(self,IPE_freq,formula_ref, dico_contrib_ajustees_formula, formula_autre):

        SECOND      = 0
        PRODUCTUNIT = 1
        MINUTE      = 2
        HEURE       = 3
        SHIFTWORK   = 4
        DAY         = 5
        WEEK        = 6
        MONTH       = 7

        dico_map_freq = { 'Second': SECOND,
                          'ProductUnit':PRODUCTUNIT,
                          'Minute':MINUTE,
                          'Hour':HEURE,
                          'ShiftWork':SHIFTWORK,
                          'Day':DAY,
                          'Week':WEEK,
                          'Month':MONTH}

        if dico_map_freq[IPE_freq] < SHIFTWORK: # On calcul les aggrégats à la fréquence SHIFTWORK, DAY, MONTH

            list_formula = list()
            current_freq = 'ShiftWork'
            list_freq_to_aggregate = ['ShiftWork','Day','Month']
            list_freq_value        = ['8h','1d','1M']

            for current_freq, freq in zip(list_freq_to_aggregate, list_freq_value):
                # Moyenne IPE
                tag_mgl_srce =  '['+'.'.join([formula_ref['ScopeMangling'],formula_ref['Identifiant'],formula_ref['Frequency']])+']'                
                formule      = tag_mgl_srce + '.Average("' + freq + '","' + self.OFFSET +'")'

                formula = { 'ScopeMangling': formula_ref['ScopeMangling'],
                            'Description': formula_ref['Description'],
                            'Identifiant': formula_ref['Identifiant'],
                            'Frequency':current_freq,
                            'Role':'Data',
                            'DataType':'Numeric',
                            'Unit':formula_ref['Unit'],
                            'Formule': formule
                    }

                list_formula.append(formula)

                # Contributions

                for f in dico_contrib_ajustees_formula.values():

                    if f['Role'] == 'Data':
                        tag_mgl_srce =  '['+'.'.join([f['ScopeMangling'],f['Identifiant'],f['Frequency']])+']'       
                    else:
                        tag_mgl_srce =  '['+'.'.join([f['ScopeMangling'],f['Identifiant'],f['Frequency'],f['Role']])+']'       



                    formule = tag_mgl_srce + '.Average("' + freq + '","' + self.OFFSET +'")'
                    formula = { 'ScopeMangling': f['ScopeMangling'],
                                'Description': f['Description'],
                                'Identifiant': f['Identifiant'],
                                'Frequency':current_freq,
                                'Role':f['Role'],
                                'DataType':'Numeric',
                                'Unit':f['Unit'],
                                'Formule': formule
                        }

                    list_formula.append(formula)

                # Autres contributions
                tag_mgl_srce =  '['+'.'.join([formula_autre['ScopeMangling'],formula_autre['Identifiant'],formula_autre['Frequency']])+']'                
                formule      = tag_mgl_srce + '.Average("' + freq + '","' + self.OFFSET +'")'

                formula = { 'ScopeMangling': formula_autre['ScopeMangling'],
                            'Description': formula_autre['Description'],
                            'Identifiant': formula_autre['Identifiant'],
                            'Frequency':current_freq,
                            'Role':'Data',
                            'DataType':'Numeric',
                            'Unit':formula_autre['Unit'],
                            'Formule': formule
                    }

                list_formula.append(formula)







        return list_formula


    def FormuleAutresContrib(self,ipe_tag_mgl_moyenne,ipe_tag_mgl,list_tag_mgl_contrib_ajustees,IPE_freq,IPE_unit):

        # Formule pour "autre"

        formule = '['+ ipe_tag_mgl + ']' + '-(' + ipe_tag_mgl_moyenne + '+'
        formule += '+'.join(list_tag_mgl_contrib_ajustees) + ')'
        ParentMglInterp = self.config['ParentMglInterp']
        
        formula = {'ScopeMangling': ParentMglInterp,
                'Description': 'Contrib Autre',
                'Identifiant': 'Contrib_Autre',
                'Frequency':IPE_freq,
                'Role':'Data',
                'DataType':'Numeric',
                'Unit':IPE_unit,
                'Formule': formule
                }


        return formula




    def FormuleContribAjustee(self,tag_mgl_erreur_ajust,tag_mgl_somme_contrib_abs,contrib_name,mgl_contrib,IPE_freq,IPE_unit):

        # Formule pour les contributions ajustées


        formule = mgl_contrib + '+' + mgl_contrib + '.Abs()/' + tag_mgl_somme_contrib_abs + '*' + tag_mgl_erreur_ajust
        
        formula = {'ScopeMangling': self.PARENT_MGL_INTERPRETATION,
                'Description': contrib_name,
                'Identifiant': contrib_name,
                'Frequency':IPE_freq,
                'Role':'Cleaned',
                'DataType':'Numeric',
                'Unit':IPE_unit,
                'Formule': formule
                }


        return formula





    def FormuleSommeContribAbs(self,dico_contrib_tag_mgl,IPE_freq,IPE_unit):
        # Formule pour l'erreur d'ajustement

        formule = '' 
        list_contrib_tag_mgl = list(dico_contrib_tag_mgl.values())
        for mgl in list_contrib_tag_mgl:
            formule += mgl + '.Abs()+'
        formule = formule[:-1]

        
        formula = {'ScopeMangling': self.PARENT_MGL_INTERPRETATION,
                'Description': "Somme Contribution Absolue",
                'Identifiant': "Somme_Contrib_Abs",
                'Frequency':IPE_freq,
                'Role':'Data',
                'DataType':'Numeric',
                'Unit':IPE_unit,
                'Formule': formule
                }


        return formula

    def FormuleErreurAjustement(self,ipe_tag_mgl_model,ipe_tag_mgl_moyenne,dico_contrib_tag_mgl,IPE_freq,IPE_unit):
        # Formule pour l'erreur d'ajustement

        formule = ipe_tag_mgl_model+'-('+ ipe_tag_mgl_moyenne 
        list_contrib_tag_mgl = list(dico_contrib_tag_mgl.values())
        formule += '+'+'+'.join(list_contrib_tag_mgl) + ')'

        
        formula = {'ScopeMangling': self.PARENT_MGL_INTERPRETATION,
                'Description': "Erreur Ajustement",
                'Identifiant': "Erreur_Ajust",
                'Frequency':IPE_freq,
                'Role':'Data',
                'DataType':'Numeric',
                'Unit':IPE_unit,
                'Formule': formule
                }


        return formula



    def FormuleIPERef(self):
        # Calcul de la valeur moyenne de l’IPE sur la période de référence

        df_data = self.data_obj.GetData()
        model_options = self.data_obj.GetModelOptions()
        header = self.data_obj.GetHeader()
        IPE_name = df_data.columns[0]
        IPE_moyen_ref = df_data[IPE_name].mean()

        ipe_tag_mgl = model_options['ipe_config']['tag_mgl']
        IPE_desc = header['Description'][0]
        IPE_freq = header['TagInfoFrequency'][0]
        IPE_unit = header['Unit'][0]
        ParentMglInterp = self.config['ParentMglInterp']


        formula = {'ScopeMangling': ParentMglInterp,
                'Description': IPE_desc + " Valeur Moyenne",
                'Identifiant': IPE_name + "_VAL_MOY",
                'Frequency':IPE_freq,
                'Role':'Data',
                'DataType':'Numeric',
                'Unit':IPE_unit,
                'Formule': str(IPE_moyen_ref) + '+0.0*[' + ipe_tag_mgl + ']'
                }
        return formula

    def CreateGraphes(self):

        import matplotlib.pyplot as plt
        import shap
        dico_graphes = dict()

        if self.svals is not None:
            fig_importance_fact = plt.figure(facecolor="#c6dee8")
            ax_fi = fig_importance_fact.axes
            shap.summary_plot(self.svals, self.df_x, plot_type="bar",show=False)
            plt.gca().tick_params(labelsize=10)
            plt.gca().set_xlabel("Impact moyen sur la variable modélisée", fontsize=14)


            fig_shape_fact = plt.figure(facecolor="#c6dee8")
            ax_shape = fig_shape_fact.axes
            shap.summary_plot(self.svals, self.df_x,show=False)
            plt.gca().tick_params(labelsize=10)
            plt.gca().set_xlabel("Impact sur la variable modélisée", fontsize=14)


            dico_graphes["fig_importance_fact"] = fig_importance_fact
            dico_graphes["fig_shape_fact"]      = fig_shape_fact


        return dico_graphes
    