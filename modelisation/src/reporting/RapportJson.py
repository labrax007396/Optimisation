
import os,sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)


from reporting.RapportJsonClasses import CleaningInfo,ContinousSerieInfo,CategoricalVariable,DiscreteSerieInfo,WeightVariable
from reporting.RapportJsonClasses import WeightVariableSerieInfo, Target, DataframeInfo, ModelInfo, UVFormula, ReportModel


from data.data import Data
import pandas as pd
import os

class RapportJson:


    def __init__(self, model_obj, data_obj:Data) -> None:
        self.model_obj = model_obj
        self.data_obj  = data_obj


    def GetJsonReportAsString(self):
        return self.modelreport_obj    

    def GetJsonReportAsDict(self):
        import json
        return json.loads(self.modelreport_obj)
    
    def SaveJsonReport(self):
        import json

        report_dict = json.loads(self.modelreport_obj)
        with open('raport_modelisation.json.json', 'w', encoding='utf-8') as f:
            json.dump(report_dict, f,indent=4)

    def CreateReport(self):
        from datetime import datetime
    
        resu_model    = self.model_obj.GetResuModel()
        data  = self.data_obj.GetData()
        entete = self.data_obj.GetHeader()

        dico_model   = self.CreateDicoModel(entete=entete, data = data)
        #clean_report = self.CleanReport(data = data, dico_model = dico_model)
        clean_report = self.data_obj.GetCleanReport()
        
        df_num_corr     = self.CorrCoef(data = data, dico_model = dico_model)
        df_matrix_corr  = self.CorrMatrix()



        model_type  =  resu_model['Type de modèle']
        formula     =  resu_model['formula_string']
        description =  dico_model['description']
        
        ref_periode_debut = dico_model['ref_periode_debut']
        ref_periode_fin = dico_model['ref_periode_fin']

        dico_fact_sens,r2_test,r2_train,mape_test,mape_train,mean_deviation_train,mean_deviation_test,standard_deviation_train,standard_deviation_test = self.MetricsAndFeaturesSensibility(dico_model=dico_model)


    

        creation_date = datetime.now().isoformat()


        if 'description' in dico_model.keys():
            description = dico_model['description']

        else:
            description = 'non définie'


        model_info = ModelInfo(model_type    = model_type,
                            description   = description, 
                            creation_date = creation_date,
                            formula       = formula,
                            r2_test       = r2_test,
                            r2_train      = r2_train,
                            mape_test          = mape_test,
                            mape_train          = mape_train,
                            mean_deviation_train = mean_deviation_train,
                            mean_deviation_test = mean_deviation_test,
                            standard_deviation_train = standard_deviation_train,
                            standard_deviation_test = standard_deviation_test)

        cleaninginfo  = CleaningInfo(line_count_before = clean_report['ndata_before_clean'],
                                    line_count_after  = clean_report['ndata_after_clean'])
    

        target_continousinfo = ContinousSerieInfo(vmin = data[dico_model['tag_name']].min(),
                                                vmax = data[dico_model['tag_name']].max(),
                                                mean = data[dico_model['tag_name']].mean(),
                                                standard_deviation = data[dico_model['tag_name']].std(),
                                                influence_weight = 0.0,
                                                missingdata = clean_report['detail']['target']['missingdata'],
                                                outliers    = clean_report['detail']['target']['outliers'])

        #print("Peuplement des features")



        listfeatures = []
        for tag in dico_model['facteurs']:
            if dico_model['facteurs'][tag]['used']:
                

                

                nom_feature = dico_model['facteurs'][tag]['nom']

                if dico_model['facteurs'][tag]['type'] == 'num':   
                    influence_weight = dico_fact_sens[nom_feature]
                    corr_coef        = df_num_corr[nom_feature]


                    feature_continousinfo = ContinousSerieInfo(vmin = data[nom_feature].min(),
                                                            vmax = data[nom_feature].max(),
                                                            mean = data[nom_feature].mean(),
                                                            standard_deviation = data[nom_feature].std(),
                                                            influence_weight = influence_weight,
                                                            missingdata = clean_report['detail']['features'][tag]['missingdata'],
                                                            outliers = clean_report['detail']['features'][tag]['outliers'])
                    #print(clean_report['detail']['features'][tag]['missingdata'])

                    
                    target = Target(tag_mgl = tag,
                                    name   = nom_feature,
                                    description = nom_feature,
                                    corr_coef = corr_coef,
                                    used = True,
                                    discrete_serie_info = None,
                                    continous_serie_info = feature_continousinfo)
                    


                    listfeatures.append(target)



                if dico_model['facteurs'][tag]['type'] == 'cat':

                    #features_weight['Facteur'] = features_weight['Facteur'].apply(lambda s:s.replace(nom_feature+'_',''))
                    #features_weight.set_index('Facteur',inplace=True)

                    cat_var_list = []
                
                    cat_count = data[nom_feature].value_counts()

                    for code, nbre in cat_count.items():

                        code = str(code)

                        #importance_percent = features_weight.loc[code,'Poids Facteur %']
                        influence_weight = dico_fact_sens[code]

                        cat_var_obj = CategoricalVariable(name=code,
                                                        occurrences=nbre,
                                                        influence_weight = influence_weight)
                        cat_var_list.append(cat_var_obj)

                    corr_coef        = df_num_corr[nom_feature]
                    disret_info_serie_objet = DiscreteSerieInfo(categorical_variables = cat_var_list)



                    target = Target(tag_mgl = tag,
                                    name   = nom_feature,
                                    description = nom_feature,
                                    corr_coef = corr_coef,
                                    used = True,
                                    discrete_serie_info = disret_info_serie_objet,
                                    continous_serie_info = None)
                    

                    listfeatures.append(target)

            else:  ## La feature n'est pas utilisée

                nom_feature = dico_model['facteurs'][tag]['nom']

                if dico_model['facteurs'][tag]['type'] == 'num':   

                    influence_weight = 0.0
                    corr_coef        = df_num_corr[nom_feature]

                    feature_continousinfo = ContinousSerieInfo(vmin = 0.0,
                                                            vmax = 0.0,
                                                            mean = 0.0,
                                                            standard_deviation = 0.0,
                                                            influence_weight = 0.0,
                                                            missingdata = 0,
                                                            outliers = 0)

                    target = Target(tag_mgl = tag,
                                    name   = nom_feature,
                                    description = nom_feature,
                                    corr_coef = corr_coef,
                                    used = False,
                                    discrete_serie_info = None,
                                    continous_serie_info = None)


                    listfeatures.append(target)

                if dico_model['facteurs'][tag]['type'] == 'cat':

                    influence_weight = 0.0
                    corr_coef        = df_num_corr[nom_feature]


                    target = Target(tag_mgl = int(tag[4:]),
                                    name   = nom_feature,
                                    description = nom_feature,
                                    corr_coef = corr_coef,
                                    used = False,
                                    discrete_serie_info = None,
                                    continous_serie_info = None)

                    listfeatures.append(target)

      
        target_modelise = Target(tag_mgl = dico_model['tag_modelise'],
                    name   = dico_model['tag_name'],
                    description = dico_model['tag_name'],
                    corr_coef = 1.0,
                    used = True,
                    discrete_serie_info = None,
                    continous_serie_info = target_continousinfo)
        

        dataframeinfo_obj = DataframeInfo(start_date = ref_periode_debut,
                                        end_date   = ref_periode_fin,
                                        cleaning_info = cleaninginfo,
                                        target = target_modelise,
                                        features = listfeatures,
                                        corr_matrix = df_matrix_corr)


        uv_formula_obj =  UVFormula(dico_model = dico_model, data = data)

        if model_type == "RegressionLineaire":
            if len(self.model_obj.formula_uv) >0 :
                uv_formula_obj.formula = self.model_obj.formula_uv

        if model_type == "RegressionExp":
            uv_formula_obj.formula = self.model_obj.formula_uv

        if model_type == "RegressionPuissance":
            uv_formula_obj.formula = self.model_obj.formula_uv

        modelreport_obj = ReportModel(site = dico_model['site'],
                                    dataframe_info = dataframeinfo_obj,
                                    model_info = model_info,
                                    uv_formula = uv_formula_obj.formula)

        self.modelreport_obj = modelreport_obj.toJson()

    def compute_mape_train(self,resu):

        import warnings
        import numpy as np
        warnings.filterwarnings("ignore", category=RuntimeWarning) 

        resu['erreur_rel'] = resu.apply(lambda row: 100*abs((row['y_pred_train']-row['y_train'])/row['y_train']),axis=1)
        resu.replace([np.inf, -np.inf], np.nan, inplace=True)
        resu['erreur_rel'].dropna(inplace=True)
        if resu.empty:
            mape = 0.0
        else:
            resu_kept = resu[resu['erreur_rel']<=100.0]
            if resu_kept.empty:
                mape = 0.0
            else:
                mape = resu_kept['erreur_rel'].mean()
        
        return mape

    def compute_mape_test(self,resu):

        import warnings
        import numpy as np
        warnings.filterwarnings("ignore", category=RuntimeWarning) 

        resu['erreur_rel'] = resu.apply(lambda row: 100*abs((row['y_pred_test']-row['y_test'])/row['y_test']),axis=1)
        resu.replace([np.inf, -np.inf], np.nan, inplace=True)
        resu['erreur_rel'].dropna(inplace=True)
        if resu.empty:
            mape = 0.0
        else:
            resu_kept = resu[resu['erreur_rel']<=100.0]
            if resu_kept.empty:
                mape = 0.0
            else:
                mape = resu_kept['erreur_rel'].mean()
        
        return mape

    def MetricsAndFeaturesSensibility(self,dico_model:dict):

        import numpy as np
        from sklearn.metrics import r2_score

        test_data_set = self.model_obj.GetTestData()
        train_data_set = self.model_obj.GetTrainData()
        fitted_model = self.model_obj.GetModel()
        data  = self.data_obj.GetData()
        resu_model    = self.model_obj.GetResuModel()
        model_type  =  resu_model['Type de modèle']


    
        #print("Calcul des sensibilités des facteurs")

        dico_fact_sens = dict()
        X = data.drop(columns=dico_model['tag_name'])

        for fact in X.columns:
            if X[fact].dtypes != 'object':   

                df_25 = pd.DataFrame(columns=X.columns.to_list())
                list_25 = list()
                df_75 = pd.DataFrame(columns=X.columns.to_list())
                list_75 = list()
                for fact1 in X.columns:
                    if X[fact1].dtypes == 'float64':
                        list_25.append(X[fact1].mean())
                        list_75.append(X[fact1].mean())
                    else:
                        list_25.append('other')
                        list_75.append('other')
                df_25.loc[0] = list_25 
                df_25[fact]  = X[fact].min() + 0.25*(X[fact].max() - X[fact].min())
                df_75.loc[0] = list_75 
                df_75[fact]  = X[fact].min() + 0.75*(X[fact].max() - X[fact].min())    

                if model_type=="rnn":

                    features_dict_25 = {name: np.array(value) 
                            for name, value in df_25.items()}

                    features_dict_75 = {name: np.array(value) 
                            for name, value in df_75.items()}

                    pred_25 = fitted_model.predict(features_dict_25)[0]
                    pred_75 = fitted_model.predict(features_dict_75)[0]
                    

                else:    

                    pred_25 = fitted_model.predict(df_25)
                    pred_75 = fitted_model.predict(df_75)

                dico_fact_sens[fact] = pred_75[0] - pred_25[0]

            else:

                list_mod_unique = list(X[fact].unique())
                list_mod_unique = [str(l) for l in list_mod_unique]



                df_sans_mod = pd.DataFrame(columns=X.columns.to_list())
                list_sans_mod = list()
                df_avec_mod  = pd.DataFrame(columns=X.columns.to_list())

                list_avec_mod = list()

    ###############################################################################

                list_fact_num = [f for f in X.columns if X[f].dtypes == 'float64']

                dico_sans_mod = dict()

                for fact1 in X.columns:
                    if X[fact1].dtypes != 'object':
                        dico_sans_mod[fact1] = X[fact1].mean()
                    else:
                        dico_sans_mod[fact1] = 'other'

                dico_avec_mod = dico_sans_mod.copy()
                df_sans_mod = pd.DataFrame(data = dico_sans_mod,index = [0])

                if model_type=="rnn":

                    features_dict_sans_mod = {name: np.array(value) 
                            for name, value in df_sans_mod.items()}

                    pred_sans_mod = fitted_model.predict(features_dict_sans_mod)[0]



                else:


                    pred_sans_mod = fitted_model.predict(df_sans_mod)

                if model_type=="rnn":

                    for mod in list_mod_unique:

                        dico_avec_mod[fact] = mod
                        df_avec_mod = pd.DataFrame(data = dico_avec_mod,index = [0])

                        features_dict_avec_mod = {name: np.array(value) 
                                for name, value in df_avec_mod.items()}

                        pred_avec_mod = fitted_model.predict(features_dict_avec_mod)[0]
                        dico_fact_sens[mod] = pred_avec_mod[0] - pred_sans_mod[0]                


                
                else:

                    for mod in list_mod_unique:

                        dico_avec_mod[fact] = mod
                        df_avec_mod = pd.DataFrame(data = dico_avec_mod,index = [0])

                        #df_avec_mod.to_pickle("df_avec_mod.pkl")
                        #pickle.dump(fitted_model, open("fitted_model.pkl", 'wb'))

                        pred_avec_mod = fitted_model.predict(df_avec_mod)
                        dico_fact_sens[mod] = pred_avec_mod[0] - pred_sans_mod[0]                
                        



        ######### Calcul des scores du modèle  #########

        #print("Calcul des scores du modèle")

        if test_data_set is not None:
        
            if isinstance(train_data_set, pd.DataFrame):
                test_data = test_data_set.drop(columns=dico_model["tag_name"])
                y_test = test_data_set[dico_model["tag_name"]].to_frame()
                train_data = train_data_set.drop(columns=dico_model["tag_name"])
                y_train = train_data_set[dico_model["tag_name"]].to_frame()
            else:
                y_test    = test_data_set.keep_columns(dico_model['tag_name']).to_pandas_dataframe()
                test_data = test_data_set.drop_columns(dico_model['tag_name']).to_pandas_dataframe()
                y_train    = train_data_set.keep_columns(dico_model['tag_name']).to_pandas_dataframe()
                train_data = train_data_set.drop_columns(dico_model['tag_name']).to_pandas_dataframe()


            if model_type=="rnn":

                #print("Calcul predictions train")
                features_dict_train = {name: np.array(value) for name, value in train_data.items()}
                y_pred_train = fitted_model.predict(features_dict_train).flatten()
                #print("Calcul predictions test")
                features_dict_test = {name: np.array(value) for name, value in test_data.items()}
                y_pred_test  = fitted_model.predict(features_dict_test).flatten()

            else:


                #print("Calcul predictions train")
                y_pred_train = fitted_model.predict(train_data)
                #print("Calcul predictions test")
                y_pred_test  = fitted_model.predict(test_data)

            #resu = pd.DataFrame({'y_model':y_pred, 'y_mesure':y.values})

            resu_train = pd.DataFrame({'y_pred_train':y_pred_train, 'y_train':y_train[dico_model['tag_name']]})
            resu_test = pd.DataFrame({'y_pred_test':y_pred_test, 'y_test':y_test[dico_model['tag_name']]})

            # calcul des métriques sur la période d'apprentissage

            ndata_train = len(y_pred_train)
            r2_train = r2_score(y_train,y_pred_train)
            
            resu_train['erreur']     = resu_train.apply(lambda row: abs(row['y_pred_train']-row['y_train']),axis=1)

            mean_deviation_train = resu_train['erreur'].mean()
            standard_deviation_train = resu_train['erreur'].std()
            mape_train = self.compute_mape_train(resu_train)


            # calcul des métriques sur la période de test

            ndata_test  = len(y_pred_test)
            r2_test  = r2_score(y_test,y_pred_test)
            
            resu_test['erreur']     = resu_test.apply(lambda row: abs(row['y_pred_test']-row['y_test']),axis=1)
            mean_deviation_test = resu_test['erreur'].mean()
            standard_deviation_test = resu_test['erreur'].std()
            mape_test = self.compute_mape_test(resu_test)




        else: # Pas de données de validation

            if isinstance(train_data_set, pd.DataFrame):
                train_data = train_data_set.drop(columns=dico_model["tag_name"])
                y_train = train_data_set[dico_model["tag_name"]]
            else:
                y_train    = train_data_set.keep_columns(dico_model['tag_name']).to_pandas_dataframe()
                train_data = train_data_set.drop_columns(dico_model['tag_name']).to_pandas_dataframe()



            if model_type=="rnn":

                #print("Calcul predictions train")
                features_dict_train = {name: np.array(value) for name, value in train_data.items()}
                y_pred_train = fitted_model.predict(features_dict_train).flatten()

            else:

                #print("Calcul predictions train")
                y_pred_train = fitted_model.predict(train_data)
            
            resu_train = pd.DataFrame(data = {'y_pred_train':y_pred_train, 'y_train':y_train})


            r2_train = r2_score(y_train,y_pred_train)
            r2_test  = None


            # calcul des métriques sur la période d'apprentissage

            ndata_train = len(y_pred_train)
            r2_train = r2_score(y_train,y_pred_train)
            
            resu_train['erreur']     = resu_train.apply(lambda row: abs(row['y_pred_train']-row['y_train']),axis=1)

            mean_deviation_train = resu_train['erreur'].mean()
            standard_deviation_train = resu_train['erreur'].std()
            mape_train = self.compute_mape_train(resu_train)

            ndata_test  = 0
            r2_test  = None

            mean_deviation_test = None
            standard_deviation_test = None
            mape_test = None

        return dico_fact_sens,r2_test,r2_train,mape_test,mape_train,mean_deviation_train,mean_deviation_test,standard_deviation_train,standard_deviation_test
    

    def CorrMatrix(self):


        dataframe = self.data_obj.GetData()
        header    = self.data_obj.GetHeader()

        dico_map = {t:d for t,d,type in zip(header['Tagname'], header['Description'], header['MeasureDataType']) if type=='Numeric'}

        data_num = dataframe[list(dico_map.keys())]

        return data_num.corr().to_dict()



    def CorrCoef(self, data:pd.DataFrame, dico_model:dict):

        import pandas as pd
        var_numerique = [v for v in data.columns if data[v].dtypes != 'object']
        data_num = data[var_numerique]
        df_num_corr = data_num.corr().drop(dico_model['tag_name'])[dico_model['tag_name']] 
        df_num_corr = df_num_corr.to_dict()

        

        var_cat  = [v for v in data.columns if data[v].dtypes == 'object']

        for vc in var_cat:

            data_cat = data[[dico_model['tag_name']]+[vc]]
            data_cat = pd.get_dummies(data_cat)
            df_cat_corr = data_cat.corr().drop(dico_model['tag_name'])[dico_model['tag_name']] 
            df_num_corr[vc] = df_cat_corr.max()

        return df_num_corr



    def CleanReport(self, data:pd.DataFrame,dico_model:dict):
            # Valeurs provisoire pour le clean report

        clean_report = dict()
        clean_report['ndata_before_clean'] = len(data)
        clean_report['ndata_after_clean']  = len(data)
        clean_report['detail'] = dict()
        clean_report['detail']['target'] = dict()

        clean_report['detail']['target']['missingdata'] = 0
        clean_report['detail']['target']['outliers'] = 0
        clean_report['detail']['features'] = dict()
        for t in dico_model['facteurs'].keys():
            clean_report['detail']['features'][t] = dict()
            clean_report['detail']['features'][t]['missingdata']= 0
            clean_report['detail']['features'][t]['outliers'] = 0


        return clean_report
        
    def CreateDicoModel(self, entete:dict, data:pd.DataFrame):

        from datetime import datetime


        dico_model = dict()

        role = '' if entete['TagInfoRole'][0] == 'Data' else '.'+entete['TagInfoRole'][0]

        dico_model["tag_modelise"] = entete['ParentScopeMangling'][0] + '.' + entete['Tagname'][0] + '.' + entete['TagInfoFrequency'][0] + role
        dico_model["site"] = entete['ParentScopeMangling'][0].split(".")[0]
        dico_model["mangling"] = entete['ParentScopeMangling'][0]
        dico_model["tag_name"] = entete['Tagname'][0]
        dico_model["ref_periode_debut"] = data.index[0].isoformat()
        dico_model["ref_periode_fin"]   = data.index[-1].isoformat()
        dico_model["tag_unit"]   = entete['Unit'][0]
        dico_model["freq"]   = entete['TagInfoFrequency'][0]
        dico_model["description"]   = entete['Description'][0]

    

        header_df = pd.DataFrame(data=entete)
        header_df.drop(0, inplace=True)

        dico_fact = dict()

        for _,row in header_df.iterrows():
            d_f = dict()
            d_f["used"] = True
            d_f["type"] = "num" if row['MeasureDataType'] == 'Numeric' else "cat"
            d_f["unit"] = row['Unit']
            d_f["nom"]  = row['Tagname']
            d_f["description"] = row['Description']
            frole = '' if row['TagInfoRole']=='Data' else '.'+row['TagInfoRole']
            fact_mangling = row['ParentScopeMangling']+'.'+row['Tagname']+'.'+row['TagInfoFrequency']+frole
            dico_fact[fact_mangling] = d_f

        dico_model['facteurs'] = dico_fact

        return dico_model
