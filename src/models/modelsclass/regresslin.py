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
    
    def CreateJsonResults(self):

        super().CreateJsonResults()

        if not self.resu_model == False:
            
            self.RegressionWithStatsmodels()

            facteurs_name = self.model_smf.pvalues.index.tolist()

            facteurs_name = self.data_obj.set_list_to_init_tagname(facteurs_name)

            self.resu_model['Statistiques']['pvalues'] = self.model_smf.pvalues.tolist()
            self.resu_model['Statistiques']['facteurs'] = facteurs_name
            self.resu_model['Statistiques']['coefs']    = self.model_smf.params.to_list()            
            self.resu_model['Statistiques']['pertinents'] = [True if pv<0.05 else False for pv in self.model_smf.pvalues.tolist() ]
        

    def RegressionWithStatsmodels(self):

        import statsmodels.formula.api as smf

        self.data_obj.set_header_to_new_tagname()

        data = self.data_obj.GetData()

        if data is not None:

            try:
                data.dropna(inplace=True)
                #col_names = data.columns.to_list()
                #new_names = ['_'+c for c in col_names]
                #data.columns = new_names
                formula = data.columns[0] + '~'
                formula = formula + '+'.join(list(data.columns[1:]))
                self.model_smf = smf.ols(formula=formula, data=data).fit()

                self.data_obj.set_header_to_init_tagname()

            except Exception as error:
                self.model_smf = None

    def CreateFormula(self):

        import numpy as np
        import math
        from sklearn.linear_model import LinearRegression

        dico_map_tn_to_tmgl = dict()
        header = self.data_obj.GetHeader()

        for tn, pscope, freq, role in zip(header['Tagname'], header['ParentScopeMangling'], header['TagInfoFrequency'],header['TagInfoRole']):
            t_role = '' if role=='Data' else '.'+role
            tag_mgl = pscope + '.' + tn + '.' + freq + t_role
            dico_map_tn_to_tmgl[tn] = tag_mgl 


        data = self.data_obj.GetData()
        data.dropna(inplace=True)
        target = data.columns[0]

        cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

        train_data_set = data.copy()
        train_x = train_data_set.drop(target,axis=1)
        train_y = train_data_set[target]

        if len(cat_feat)>0:
            X = pd.get_dummies(train_x,columns=cat_feat)
        else:
            X = train_x

        model_sansprepro = LinearRegression()
        model_sansprepro.fit(X ,train_y)
     
        formula = self.format_coef(model_sansprepro.intercept_)   

        self.dico_coef = dict()
        self.dico_coef['intercept'] = model_sansprepro.intercept_

        for f,c in zip(X.columns, model_sansprepro.coef_):
            self.dico_coef[f] = c
            if math.copysign(1,c) > 0.0:
                op = '+'
            else:
                op = ''

            c_str = self.format_coef(c)
            formula = formula + op + c_str + '*' + f

        formula = formula.replace('-',' - ').replace('+',' + ').replace('*',' * ')
        if formula[0] == ' ':
            formula = formula[1:]

        self.formula = formula



        if len(cat_feat)>0:
            self.formula_uv = ''
        else:    
            formula_uv = str(model_sansprepro.intercept_)
            for f,c in zip(X.columns, model_sansprepro.coef_):
                formula_uv = formula_uv + "+(" + str(c) + ")*[" + dico_map_tn_to_tmgl[f] + "]"
            self.formula_uv = formula_uv


    def format_coef(self,coef):

        import numpy as np
        SEUILS_VIRGULE = [(-np.inf,0.000001,"{coef:.10f}"),
                        (0.000001,0.00001,"{coef:.9f}"),
                        (0.00001,0.0001,"{coef:.7f}"),
                        (0.0001,0.001,"{coef:.6f}"),
                        (0.001,0.01,"{coef:.5f}"),
                        (0.01,0.1,"{coef:.4f}"),
                        (0.1,10.0,"{coef:.3f}"),
                        (10.0,1000.0,"{coef:.2f}"),
                        (1000.0,np.inf,"{coef:.1f}")]
        

        for stuple in SEUILS_VIRGULE:
            min_v = stuple[0]
            max_v = stuple[1]
            txt   = stuple[2]
            if np.abs(coef) < max_v and np.abs(coef) >= min_v:
        
                coef_str = txt.format(coef=coef)
                break

        return coef_str