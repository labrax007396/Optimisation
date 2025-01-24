import pandas as pd
from ..ModeleGenerique import GenericModel

from sklearn.linear_model import LinearRegression,Ridge

class RegressionPoly(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge

        super().BuildModel()

        self.BuildPrepro()

        self.ModelSklearn = Pipeline([("preprocessor",self.prepro),("model", LinearRegression())])
        self.IsBuild = True

        
    def BuildPrepro(self):


        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures 

        data   = self.data_obj.GetData()
        
        num_feat = [f for f in data.columns[1:] if data.dtypes[f]==np.float64]
        cat_feat = [f for f in data.columns[1:] if data.dtypes[f]==object]

        num_transformer   = Pipeline(steps=[('minmax', MinMaxScaler()),('poly',PolynomialFeatures(degree=2,interaction_only=False))])
     
        
        if len(cat_feat)>0:


            cat_transformer = Pipeline(steps=[('cat', OneHotEncoder(handle_unknown='ignore'))])

            self.prepro = ColumnTransformer(
                    remainder='passthrough', #passthough features not listed
                    transformers=[
                        ('num', num_transformer , num_feat),
                        ('cat', cat_transformer , cat_feat)
                    ])
       

        else:
            self.prepro = ColumnTransformer(remainder='passthrough', transformers=[('num', num_transformer , num_feat)])


    def CreateFormula(self):

        import numpy as np
        import math

        model = self.GetModel()

        formula = self.format_coef(model.named_steps['model'].intercept_)

        coefficients = list(model.named_steps['model'].coef_)
        features = list(model.named_steps['preprocessor'].get_feature_names_out())
        coefficients.pop(0)
        features.pop(0)

        for coef, feat in zip(coefficients, features):
            feat_renamed = feat.replace('num__','').replace('cat__','').replace(' ','*')

            if math.copysign(1,coef) > 0.0:
                op = '+'
            else:
                op = ''

            c_str = self.format_coef(coef)

            formula = formula + op + c_str + '*' + feat_renamed

        formula = formula.replace('-',' - ').replace('+',' + ').replace('*',' * ')
        if formula[0] == ' ':
            formula = formula[1:]


        data = self.data_obj.GetData()
        col_to_keep = [c for c in data.columns[1:] if data[c].dtypes=='float64']
        df_min_max = data[col_to_keep].agg(['min','max'])

        for fact_num in df_min_max.columns:
            v_min = df_min_max.loc['min',fact_num]
            v_max = df_min_max.loc['max',fact_num]
            denom = v_max-v_min
            denom_str = str(denom)
            if v_min == 0.0:
                num_str = fact_num
            else:
                if v_min<0:
                    num_str = '('+fact_num+ str(v_min)+')'
                else:
                    num_str = '('+fact_num+ '-' + str(v_min)+')'
            fact_norme = num_str+'/'+denom_str
            fact_norme = '('+fact_norme+')'
            formula = formula.replace(fact_num,fact_norme)

        self.formula = formula

 

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

    def FindHyperParams(self,data:pd.DataFrame):

        from sklearn.model_selection import GridSearchCV,train_test_split
        import numpy as np
       

        data = self.data_obj.GetData()

        data.dropna(inplace=True)

        X=data[data.columns[1:]]
        y=data[data.columns[0]]

        X_trs = self.prepro.fit_transform(X)

        search_grid = {'alpha':np.arange(0.1,10,0.5)}


        search = GridSearchCV(estimator=Ridge(),param_grid=search_grid,
                            cv = 5, scoring="r2",verbose=0)
        search.fit(X_trs,y)


        self.best_params = search.best_params_


    def Learn(self, model_options=dict, data=pd.DataFrame):
        
        from sklearn.model_selection import train_test_split

        data.dropna(inplace=True)
        target = data.columns[0]
        #X = data.drop(target,axis=1)
        #y = data[target]       
        #self.ModelSklearn.fit(X,y)
        #self.IsLearned = True



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
       