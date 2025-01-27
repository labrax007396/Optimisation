import pandas as pd
from ..ModeleGenerique import GenericModel
from sklearn.base import BaseEstimator


class RegressionPuissance(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):

        params = {'a':30000,'b':-0.8}
        self.ModelSklearn = PuissanceRegression(params['a'],params['b'])
        self.IsBuild = True



    def Learn(self, model_options=dict, data=pd.DataFrame):

        data.dropna(inplace=True)
        target = data.columns[0]
        X = data.drop(target,axis=1)
        y = data[target]       
        self.ModelSklearn.fit(X,y)
        self.IsLearned = True
           
    def CreateFormula(self):

        import numpy as np
        import math

        dico_map_tn_to_tmgl = dict()
        header = self.data_obj.GetHeader()

        for tn, pscope, freq, role in zip(header['Tagname'], header['ParentScopeMangling'], header['TagInfoFrequency'],header['TagInfoRole']):
            t_role = '' if role=='Data' else '.'+role
            tag_mgl = pscope + '.' + tn + '.' + freq + t_role
            dico_map_tn_to_tmgl[tn] = tag_mgl 


        data = self.data_obj.GetData()
        target = data.columns[0]
        train_data_set = data.copy()
        train_x = train_data_set.drop(target,axis=1)
        mglX = dico_map_tn_to_tmgl[train_x.columns[0]]
        self.formula_uv = str(self.ModelSklearn.a)+'*[' + mglX + '].Pow(' + str(self.ModelSklearn.b) + ')'
        self.formula    = self.formula_uv.replace(mglX,train_x.columns[0])

  

    def GetTrainData(self):
        return self.data_obj.GetData()



class PuissanceRegression(BaseEstimator):

    ''' Fittage regression exponentielle y = a*x**b '''

    def __init__(self, a,b):
        self.a_init  = a
        self.b_init  = b
        self.isfitted = False

    def fit(self, X, y):
       
        import warnings
        warnings.simplefilter('ignore')
        from scipy.optimize import curve_fit
        import numpy as np
        # détermination des des paramètres initiaux avec une régression linéaire simple
        from sklearn.linear_model import LinearRegression

        p_init = [self.a_init,self.b_init]

        X = X[X.columns[0]]

        X_log = X.apply(np.log)
        y_log = y.apply(np.log)

        reglin = LinearRegression().fit(X_log.values.reshape(-1, 1),y_log)
        self.a = np.exp(reglin.intercept_)
        self.b = reglin.coef_[0]

        self.X_ = X
        self.y_ = y
        self.isfitted = True
    

    # objective function
    def objective(self,x, a, b):
        return a * x**b



    def predict(self, X):
        X = X[X.columns[0]]
        ypred = self.objective(X.values, self.a, self.b)
        return ypred

    def score(self,X,y):
        from sklearn.metrics import r2_score
        ypred = self.predict(X)

        return r2_score(y.values,ypred)
    
    def get_formula(self):
        formula = str(self.a)+'*[' + self.X_.name + '].Pow(' + str(self.b) + ')'
        return formula

    def __repr__(self) -> str:
        if self.isfitted:
            model_str = self.get_formula()
        else:
            model_str = "Regression puissance"
        return model_str