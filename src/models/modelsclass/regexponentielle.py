import pandas as pd
from ..ModeleGenerique import GenericModel
from sklearn.base import BaseEstimator


class RegressionExp(GenericModel):

    def __init__(self, data_obj):
        GenericModel.__init__(self, data_obj)


    def BuildModel(self):

        params = {'a':30000,'b':-0.8,'c':20.0}
        self.ModelSklearn = PowerRegression(params['a'],params['b'],params['c'])
        self.IsBuild = True



    def Learn(self, model_options=dict, data=pd.DataFrame):

        data.dropna(inplace=True)
        target = data.columns[0]

        data = data[(data[target]>0) & (data[data.columns[1]]>0)]

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
        self.formula_uv = str(self.ModelSklearn.a)+'*[' + mglX + '].Pow(' + str(self.ModelSklearn.b) + ')+(' + str(self.ModelSklearn.c) + ')'
        self.formula    = self.formula_uv.replace(mglX,train_x.columns[0])

  

    def GetTrainData(self):
        return self.data_obj.GetData()



class PowerRegression(BaseEstimator):

    ''' Fittage regression exponentielle y = a*x**b+c '''

    def __init__(self, a,b,c):
        self.a_init  = a
        self.b_init  = b
        self.c_init  = c
        self.isfitted = False

    def fit(self, X, y):
       
        import warnings
        warnings.simplefilter('ignore')
        from scipy.optimize import curve_fit
        import numpy as np
        # détermination des des paramètres initiaux avec une régression linéaire simple
        from sklearn.linear_model import LinearRegression

        p_init = [self.a_init,self.b_init,self.c_init]

        X = X[X.columns[0]]

        X_log = X.apply(np.log)
        y_log = y.apply(np.log)

        reglin = LinearRegression().fit(X_log.values.reshape(-1, 1),y_log)
        A_INI = np.exp(reglin.intercept_)
        B_INI = reglin.coef_[0]
        X_max = X.max()
        C_INI = A_INI*X_max**B_INI
        p_init = [A_INI,B_INI,C_INI]


        popt, _ = curve_fit(self.objective, X.values,y.values,p0=p_init)
        self.a, self.b, self.c = popt

        self.X_ = X
        self.y_ = y
        self.isfitted = True
        #return self
    

    # objective function
    def objective(self,x, a, b,c):
        return a * x**b +c



    def predict(self, X):
        X = X[X.columns[0]]
        ypred = self.objective(X.values, self.a, self.b, self.c)
        return ypred

    def score(self,X,y):
        from sklearn.metrics import r2_score
        ypred = self.predict(X)

        return r2_score(y.values,ypred)
    
    def get_formula(self):
        formula = str(self.a)+'*[' + self.X_.name + '].Pow(' + str(self.b) + ')+(' + str(self.c) + ')'
        return formula

    def __repr__(self) -> str:
        if self.isfitted:
            model_str = self.get_formula()
        else:
            model_str = "Regression exponentielle"
        return model_str