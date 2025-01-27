from typing import List, Optional
from datetime import datetime


class CleaningInfo:
    line_count_before: int
    line_count_after: int

    def __init__(self, line_count_before: int, line_count_after: int) -> None:
        self.line_count_before = line_count_before
        self.line_count_after = line_count_after
    def getdico(self):
        return {"line_count_before":self.line_count_before,"line_count_after":self.line_count_after}


class ContinousSerieInfo:
    vmin: float
    vmax: float
    mean: float
    standard_deviation: float
    influence_weight: float
    missingdata: int
    outliers: int


    def __init__(self, vmin: float, vmax: float, mean: float, standard_deviation: float, influence_weight: float, missingdata: int, outliers:int) -> None:
        self.min = vmin
        self.max = vmax
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.influence_weight = influence_weight
        self.missingdata = missingdata
        self.outliers = outliers

    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)



class CategoricalVariable:
    name: str
    occurrences: int
    importance_percent: float

    def __init__(self, name: str, occurrences: int, influence_weight: float) -> None:
        self.name = name
        self.occurrences = occurrences
        self.influence_weight = influence_weight

    def getdico(self):
        return {"name": self.name, "occurrences":self.occurrences, "influence_weight":self.influence_weight}


class DiscreteSerieInfo:
    categorical_variables: List[CategoricalVariable]

    def __init__(self, categorical_variables: List[CategoricalVariable]) -> None:
        self.categorical_variables = categorical_variables

    def getlist(self):
        list_ = list()

        for modalite in self.categorical_variables:
            list_.append(modalite.getdico())
        return list_



class WeightVariable:
    name: str
    weigth: float


    def __init__(self, name: str, weight: float) -> None:
        self.name = name
        self.weight = weight


class WeightVariableSerieInfo:
    weigth_variables: List[WeightVariable]

    def __init__(self, weigth_variables: List[WeightVariable]) -> None:
        self.weigth_variables = weigth_variables








class Target:
    tag_mgl: str
    name: str
    description: str
    corr_coef: float
    used: bool
    discrete_serie_info: Optional[DiscreteSerieInfo]
    continous_serie_info: Optional[ContinousSerieInfo]

    def __init__(self, tag_mgl: str, name: str, description: str, corr_coef: float, used:bool, discrete_serie_info:Optional[DiscreteSerieInfo], continous_serie_info: Optional[ContinousSerieInfo]) -> None:
        self.tag_mgl = tag_mgl
        self.name = name
        self.description = description
        self.corr_coef = corr_coef
        self.used = used
        self.discrete_serie_info = discrete_serie_info
        self.continous_serie_info = continous_serie_info

    def getdico(self):
        dict_ = dict()
        dict_["tag_mgl"] = self.tag_mgl
        dict_["name"]   = self.name
        dict_["description"] = self.description
        dict_["corr_coeff"] = self.corr_coef
        dict_["used"] = self.used

        if self.discrete_serie_info is None:
            dict_["discrete_serie_info"] = None
        else:
            dict_["discrete_serie_info"] = dict()
            dict_["discrete_serie_info"]["categorical_variables"] = self.discrete_serie_info.getlist()
            #dict_["discrete_serie_info"] = self.discrete_serie_info.getlist()

        if self.continous_serie_info is None:
            dict_["continous_serie_info"] = None
        else:
            dict_["continous_serie_info"] = self.continous_serie_info.__dict__

        return dict_


    #def toJson(self):
    #    import json
    #    return json.dumps(self, default=lambda o: o.__dict__)


class DataframeInfo:
    start_date: str
    end_date: str
    cleaning_info: CleaningInfo
    target: Target
    features: List[Target]
    corr_matrix: dict

    def __init__(self, start_date: str, end_date: str, cleaning_info: CleaningInfo, target: Target, features: List[Target],corr_matrix: dict) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.cleaning_info = cleaning_info
        self.target = target
        self.features = features
        self.corr_matrix = corr_matrix

    def getdico(self):
        dict_ = dict()
        dict_["start_date"] = self.start_date
        dict_["end_date"]   = self.end_date
        dict_["cleaning_info"] = self.cleaning_info.getdico()

        dict_["target"] = self.target.__dict__
        dict_["target"]["continous_serie_info"] = self.target.continous_serie_info.__dict__

        dict_["features"] = list()

        for feat in self.features:
            dict_["features"].append(feat.getdico())

        dict_["corr_matrix"] = self.corr_matrix


        return dict_


    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)


class ModelInfo:
    model_type: str
    description: str
    creation_date: datetime
    formula: str
    r2_test: float
    r2_train: float
    mape_train: float
    mape_test: float
    mean_deviation_train: float
    mean_deviation_test: float
    standard_deviation_train: float
    standard_deviation_test: float

    def __init__(self, model_type: str, description: str, creation_date: str, formula: str, r2_test: float, r2_train: float, mape_train: float, mape_test: float, mean_deviation_train: float, mean_deviation_test: float, standard_deviation_train: float, standard_deviation_test: float) -> None:

        self.model_type = model_type
        self.description = description
        self.creation_date = creation_date
        self.formula = formula
        self.r2_test = r2_test
        self.r2_train = r2_train
        self.mape_train = mape_train
        self.mean_deviation_train = mean_deviation_train
        self.standard_deviation_train = standard_deviation_train
        self.mape_test = mape_test
        self.mean_deviation_test = mean_deviation_test
        self.standard_deviation_test = standard_deviation_test        

    def toJson(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__)


class UVFormula:

    import pandas 
    formula: str

    def __init__(self, dico_model: dict, data: pandas.core.frame.DataFrame) -> None:

        formula = '[model] '

        for tag in dico_model['facteurs'].keys():
            if dico_model['facteurs'][tag]['used']:
                if dico_model['facteurs'][tag]['type'] == 'num':
                    nom_feat = dico_model['facteurs'][tag]['nom']
                    min_val = str(data[nom_feat].min())
                    max_val = str(data[nom_feat].max())
                    formula = formula + " .Arg(" + '"' + nom_feat +'"'+ ", [" + tag + "]"
                    formula = formula + ", " + min_val + ", " + max_val +  ")"

                elif dico_model['facteurs'][tag]['type'] == 'cat':
                    nom_feat = dico_model['facteurs'][tag]['nom']
                    mod_liste = list(data[nom_feat].unique())
                    mod_liste = '","'.join(map(str,mod_liste))
                    mod_liste = '"'+mod_liste+'"'
                    formula = formula + " .Arg(" + '"' + nom_feat +'"'+ ", [" + tag + "]"
                    formula = formula + ", " +mod_liste +  ")"


        formula = formula + " .Outputs(" + '"' + 'target' + '"' + ")"
        self.formula = formula

class ReportModel:
    site: str
    dataframe_info: DataframeInfo
    model_info: ModelInfo
    uv_formula: str

    def __init__(self, site: str, dataframe_info: DataframeInfo, model_info: ModelInfo, uv_formula: str) -> None:
        self.site = site
        self.dataframe_info = dataframe_info
        self.model_info = model_info
        self.uv_formula = uv_formula

    

    def myconverter(self,obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
      
        


    def toJson(self):
        import json
        rapport = dict()
        rapport["site"] = self.site
        rapport["dataframe_info"] = self.dataframe_info.getdico()
        rapport["model_info"] = self.model_info.__dict__
        rapport["uv_formula"] = self.uv_formula
        
        return json.dumps(rapport, default=self.myconverter) 