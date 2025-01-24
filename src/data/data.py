import pandas as pd
from datetime import datetime

class Data:


    def __init__(self, data_file_location:str, 
                 config_location:str):

        
        self.data_file_location = data_file_location
        self.config_location = config_location 
        self.data = pd.DataFrame
        self.options = dict()



    def ReadModelingOptions(self):

        """
            Lecture du fichier .json contenant les options
        
        """

        import commentjson

        with open(self.config_location, encoding='utf-8') as file:
        
            self.options = commentjson.load(file)


    def ReadCSVData(self):

        """
            Lecture du fichier de données
        
        """

        # Lecture de l'entête

        var_modelisee = self.options['y_config']['name']
        facteurs      = [d for d in self.options['f_configs'].keys()]
        index_name    = self.options['dataformat']['index_name']
        col_to_import = [index_name] + [var_modelisee] + facteurs

        self.data = pd.read_csv(self.data_file_location,
                              sep     = self.options['dataformat']['sep'],
                              decimal = self.options['dataformat']['decimal'],
                              usecols = col_to_import
                              )
        
        if self.options['dataformat']['index_type'] == "date":
            date_format = self.options['dataformat']['date_format']
            self.data[index_name] = self.data[index_name].apply(lambda x: datetime.strptime(str(x), date_format))
            self.data.set_index(index_name,inplace=True)
            debut = self.options['debut']
            fin   = self.options['fin']
            self.data = self.data[(self.data.index>=debut) & (self.data.index<=fin)]


        for col in self.options['f_configs'].keys():
            type_c = self.options['f_configs'][col]['type']
            if type_c == 'num':
                self.data[col] = self.data[col].astype(float)
            elif type_c == 'disc':
                self.data[col] = self.data[col].astype(str)
        
        self.DeleteOutliers()

    def DeleteOutliers(self):

        for col in self.options['f_configs'].keys():
            type_c = self.options['f_configs'][col]['type']
            if type_c == 'num':
                vmin = self.options['f_configs'][col]['vmin']
                vmax = self.options['f_configs'][col]['vmax']
                self.data = self.data[ (self.data[col]>=vmin) & (self.data[col]<=vmax) ]
            if type_c == 'cat':
                modalites = self.options['f_configs'][col]['modalites']
                if type(modalites) == list:
                    self.data[self.data[col].isin(modalites)]




    def CheckCompatibilityOptions(self):

        ''' Vérifie la compatibilité des options de modélisation avec le nombre de facteurs
            Pour les modèles "RegressionExp" et "RegressionPuissance" il ne peut y avoir qu'un seul facteur
        '''

        model_option = self.GetModelOptions()
        model_type   = model_option["model_type"]
        nbre_features = len(model_option["factor_configs"])

        if model_type in ["RegressionExp","RegressionPuissance"]:
            if nbre_features > 1:
                message = "Pour le type de modèle " + model_type + " Un seul facteur est possible"
                assert False, message


    def CountOutliers(self, dico_seuils_data_tag, tag_name):

   

        if 'min' in dico_seuils_data_tag.keys():
            seuil_min = dico_seuils_data_tag['min']
            nb_val_1 = len(self.data[self.data[tag_name]<=seuil_min])
        else:
            nb_val_1 = 0

        if 'max' in dico_seuils_data_tag.keys():
            seuil_max = dico_seuils_data_tag['max']
            nb_val_2 = len(self.data[self.data[tag_name]>=seuil_max])
        else:
            nb_val_2 = 0


        if 'string_values' in dico_seuils_data_tag.keys():

           
            
            modalite_to_include = dico_seuils_data_tag['string_values']
            if len(modalite_to_include) > 0:
                 #nombre de valeur avec une modalité vide ''
                nb_val_missing = len(self.data[self.data[tag_name] == ''])
                nb_val_2 = len(self.data[~self.data[tag_name].isin(modalite_to_include)]) - nb_val_missing
                nb_val_1 = 0
            else:
                nb_val_2 = 0
                nb_val_1 = 0


        

        return nb_val_1 + nb_val_2

        '''
        except Exception as error:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            self.MessageLogger.write_msg('erreur',"Erreur comptage des ouliers: "+type(error).__name__ +" " +fname+ " Ligne "+ str(exc_tb.tb_lineno))
            self.options = dict()
            self.IsModelOption = False
        '''
    
  

    def CleanReport(self):

        columns = self.data.columns.to_list()
        dico_seuils_data = dict()
        dico_seuils_data[columns[0]] = self.options['ipe_config']

        for tag_name, d_fact_conf in zip(columns[1:],self.options['factor_configs']):
            dico_seuils_data[tag_name] = d_fact_conf


        clean_report = dict()
        clean_report['ndata_before_clean'] = len(self.data)
        clean_report['detail'] = dict()


        # Infos pour l'IPE
        clean_report['detail']['target'] = dict()
        tag_name = columns[0]

        tag_descr = self.data[tag_name].describe()
        clean_report['detail']['target']['missingdata'] = len(self.data) - tag_descr.loc['count']


        clean_report['detail']['target']['outliers']    = self.CountOutliers(dico_seuils_data[tag_name], tag_name)


        # Infos pour les facteurs
        clean_report['detail']['features'] = dict()

        for tag_name in columns[1:]:
            t = dico_seuils_data[tag_name]['tag_mgl']
            clean_report['detail']['features'][t] = dict()

            if self.data[tag_name].dtypes=='float64':
                tag_descr = self.data[tag_name].describe()
                clean_report['detail']['features'][t]['missingdata'] = len(self.data) - tag_descr.loc['count']

            if self.data[tag_name].dtypes=='object':
                clean_report['detail']['features'][t]['missingdata'] = len(self.data[self.data[tag_name] == ''])


            clean_report['detail']['features'][t]['outliers'] = self.CountOutliers(dico_seuils_data[tag_name], tag_name)
            #print(tag_name,clean_report['detail']['features'][t]['outliers'])

        # Suppression des outliers

        self.data.dropna(inplace=True)
        self.CheckRemainingData()

        
        for tag_name in columns:
            self.DeleteOutliers(dico_seuils_data[tag_name], tag_name)

        clean_report['ndata_after_clean']  = len(self.data)        

        self.clean_report = clean_report

        '''
        except Exception as error:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            self.MessageLogger.write_msg('erreur',"Erreur durant la création du clean report: "+type(error).__name__ +" " +fname+ " Ligne "+ str(exc_tb.tb_lineno))
            self.options = dict()
            self.IsModelOption = False        
        '''

    def CheckRemainingData(self):


        """
            Vérifie si il reste des données après la suppression des outliers
        
        """

        if self.data.empty:
            assert False, "Nombre de données nulle: vérifiez les Valeurs min/max de l'IPE ou des facteurs"
        elif self.data.shape[0]/(self.data.shape[1]-1) < 3:
           
            msg1 = "Le nombre de données (" + str(self.data.shape[0]) + ") est insuffisant compte tenu du nombre de facteurs (" + str(self.data.shape[1]-1) +")\n"
            msg2 = "Vérifiez les Valeurs min/max de l'IPE ou des facteurs ou supprimez des facteurs\n"

            assert False, msg1+msg2



    def ChangeTagname(self,tn):
        tn_new = [t if t.isalnum() else '_' for t in tn]
        tn_new = ''.join(tn_new)
        return tn_new
    


    def set_header_to_new_tagname(self):

        df = self.GetData()
        initial_tagname  = df.columns.to_list()
        new_tagname      = [self.dict_map_tagname_to_new[f] for f in initial_tagname]
        df.columns = new_tagname


        '''
        for init_tn in self.dict_map_tagname_to_new.keys():
            if init_tn in self.header['ipe_config']['tag_mgl']:
                self.header['ipe_config']['tag_mgl'] = self.header['ipe_config']['tag_mgl'].replace(init_tn,self.dict_map_tagname_to_new[init_tn] )
            for d_init_fn in self.header['factor_configs']:
                if init_tn in d_init_fn['tag_mgl']:
                    d_init_fn['tag_mgl'] = d_init_fn['tag_mgl'].replace(init_tn,self.dict_map_tagname_to_new[init_tn] )
        '''


    def set_header_to_init_tagname(self):

        df = self.GetData()
        modified_tagname  = df.columns.to_list()


        init_tagname       = [self.dict_map_tagname_to_init[f] for f in modified_tagname]
        df.columns = init_tagname

        '''
        for mod_tn in self.dict_map_tagname_to_init.keys():
            if mod_tn in self.header['ipe_config']['tag_mgl']:
                self.header['ipe_config']['tag_mgl'] = self.header['ipe_config']['tag_mgl'].replace(mod_tn,self.dict_map_tagname_to_init[mod_tn] )
            for d_mod_fn in self.header['factor_configs']:
                if mod_tn in d_mod_fn['tag_mgl']:
                    d_mod_fn['tag_mgl'] = d_mod_fn['tag_mgl'].replace(mod_tn,self.dict_map_tagname_to_init[mod_tn])
        '''

    def set_list_to_init_tagname(self, liste):

        new_list = list()
        for e in liste:
            if e in self.dict_map_tagname_to_init.keys():
                new_list.append(self.dict_map_tagname_to_init[e])
            else:
                new_list.append(e)
                
        return new_list

    def GetData(self):
        return self.data
    
    def SetData(self,data):
        self.data = data

    def GetHeader(self):
        return self.header
    
    def GetModelOptions(self):
        return self.options
    
    def GetCleanReport(self):
        return self.clean_report
    
    def ReduceCorpusSize(self):
        from random import seed
        from random import sample

        NMAX = 100000
        # seed random number generator
        seed(1)
        if self.data.shape[0] > NMAX:
            # prepare a sequence
            sequence = [i for i in range(self.data.shape[0])]
            # select a subset without replacement
            subset = sample(sequence, NMAX)
            self.data = self.data.iloc[subset]
