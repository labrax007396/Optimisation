
import os,sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)

from models.ModeleGenerique import GenericModel
from interpretation.shapeley import Interpreteur

import pandas as pd
import tempfile

class CreateFigure:

    def __init__(self, model_obj:GenericModel) -> None:
        self.model_obj  = model_obj
        self.curr_rep   = os.path.dirname(os.path.realpath(__file__))
        self.FigureCreated = True
        self.LIST_OF_INTERPRETABLES_MODELS = ["RegressionLineaire","RegressionLgbm","RegressionRandomForest","RegressionXGBoost"]
            
  

    def delete_files_in_directory(self,directory_path):

        try:
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except OSError:
            pass

    def SavePlotlyToDisk(self,fig,file):



        f_height = fig.layout.height
        f_width  = fig.layout.width
        
        fig.write_image(file,scale=5.0,height=f_height,width=f_width)

    
    def CreateAllFigures(self):

        import os

        self.model_obj.data_obj.ReduceCorpusSize()

        
        g_correlations           = self.correlation()
        g_histo_num, g_histo_cat = self.histo()
        g_correlogramme          = self.correlogramme()
        g_sensibilite            = self.sensibilite()
        g_interp                 = self.interpretation()


        # Création du répertoire où seront stockées les images

        self.fig_rep = tempfile.mkdtemp(prefix="images_",suffix="_report")

        #print(self.fig_rep)

        self.PlotModVsMes()

        self.SavePlotlyToDisk(g_histo_num,os.path.join(self.fig_rep,'histo_num.png'))

        if g_histo_cat is not None:
            self.SavePlotlyToDisk(g_histo_cat,os.path.join(self.fig_rep,'histo_cat.png'))

        self.SavePlotlyToDisk(g_correlations,os.path.join(self.fig_rep,'graphe_correlations.png'))    

        g_correlogramme.savefig(os.path.join(self.fig_rep,'correlogramme.png'),bbox_inches='tight',dpi=500)


        self.SavePlotlyToDisk(g_sensibilite,os.path.join(self.fig_rep,'graphe_sensibilite.png'))  

        if g_interp:
            for type, g in g_interp.items():
                rep_cur_fig = os.path.join(self.fig_rep,type+'.png')
                g.savefig(rep_cur_fig,bbox_inches='tight',dpi=500, facecolor=g.get_facecolor())
        

    def interpretation(self):  

        model_options = self.model_obj.data_obj.GetModelOptions()



        if model_options['model_type'] in self.LIST_OF_INTERPRETABLES_MODELS:

            interpreteur_obj = Interpreteur(model_obj = self.model_obj, 
                                            data_obj = self.model_obj.data_obj,
                                            rapport_obj = self.model_obj.rapport_obj)
            interpreteur_obj.ComputeShapeleyValues()
            dico_graphes = interpreteur_obj.CreateGraphes()

        else:
            dico_graphes = None

        return dico_graphes

    def sensibilite(self):

        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rapport_mod = self.model_obj.rapport_obj.GetJsonReportAsDict()
        header      = self.model_obj.data_obj.GetHeader()

        dico_features = rapport_mod['dataframe_info']['features']

        list_fact_num = [d['description'] for d in dico_features if d['continous_serie_info'] is not None and d['used']]
        sens_fact_num = [d['continous_serie_info']['influence_weight'] for d in dico_features if d['continous_serie_info'] is not None and d['used']]
        df_sens_fact_num = pd.DataFrame(data={'Facteur':list_fact_num,'Sensibilite':sens_fact_num})
        df_sens_fact_num.sort_values(by=['Sensibilite'], ascending=False,inplace=True)

        dico_fact_disc = {d['description']:d['discrete_serie_info']['categorical_variables'] for d in dico_features if d['discrete_serie_info'] is not None and d['used']}
        
        
        if dico_fact_disc:

            dico_df_sens_fact_disc = dict()

            for fd, list_dico_sens in dico_fact_disc.items():
                list_modalites   = [d['name'] for d in list_dico_sens]
                list_sensibilite = [d['influence_weight'] for d in list_dico_sens]
                df_sens_fact_disc = pd.DataFrame(data={'Facteur':list_modalites,'Sensibilite':list_sensibilite})
                df_sens_fact_disc.sort_values(by=['Sensibilite'], ascending=False,inplace=True)
                dico_df_sens_fact_disc[fd] = df_sens_fact_disc

            nb_graphe = 1+len(dico_df_sens_fact_disc)
            title_list = ["Facteurs numériques"]
            for fd in dico_df_sens_fact_disc.keys():
                title_list.append("Facteur discret: "+fd)



            fig = make_subplots(
                rows=nb_graphe, cols=1,vertical_spacing=0.15,
                subplot_titles=tuple(title_list))

            units = [header['Unit'][0]]*nb_graphe

            fig.add_trace(go.Bar(x=df_sens_fact_num['Facteur'], y=df_sens_fact_num['Sensibilite']),row=1, col=1)

            ng = 1
            for fd, df_sens_disc in dico_df_sens_fact_disc.items():
                ng = ng+1
                fig.add_trace(go.Bar(x=df_sens_disc['Facteur'], y=df_sens_disc['Sensibilite']),row=ng, col=1)


            for i, unit in enumerate(units): 
                fig['layout']['yaxis{}'.format(i+1)]['title'] = unit

            fig.update_annotations(font_size=20)

            fig.update_layout(height=600*nb_graphe, 
                            width=1200,showlegend=False,
                            paper_bgcolor="#c6dee8",
                            font=dict(size=16),
                            title_text="")


        else:

            fig = go.Figure(go.Bar(x=df_sens_fact_num['Facteur'], y=df_sens_fact_num['Sensibilite']))
            fig.update_annotations(font_size=20)

            fig.update_layout(height=1200, 
                                width=1200,
                                showlegend=False,
                                yaxis_title = header['Unit'][0],
                                paper_bgcolor="#c6dee8",
                                font=dict(size=16),
                                title_text="")
            
        return fig
        




    def correlogramme(self):
            
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sn

        data = self.model_obj.data_obj.GetData()
        header = self.model_obj.data_obj.GetHeader()

        dico_map = {t:d for t,d,type in zip(header['Tagname'], header['Description'], header['MeasureDataType']) if type=='Numeric'}

        data_num = data[list(dico_map.keys())]

        fig_size = data_num.shape[1]
        plt.figure(figsize = (fig_size,fig_size),facecolor="#c6dee8")
        corrMatrix = data_num.corr()


        #corrMatrix.rename(columns=dico_map,index=dico_map, inplace=True)

        ax = plt.axes()
        mask = np.triu(np.ones_like(corrMatrix))

        plot_coefcor = sn.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1, center=0,mask=mask,
                                cmap=sn.diverging_palette(20, 220, n=200),
                                square=True, ax = ax)
        plt.xticks(fontsize = 9)
        plt.yticks(fontsize = 9)



        cbar = plot_coefcor.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)


        fig = plot_coefcor.get_figure()
        
        return fig

    def correlation(self):

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        data = self.model_obj.data_obj.GetData()
        header = self.model_obj.data_obj.GetHeader()

        var_numerique_alias = [n for n,type in zip(header['Tagname'],header['MeasureDataType']) if type=='Numeric']
        var_numerique       = [n for n,type in zip(header['Description'],header['MeasureDataType']) if type=='Numeric']
        #list_unit           = [n for n,type in zip(header['Unit'],header['MeasureDataType']) if type=='Numeric']

        data_num = data[var_numerique_alias]

        data_num.columns = var_numerique

        n_var_num = len(var_numerique)



        if n_var_num%2 == 0:
            nrows = int(n_var_num/2)
        else:
            nrows = int((n_var_num+1)/2)-1



        fig = make_subplots(rows=nrows, cols=2, subplot_titles=tuple(var_numerique[1:]),vertical_spacing=0.045,horizontal_spacing=0.075)

        if n_var_num%2 == 0:
            for row in range(nrows-1):
                fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+1]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=1)
                fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+2]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=2)
                fig.update_yaxes(title_text=var_numerique[0], row=row+1, col=1)
                fig.update_yaxes(title_text=var_numerique[0], row=row+1, col=2)

            fig.add_trace(go.Scatter(x=data_num[var_numerique[n_var_num-1]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=int(n_var_num/2), col=1)
            fig.update_yaxes(title_text=var_numerique[0], row=int(n_var_num/2), col=1)
            

        else:
            for row in range(nrows):
                fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+1]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=1)
                fig.add_trace(go.Scatter(x=data_num[var_numerique[2*row+2]], y=data_num[var_numerique[0]],mode='markers',showlegend=False),row=row+1, col=2)
                fig.update_yaxes(title_text=var_numerique[0], row=row+1, col=1)
                    

        fig.update_annotations(font_size=15)
        fig.update_layout(
            title=dict(text="Visualisation des corrélations entre variable modélisée et facteurs", font=dict(size=16)),
            title_x=0.5,
            paper_bgcolor="#c6dee8",
            autosize=False,
            width=1000,
            height=nrows*400)

        return fig
        


    def histo(self):
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        data = self.model_obj.data_obj.GetData()
        header = self.model_obj.data_obj.GetHeader()

        var_numerique_alias = [n for n,type in zip(header['Tagname'],header['MeasureDataType']) if type=='Numeric']
        var_numerique_desc  = [n for n,type in zip(header['Description'],header['MeasureDataType']) if type=='Numeric']
        list_unit           = [n for n,type in zip(header['Unit'],header['MeasureDataType']) if type=='Numeric']



        data_num = data[var_numerique_alias]

        n_var_num = len(var_numerique_alias)

        if n_var_num%2 == 0:
            nrows = int(n_var_num/2)
        else:
            nrows = int((n_var_num+1)/2)

        fig_histo = make_subplots(rows=nrows, cols=2,subplot_titles=tuple(var_numerique_desc),vertical_spacing=0.1,horizontal_spacing=0.075)

        if n_var_num%2 == 0:
            for row in range(nrows):
                fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row]]),row=row+1, col=1)
                fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row+1]]),row=row+1, col=2)
        else:
            for row in range(nrows-1):
                fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row]]),row=row+1, col=1)
                fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*row+1]]),row=row+1, col=2)

            fig_histo.add_trace(go.Histogram(x=data_num[var_numerique_alias[2*nrows-2]]),row=nrows, col=1)
            

        fig_histo.update_annotations(font_size=15)
        fig_histo.update_layout(
            title=dict(text="Histogramme des variables numériques", font=dict(size=16)),
            title_x=0.5,
            paper_bgcolor="#c6dee8",
            autosize=False,
            width=1000,
            showlegend=False,
            height=nrows*300)
        
        

        for i, unit in enumerate(list_unit): 
            fig_histo['layout']['xaxis{}'.format(i+1)]['title']=unit


        var_disc_alias = [n for n,type in zip(header['Tagname'],header['MeasureDataType']) if type!='Numeric']
        var_disc_desc  = [n for n,type in zip(header['Description'],header['MeasureDataType']) if type!='Numeric']
        list_unit      = [n for n,type in zip(header['Unit'],header['MeasureDataType']) if type!='Numeric']
 
        if len(var_disc_alias)>0:


            liste_df_var_cat = list()

            for var_cat in var_disc_alias:
                df_perc_per_cat = 100*data.groupby([var_cat])[var_cat].count()/len(data)
                df_perc_per_cat = df_perc_per_cat.sort_values(ascending = False)
                liste_df_var_cat.append(df_perc_per_cat)


            n_var_cat = len(var_disc_alias)

            fig = make_subplots(rows=n_var_cat, cols=1,subplot_titles=tuple(var_disc_desc))


            for row,df in enumerate(liste_df_var_cat):
                fig.add_trace(go.Bar(y=df.values,x=df.index.to_list()),row=row+1, col=1)
                fig.update_layout(xaxis_tickangle=-45)


                

            fig.update_annotations(font_size=12)
            fig.update_layout(
                title_text="Répartition des variables discrètes",
                paper_bgcolor="#c6dee8",
                title_x=0.5,
                autosize=False,
                showlegend=False,
                width=1000,
                height=nrows*200)

            for i, unit in enumerate(var_disc_alias): 
                fig['layout']['yaxis{}'.format(i+1)]['title']="Occurence (%)"


        else:
            fig = None


        return fig_histo, fig











    def PlotModVsMes(self):

        import matplotlib.pyplot as plt
        import matplotlib
        #%matplotlib inline
        # Turn interactive plotting off
        plt.ioff()
        model = self.model_obj.GetModel()

        self.model_obj.ReduceCorpusSize()

        train_data_set = self.model_obj.GetTrainData()
        test_data_set  = self.model_obj.GetTestData()

        fig = plt.figure(figsize=(8,8),facecolor="#c6dee8")
        

        target = train_data_set.columns[0]


        if test_data_set is not None:
            train_x = train_data_set.drop(target,axis=1)
            train_y = train_data_set[target]
            test_x  = test_data_set.drop(target,axis=1)
            test_y  = test_data_set[target]    
            y_train_modelise = model.predict(train_x)
            y_test_modelise  = model.predict(test_x)

            plt.scatter(train_y, y_train_modelise, marker= 'o', s=30, alpha=0.8,label='Apprentissage')
            plt.scatter(test_y, y_test_modelise, marker= 's',color='green', s=30, alpha=0.8,label='Validation')

        else:
            
            train_x = train_data_set.drop(target,axis=1)
            train_y = train_data_set[target]    
            y_train_modelise = model.predict(train_x)

            plt.scatter(train_y, y_train_modelise, marker= 'o', s=30, alpha=0.8,label='Apprentissage')

        plt.plot([0,train_y.max()], [0,train_y.max()], 'r-')


        
        plt.xlabel('Mesure')
        plt.ylabel('Modèle')
        plt.axis('equal')
        plt.axis([0,train_y.max(), 0, train_y.max()])
        plt.legend()
        #plt.show()
        plt.title("Modèle en fonction de la mesure", fontsize=16)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)   

        plt.xlabel('Mesure', fontsize=14)
        plt.ylabel('Modèle', fontsize=14)


        rep_cur_fig = os.path.join(self.fig_rep,'model_vs_mesure.png')
        fig.savefig(rep_cur_fig, dpi=500, facecolor=fig.get_facecolor())

        #plt.savefig(rep_cur_fig, dpi=500)
        #g_model_mesure.savefig(rep_cur_fig,bbox_inches='tight')

        #return fig
