import os,sys

parent_dir = os.path.dirname(os.path.realpath(__file__))
src_dir    = os.path.dirname(parent_dir)
sys.path.append(src_dir)



from models.ModeleGenerique import GenericModel
from fpdf import FPDF
import shutil




list_txt_coef_corr = ["La matrice des corrélations permet de comprendre les relation entre les variables",
                      "Le coefficent est compris entre [-1 1]",
                      "Une valeur de 1 ou -1 indique une corrélation parfaite positive (respectivemment négative) entre 2 variables)"
]
txt = '\n'.join(list_txt_coef_corr)



class RapportPdf:

    def __init__(self, model_obj:GenericModel, figures_dir:str) -> None:
        self.model_obj = model_obj
        self.RepportCreated = True
        self.fig_rep = figures_dir

    def CreateReport(self):

    

        pdf = PDF(orientation="portrait",unit="mm",format="A4")

        pdf.add_page()

        # Infos générales

        pdf.chapter_title(1,'Informations Générales')

        d_infos = self.Dico_Infos()
        pdf.set_font('helvetica','',11)
        pdf.create_table(table_data = d_infos,cell_width=[40,80])
        pdf.ln()

        # Performances du modèle

        pdf.chapter_title(2,'Performances du modèle')
        d_perf = self.DicoPerf()
        pdf.set_font('helvetica','',11)
        pdf.create_table(table_data = d_perf,cell_width=[40,80])
        pdf.ln()

        path_cur_fig = os.path.join(self.fig_rep, 'model_vs_mesure.png') 

        y_pos = pdf.get_y()
        pdf.image(path_cur_fig,50,y_pos,90)


        # Facteurs

        dico_facteurs = self.DicoFacteurs()
        pdf.add_page()
        pdf.chapter_title(3,'Facteurs pertinents')
        pdf.set_font('helvetica','',11)


        list_txt = ["Les coefficients sont les valeurs des coefficients de la régression linéaire.",
               "La pvalue indique la probabilité que le coefficient soit nul (donc que le facteur soit non significatif).", 
               "Un seuil > 0.05 (5%) est la valeur généralement considérée pour estimer qu'un facteur est non significatif.",
               "Dans le cas où un des facteurs est de type catégoriel (décrit par une chaine de caractères), une transformation",
               "est préalablement effectuée pour encoder ce facteur avec des 0/1. Si le facteur contient N modalités,", 
               "N nouveaux facteurs sont créés. Si, par exemple, le facteur 'TYPE_PRODUIT' peut prendre les valeurs",  
               "'P1', 'P2' ou 'P3, les facteurs suivants sont créés: TYPE_PRODUIT [P1], TYPE_PRODUIT [P2], TYPE_PRODUIT [P3]]"]
        txt = '\n'.join(list_txt)

        if len(dico_facteurs) == 3:
            pdf.create_table(table_data = dico_facteurs,cell_width=[40,80,20])
        else:

            pdf.chapter_body(txt=txt,font='times',size=10,style='I')
            pdf.create_table(table_data = dico_facteurs,cell_width=[40,60,20,20,20])

        pdf.ln()

        # graphe de distribution des variables

        pdf.add_page()
        pdf.chapter_title(4,'Distribution des données')

        path_cur_fig = os.path.join(self.fig_rep, 'histo_num.png') 

        y_pos = pdf.get_y()

        img_w = 150
        x_pos = 30

        pdf.image(path_cur_fig,x_pos,y_pos,img_w)
        pdf.ln()
        y_pos = pdf.get_y()
        img_w = 130
        x_pos = 30
        #path_cur_fig = os.path.join(self.fig_rep,'histo_cat.png')

        #if os.path.isfile(path_cur_fig):
        #    pdf.image(path_cur_fig,x=x_pos,y=y_pos+135,w=150,h=90)
        # graphe des corrélations

        pdf.add_page()
        pdf.chapter_title(5,'Corrélations entre IPE et facteurs')      





        path_cur_fig = os.path.join(self.fig_rep,'graphe_correlations.png')
        y_pos = pdf.get_y()
        img_w = 150
        x_pos = 30
        pdf.image(path_cur_fig,x=x_pos,y=y_pos,h=200)
        pdf.ln()

        # corrélogramme

        pdf.add_page()
        pdf.chapter_title(6,'Matrice des corrélations')  



        list_txt_coef_corr = ["La matrice des corrélations permet de comprendre les relation entre les variables",
                            "Le coefficent est compris entre [-1 1]",
                            "Une valeur de 1 ou -1 indique une corrélation parfaite positive (respectivemment négative) entre 2 variables)"]
        txt = '\n'.join(list_txt_coef_corr)
        pdf.chapter_body(txt=txt,font='times',size=12,style='I')


        path_cur_fig = os.path.join(self.fig_rep,'correlogramme.png')
        y_pos = pdf.get_y()
        img_w = 150
        x_pos = 30
        pdf.image(path_cur_fig,x_pos,y_pos,img_w)
        pdf.ln()


        # formule du modèle

        pdf.add_page()
        pdf.chapter_title(7,'Formule')   
        formula = self.model_obj.formula
        if len(formula)<1:
            formula_lign = "Pas de formule possible pour ce type de modèle"
        else:
            header_data = self.model_obj.data_obj.GetHeader()
            IPE_desc = header_data['Description'][0]
            formula_lign = IPE_desc + ' (Modèle) = ' + formula        
             # formula_lign = formula.replace('+','+\n').replace('-','-\n')
             # if formula_lign[0:2] == '-\n':
              #    formula_lign = '-'+formula_lign[2:]




        pdf.chapter_body(txt=formula_lign,font='times',size=10,style='')

        # graphe des sensibilités

        pdf.add_page()
        pdf.chapter_title(8,'Sensibilité des facteurs')      
       
        lign1 = "Le graphe des sensibilités des facteurs donne une indication de l'importance des facteurs sur la variable modélisée."
        lign2 = "Pour un facteur donné, la valeur correpond à la réponse du modèle lorsque ce facteur varie entre 25% et 75% de sa plage, les autres facteurs étant pris à leurs valeurs moyennes."
        lign3 = "Par exemple: Si le facteur F varie entre 0 et 10 sur les données d'apprentissage,on a: F(25%) = 2.5 et F(75%) = 7.5,"
        lign4 = "Si le modèle = 20 pour F(25%) et modèle = 30 pour F(75%), alors la sensibilité = 30-20 = 10"

        txt = '\n'.join([lign1,lign2,lign3,lign4])

        pdf.chapter_body(txt=txt,font='times',size=12,style='I')

        path_cur_fig = os.path.join(self.fig_rep,'graphe_sensibilite.png')
        y_pos = pdf.get_y()
        img_w = 150
        x_pos = 30
        pdf.image(path_cur_fig,x_pos,y_pos,img_w)
        pdf.ln()

        path_cur_fig = os.path.join(self.fig_rep,'fig_importance_fact.png')

        if os.path.isfile(path_cur_fig):
            pdf.add_page()
            pdf.chapter_title(9,'Interprétation du modèle')      
        
            path_cur_fig = os.path.join(self.fig_rep,'fig_importance_fact.png')
            y_pos = pdf.get_y()
            img_w = 150
            x_pos = 30
            pdf.image(path_cur_fig,x_pos,y_pos,img_w)
            pdf.ln()

            pdf.add_page()
            path_cur_fig = os.path.join(self.fig_rep,'fig_shape_fact.png')
            y_pos = pdf.get_y()
            img_w = 150
            x_pos = 30
            pdf.image(path_cur_fig,x_pos,y_pos,img_w)
            pdf.ln()







        pdf.output('rapport_modelisation.pdf')
        #shutil.rmtree(self.fig_rep)

    def DicoFacteurs(self):

        import pandas as pd
        import math

        resu_model = self.model_obj.GetResuModel()
        stats_fact = resu_model['Statistiques']
        header     = self.model_obj.data_obj.GetHeader()
        model_options = self.model_obj.data_obj.GetModelOptions()

        #print(resu_model)

        dico_facteurs = dict()


        if model_options['model_type'] == "RegressionLineaire":

            df_stat_fact = pd.DataFrame(data={'Nom':stats_fact['facteurs'],'Coefficient':stats_fact['coefs'],'pvalue':stats_fact['pvalues']})
            dico_map_desc = {n:d for n,d in zip(header['Tagname'],header['Description'])}
            list_fact_full = df_stat_fact['Nom'].to_list()

            

            for n,d in dico_map_desc.items():
                list_fact_full = [l.replace(n,d) for l in list_fact_full]
            df_stat_fact['Description'] = list_fact_full

            dico_map_unit = {n:u for n,u in zip(header['Tagname'],header['Unit'])}
            df_stat_fact['Unité'] = df_stat_fact['Nom'].map(dico_map_unit, na_action='ignore')
            df_stat_fact['Unité'].fillna('',inplace=True)

            dico_facteurs['Nom'] = df_stat_fact['Nom'].values.tolist()
            dico_facteurs['Description'] = df_stat_fact['Description'].values.tolist()
            dico_facteurs['Unité'] = df_stat_fact['Unité'].values.tolist()
     
            coef_list = df_stat_fact['Coefficient'].values.tolist()
            dico_facteurs['Coefficient'] = ['{:.5f}'.format(coef) if math.modf(coef)[1] == 0 else '{:.2f}'.format(coef) for coef in coef_list]
            
            pvalue_list = df_stat_fact['pvalue'].values.tolist()
            dico_facteurs['pvalue'] = ['{:.3f}'.format(p) for p in pvalue_list]

            
        else:
            dico_facteurs['Nom'] = header['Tagname'][1:]
            dico_facteurs['Description'] = header['Description'][1:]
            dico_facteurs['Unité'] = header['Unit'][1:]



        return dico_facteurs




    def DicoPerf(self):

        json_rep = self.model_obj.rapport_obj.GetJsonReportAsDict()

        header_data = self.model_obj.data_obj.GetHeader()
        unit = header_data['Unit'][0]


        if json_rep['model_info']['r2_test'] is not None:
            r2 = "Apprentissage {v1:5.1f} Validation {v2:5.1f}".format(v1=100.0*json_rep['model_info']['r2_train'], v2=100.0*json_rep['model_info']['r2_test'])
        else:
            r2 = "{v1:5.1f} (%)".format(v1=100.0*json_rep['model_info']['r2_train'])

        if json_rep['model_info']['mape_test'] is not None:
            mape = "Apprentissage {v1:5.1f} Validation {v2:5.1f}".format(v1=json_rep['model_info']['mape_train'], v2=json_rep['model_info']['mape_test'])
        else:
            mape = "{v1:5.1f}".format(v1=json_rep['model_info']['mape_train'])

        if json_rep['model_info']['standard_deviation_test'] is not None:
            std = "Apprentissage {v1:5.1f} Validation {v2:5.1f}".format(v1=json_rep['model_info']['standard_deviation_train'], v2=json_rep['model_info']['standard_deviation_test'])
        else:
            std = "{v1:5.1f}".format(v1=json_rep['model_info']['standard_deviation_train'])


        d_perf = dict()

        d_perf['Type de modèle'] = ["Coefficient R2 (%)","Erreur moy. relative (%)","Erreur standard ("+unit+")"]

        d_perf[json_rep['model_info']['model_type']] = [r2,mape,std]

        return d_perf


    def Dico_Infos(self):

        from datetime import datetime 
        json_report = self.model_obj.rapport_obj.GetJsonReportAsDict()
        header_data = self.model_obj.data_obj.GetHeader()
        
        d_infos = dict()


        ParentScopeMangling_Splitted = header_data['ParentScopeMangling'][0].split(".")

        json_report['site'] = ParentScopeMangling_Splitted[0]
        client              = json_report['site']

        if len(ParentScopeMangling_Splitted) < 2:
            depart = client
        else:
            depart = ParentScopeMangling_Splitted[1]


        start_date = json_report['dataframe_info']['start_date']
        end_date   = json_report['dataframe_info']['end_date']

        start_date_obj =  datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
        start_date     =  start_date_obj.strftime("%d %b %Y %H:%M")
        end_date_obj   =  datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')
        end_date       =  end_date_obj.strftime("%d %b %Y %H:%M")

        list_facteurs_used = [d_f['name'] for d_f in json_report["dataframe_info"]["features"] if d_f['used']]

        d_infos['Client'] = ['Site','Variable modélisée','Fréquence','Période','Nombre de valeurs','Nombre de facteurs']
        d_infos[client]    = [depart,
                                header_data['Description'][0] + " (" + header_data['Unit'][0] + ")",
                                header_data['TagInfoFrequency'][0],
                                "Du " + start_date + " Au " + end_date,
                                str(json_report['dataframe_info']['cleaning_info']['line_count_after']),
                                str(len(list_facteurs_used))
                            ]

        return d_infos



class PDF(FPDF):

    def header(self):
        from datetime import datetime
        dir_path = os.path.dirname(os.path.realpath(__file__))
        icone_file = os.path.join(dir_path, "logouw.png")
        self.image(icone_file,10,8,35)
        self.set_text_color(128,128,128)
        self.set_font('helvetica','I',12)
        self.cell(0,10,'Rapport modélisation',align='C')
        now = datetime.now() # current date and time
        date_str = "Le "+now.strftime("%d/%m/%Y")
        self.set_font('helvetica','I',12)
        self.cell(0,10,date_str,ln=1,align='R')

        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica','I',10)
        self.cell(0,10,f'Page {self.page_no()}/{{nb}}',align='C')

    def chapter_title(self,ch_num, ch_title):
        self.set_font('helvetica','',12)
        self.set_fill_color(200,220,255)
        chapter_title = f'{ch_num} : {ch_title}'
        self.cell(0,5,chapter_title,ln=1,fill=1)
        self.ln()

    def chapter_body(self,txt:str,font:str,size:int,style:str):
        self.set_font(font,style,size)
        self.multi_cell(0,5,txt)
        self.ln()

    def create_table(self, table_data, title='', data_size = 10, title_size=12, align_data='L', align_header='L', cell_width='even', x_start='x_default',emphasize_data=[], emphasize_style=None,emphasize_color=(0,0,0)): 
        """
        table_data: 
                    list of lists with first element being list of headers
        title: 
                    (Optional) title of table (optional)
        data_size: 
                    the font size of table data
        title_size: 
                    the font size fo the title of the table
        align_data: 
                    align table data
                    L = left align
                    C = center align
                    R = right align
        align_header: 
                    align table data
                    L = left align
                    C = center align
                    R = right align
        cell_width: 
                    even: evenly distribute cell/column width
                    uneven: base cell size on lenght of cell/column items
                    int: int value for width of each cell/column
                    list of ints: list equal to number of columns with the widht of each cell / column
        x_start: 
                    where the left edge of table should start
        emphasize_data:  
                    which data elements are to be emphasized - pass as list 
                    emphasize_style: the font style you want emphaized data to take
                    emphasize_color: emphasize color (if other than black) 
        
        """
        default_style = self.font_style
        if emphasize_style == None:
            emphasize_style = default_style
        # default_font = self.font_family
        # default_size = self.font_size_pt
        # default_style = self.font_style
        # default_color = self.color # This does not work

        # Get Width of Columns
        def get_col_widths():
            col_width = cell_width
            if col_width == 'even':
                col_width = self.epw / len(data[0]) - 1  # distribute content evenly   # epw = effective page width (width of page not including margins)
            elif col_width == 'uneven':
                col_widths = []

                # searching through columns for largest sized cell (not rows but cols)
                for col in range(len(table_data[0])): # for every row
                    longest = 0 
                    for row in range(len(table_data)):
                        cell_value = str(table_data[row][col])
                        value_length = self.get_string_width(cell_value)
                        if value_length > longest:
                            longest = value_length
                    col_widths.append(longest + 4) # add 4 for padding
                col_width = col_widths



                        ### compare columns 

            elif isinstance(cell_width, list):
                col_width = cell_width  # TODO: convert all items in list to int        
            else:
                # TODO: Add try catch
                col_width = int(col_width)
            return col_width

        # Convert dict to lol
        # Why? because i built it with lol first and added dict func after
        # Is there performance differences?
        if isinstance(table_data, dict):
            header = [key for key in table_data]
            data = []
            for key in table_data:
                value = table_data[key]
                data.append(value)
            # need to zip so data is in correct format (first, second, third --> not first, first, first)
            data = [list(a) for a in zip(*data)]

        else:
            header = table_data[0]
            data = table_data[1:]

        line_height = self.font_size * 2.5

        col_width = get_col_widths()
        self.set_font(size=title_size)

        # Get starting position of x
        # Determin width of table to get x starting point for centred table
        if x_start == 'C':
            table_width = 0
            if isinstance(col_width, list):
                for width in col_width:
                    table_width += width
            else: # need to multiply cell width by number of cells to get table width 
                table_width = col_width * len(table_data[0])
            # Get x start by subtracting table width from pdf width and divide by 2 (margins)
            margin_width = self.w - table_width
            # TODO: Check if table_width is larger than pdf width

            center_table = margin_width / 2 # only want width of left margin not both
            x_start = center_table
            self.set_x(x_start)
        elif isinstance(x_start, int):
            self.set_x(x_start)
        elif x_start == 'x_default':
            x_start = self.set_x(self.l_margin)


        # TABLE CREATION #

        # add title
        if title != '':
            self.multi_cell(0, line_height, title, border=0, align='j', ln=3, max_line_height=self.font_size)
            self.ln(line_height) # move cursor back to the left margin

        self.set_font(size=data_size)
        # add header
        y1 = self.get_y()
        if x_start:
            x_left = x_start
        else:
            x_left = self.get_x()
        x_right = self.epw + x_left
        if  not isinstance(col_width, list):
            if x_start:
                self.set_x(x_start)
            for datum in header:
                self.multi_cell(col_width, line_height, datum, border=0, align=align_header, ln=3, max_line_height=self.font_size)
                x_right = self.get_x()
            self.ln(line_height) # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left,y1,x_right,y1)
            self.line(x_left,y2,x_right,y2)

            for row in data:
                if x_start: # not sure if I need this
                    self.set_x(x_start)
                for datum in row:
                    if datum in emphasize_data:
                        self.set_text_color(*emphasize_color)
                        self.set_font(style=emphasize_style)
                        self.multi_cell(col_width, line_height, datum, border=0, align=align_data, ln=3, max_line_height=self.font_size)
                        self.set_text_color(0,0,0)
                        self.set_font(style=default_style)
                    else:
                        self.multi_cell(col_width, line_height, datum, border=0, align=align_data, ln=3, max_line_height=self.font_size) # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                self.ln(line_height) # move cursor back to the left margin
        
        else:
            if x_start:
                self.set_x(x_start)
            for i in range(len(header)):
                datum = header[i]
                self.multi_cell(col_width[i], line_height, datum, border=0, align=align_header, ln=3, max_line_height=self.font_size)
                x_right = self.get_x()
            self.ln(line_height) # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left,y1,x_right,y1)
            self.line(x_left,y2,x_right,y2)


            for i in range(len(data)):
                if x_start:
                    self.set_x(x_start)
                row = data[i]
                for i in range(len(row)):
                    datum = row[i]
                    if not isinstance(datum, str):
                        datum = str(datum)
                    adjusted_col_width = col_width[i]
                    if datum in emphasize_data:
                        self.set_text_color(*emphasize_color)
                        self.set_font(style=emphasize_style)
                        self.multi_cell(adjusted_col_width, line_height, datum, border=0, align=align_data, ln=3, max_line_height=self.font_size)
                        self.set_text_color(0,0,0)
                        self.set_font(style=default_style)
                    else:
                        self.multi_cell(adjusted_col_width, line_height, datum, border=0, align=align_data, ln=3, max_line_height=self.font_size) # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                self.ln(line_height) # move cursor back to the left margin
        y3 = self.get_y()
        self.line(x_left,y3,x_right,y3)


'''
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph

class RapportPdf:

    def __init__(self, model_obj:GenericModel) -> None:
        self.model_obj = model_obj



    def CreateHeader(self,c):

        from reportlab.lib.pagesizes import letter
        from datetime import datetime
        import os

        rapport_dict = self.model_obj.rapport_obj.GetJsonReportAsDict()

        width, height = letter
        c.setStrokeColor("grey")
        c.setLineWidth(0.5)

        x_rect = 5
        y_rect = height-10
        w_rect = width-30
        h_rect = 50
        c.roundRect(x_rect, y_rect, w_rect, h_rect,5,stroke=1, fill=0)                

        x_text = x_rect+6
        y_text = y_rect+2/3*h_rect
        c.setFont('Times-Roman',10)
        c.setFillColor("grey")

        site_et_ipe = rapport_dict['site']+": Rapport modélisation de " + rapport_dict['dataframe_info']['target']['description']
        c.drawString(x_text,y_text,site_et_ipe)

        x_text = x_rect+6
        y_text = y_rect+1/3*h_rect

        now = datetime.now() # current date and time
        date_str = "Le "+now.strftime("%d/%m/%Y")
        c.drawString(x_text,y_text,date_str)
 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        icone_file = os.path.join(dir_path, "logouw.png")

        w_img = 100

        x_img = x_rect+w_rect-w_img-5
        y_img = y_rect-h_rect/2.0-5

        c.drawImage(icone_file, x_img, y_img, width=w_img, preserveAspectRatio=True, mask='auto')

            




    def CreateReport(self):

        # Entete

        from reportlab.pdfgen import canvas
        c = canvas.Canvas("rapport_modelisation.pdf")
        self.CreateHeader(c)

        # 1 ère page : info générales, synthèse et facteurs (3 tableaux)

        # Info générale
        x_title = 20
        y_cur = 700
  
        c.setFont('Times-Bold',14)
        c.setFillColor("cornflowerblue")        
        c.drawString(x_title,y_cur,"Informations générales")


        c.save()
        
'''
