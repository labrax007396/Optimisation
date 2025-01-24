
class Phdformat:


    def __init__(self,file_location:str):
        self.file_location = file_location

    
    def read_file(self):

        import pandas as pd
        from datetime import datetime

        data_header = pd.read_csv(self.file_location,sep=',',usecols=[0],nrows=8,header=None)
        for idx, row in data_header.iterrows():
            is_date = self.can_convert_to_datetime(row[0])
            if is_date:
                nb_header_rows = idx
                break

        list_header = [ix for ix in range(nb_header_rows)]
        data = pd.read_csv(self.file_location,sep=',',header=list_header)

        header_list = data.columns.to_list()

        POSSIBLE_KEYS = ['Tagname','Description','ParentScopeMangling','MeasureDataType','TagInfoFrequency','TagInfoRole','Unit']

        cles = header_list[0]

        header = dict()
        for pos, key in enumerate(cles):
            if key in POSSIBLE_KEYS:
                header[key] = [head[pos] for head in header_list[1:]]
            header[key] = ['' if 'Unnamed:' in u else u for u in header[key]]

        list_keys = list(header.keys())

        for key in POSSIBLE_KEYS:
            if key not in list_keys:
                header[key] = ['']*(len(header_list)-1)  

        new_col_name = ['Date']+header['Tagname']
        data.columns = new_col_name

        self.header = header
        self.data = data


    def can_convert_to_datetime(self,date_string):
        from datetime import datetime
        try:
            datetime.strptime(date_string, "%Y/%m/%d %H:%M:%S")
            return True
        except ValueError:
            return False