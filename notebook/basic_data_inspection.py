from typing import Any
import pandas as pd
class basic_data_inspection:
    def data_info(self,df:pd.DataFrame)->Any:
        
        return df.info()
    
    def data_types(self,df):
        return  df.dtypes
    
    def NumericalSummaryStatistics(self,df):
        numerical_summmary = df.describe()
        return numerical_summmary
    
    def CategoricalSummaryStatistics(self,df):
        cat_summary = df.describe(include='O')
        return cat_summary


