import pandas as pd
import os
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np


#group together weekly csvs
#read the path
file_path = "2023/weekly data/WR/"
#list all the files from the directory
file_list = os.listdir(file_path)

df_append = pd.DataFrame()
#append all files together
for file in file_list:
            df_temp = pd.read_csv(file_path + file)
            df_append = df_append.append(df_temp, ignore_index=True)
#df_append

df_append.to_csv(fr'{file_path}\combined_wr.csv')

#format schedule
team_sh = pd.read_csv(f'2023/modified_nfl_schedule.csv', sep=',', encoding='utf-8',
    low_memory=False)

wk_dt = pd.read_csv(f'2023/week_date.csv', sep=',', encoding='utf-8',
    low_memory=False)
wk_dt['date'] = pd.to_datetime(wk_dt.date , format = '%Y-%m-%d %H:%M:%S')

df3 = pd.merge(team_sh, wk_dt, on=['week'], how='outer')
df3.to_csv(fr'2023\nfl_schedule_w_date.csv')

#DEFENSE
file_path = "2023/weekly data/DST/"
df = pd.read_csv(f'{file_path}/combined_dst.csv', sep=',', encoding='utf-8',
    low_memory=False, index_col=0)

df.rename({'FPTS/G': 'fpts_g', 'DEF TD': 'def_td', 'SPC TD': 'spc_td'}, axis=1, inplace=True)
#DST scoring formula
df['actual_pts'] = df.apply(lambda row: (row.INT *2) + (row.def_td *6) + (row.spc_td *6) 
                            + (row.SACK *1) + (row.FR *2) + (row.FF *1), axis=1)
df['team_abbr'] = df['Player'].str[-4:-1] 
df['team_abbr'] = df['team_abbr'].str.replace(r"(","")
#join w/ schedule
df1 = pd.merge(df, df3, on=['team_abbr', 'week'], how='outer')
df2 = df1[['Player', 'week', 'date', 'INT', 'def_td', 'spc_td', 'SACK', 'FR', 'FF',
           'actual_pts', 'team_abbr', 'opp_team']]
df2.to_csv(fr'{file_path}\combined_clean_dst.csv')

#KICKER
file_path = "2023/weekly data/K/"
df = pd.read_csv(f'{file_path}/combined_k.csv', sep=',', encoding='utf-8',
    low_memory=False, index_col=0)

df.rename({'19-Jan': 'y1_19', '20-29': 'y20_29',
     '30-39': 'y30_39', '40-49': 'y40_49', '50+': 'y50_plus',
           'FPTS/G': 'fpts_g'}, axis=1, inplace=True)
#K scoring formula
df['actual_pts'] = df.apply(lambda row: (row.y40_49 *4) + (row.y1_19 *3) + (row.y20_29 *3) 
                            + (row.y30_39 *3) + (row.y50_plus *5) + (row.XPT * 1)
                             + ((row.FGA - row.FG) * -1), axis=1)
df['team_abbr'] = df['Player'].str[-4:-1] 
df['team_abbr'] = df['team_abbr'].str.replace(r"(","")
#join w/ schedule
df1 = pd.merge(df, df3, on=['team_abbr', 'week'], how='outer')
df2 = df1[['Player', 'week', 'date', 'y40_49', 'y1_19', 'y20_29', 'y30_39', 'y50_plus', 'XPT',
           'FGA', 'FG', 'actual_pts', 'team_abbr', 'opp_team']]
df2.to_csv(fr'{file_path}\combined_clean_k.csv')

#QUARTERBACK
file_path = "2023/weekly data/QB/"
df = pd.read_csv(f'{file_path}/combined_qb.csv', sep=',', encoding='utf-8',
    low_memory=False, index_col=0)

df.rename({'ATT.1': 'att_r', 'YDS.1': 'ydr_r',
           'TD.1': 'td_r', 'Y/A': 'y_a',
           'FPTS/G': 'fpts_g'}, axis=1, inplace=True)
#QB scoring formula
df['actual_pts'] = df.apply(lambda row: (row.YDS *0.04) + (row.TD *4) + (row.INT *-2) + (row.ydr_r *0.1) + (row.FL *-2)
                            + (row.td_r *6), axis=1)

#df = df.loc[(df['G'] !=0)]

df['team_abbr'] = df['Player'].str[-4:-1] 
df['team_abbr'] = df['team_abbr'].str.replace(r"(","")
#join w/ schedule
df1 = pd.merge(df, df3, on=['team_abbr', 'week'], how='inner')
df2 = df1[['Player', 'week', 'date', 'YDS', 'TD', 'INT', 'ydr_r', 'FL', 'td_r', 
           'actual_pts', 'team_abbr', 'opp_team']]
df2.to_csv(fr'{file_path}\combined_clean_qb.csv')

#RUNNING BACK
file_path = "2023/weekly data/RB/"
df = pd.read_csv(f'{file_path}/combined_rb.csv', sep=',', encoding='utf-8',
    low_memory=False, index_col=0)

df.rename({'Y/A': 'y_per_att', '20+': 'y20_plus', 'YDS.1': 'y_rec', 'Y/R': 'y_per_rush',
           'TD.1': 'td_rec', 'FPTS/G': 'fpts_g'}, axis=1, inplace=True)

#RB scoring formula
df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.y_rec *0.1) + (row.REC *0.5) 
                            + (row.TD *6) + (row.td_rec *6) + (row.FL *-2), axis=1)

df['team_abbr'] = df['Player'].str[-4:-1] 
df['team_abbr'] = df['team_abbr'].str.replace(r"(","")
#join w/ schedule
df1 = pd.merge(df, df3, on=['team_abbr', 'week'], how='inner')
df2 = df1[['Player', 'week', 'date', 'YDS', 'y_rec', 'REC', 'TD', 'td_rec', 'FL', 
           'actual_pts', 'team_abbr', 'opp_team']]
df2.to_csv(fr'{file_path}\combined_clean_rb.csv')

#TIGHT END
file_path = "2023/weekly data/TE/"
df = pd.read_csv(f'{file_path}/combined_te.csv', sep=',', encoding='utf-8',
    low_memory=False, index_col=0)
df.rename({'Y/R': 'y_per_rec', '20+': 'y20_plus', 'YDS.1': 'y_rush', 
           'TD.1': 'td_rec', 'FPTS/G': 'fpts_g'}, axis=1, inplace=True)
#TE scoring formula
df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.TD *6) + (row.REC *0.5) 
                            + (row.FL *-2), axis=1)

df['team_abbr'] = df['Player'].str[-4:-1] 
df['team_abbr'] = df['team_abbr'].str.replace(r"(","")
#join w/ schedule
df1 = pd.merge(df, df3, on=['team_abbr', 'week'], how='inner')
df2 = df1[['Player', 'week', 'date', 'YDS', 'TD', 'REC', 'FL',  
           'actual_pts', 'team_abbr', 'opp_team']]
df2.to_csv(fr'{file_path}\combined_clean_te.csv')

#WIDE RECEIVER
file_path = "2023/weekly data/WR/"
df = pd.read_csv(f'{file_path}/combined_wr.csv', sep=',', encoding='utf-8',
    low_memory=False, index_col=0)

df.rename({'Y/R': 'y_per_r', '20+': '20_plus',
     'YDS.1': 'ydr_rush', 'TD.1': 'td_r', 'Y/A': 'y_a',
           'FPTS/G': 'fpts_g'}, axis=1, inplace=True)
#WR scoring formula
df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.TD *6) + (row.REC *0.5) 
                            + (row.ydr_rush *0.1) + (row.FL *-2), axis=1)

df['team_abbr'] = df['Player'].str[-4:-1] 
df['team_abbr'] = df['team_abbr'].str.replace(r"(","")
#join w/ schedule
df1 = pd.merge(df, df3, on=['team_abbr', 'week'], how='inner')
df2 = df1[['Player', 'week', 'date', 'YDS', 'TD', 'REC', 'ydr_rush', 'FL',  
           'actual_pts', 'team_abbr', 'opp_team']]
#data = df2.loc[(df2['Player'] == 'Tyreek Hill (MIA)')]
#data.head(20)
df2.to_csv(fr'{file_path}\combined_clean_wr.csv')
