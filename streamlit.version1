import pandas as pd
import streamlit as st
import numpy as np
from pmdarima.arima import auto_arima
from PIL import Image

st.title('NFL Fantasy Football Forecasting Tool')
st.header('Football Player Stats*')
st.write('*2023 Projections Using Half PPR league Calculations')
image = Image.open('2023/ff.png')
st.image(image)

def load_data(option):
    st.snow()
    if format_func(option) == 'Defense/Special Teams':
        file_path = "2023/weekly data/DST/"
        df = pd.read_csv(f'{file_path}/combined_clean_dst.csv', sep=',', encoding='utf-8',
        low_memory=False, index_col=0)
    if format_func(option) == 'Kicker':
        file_path = "2023/weekly data/K/"
        df = pd.read_csv(f'{file_path}/combined_clean_k.csv', sep=',', encoding='utf-8',
        low_memory=False, index_col=0)
    if format_func(option) == 'Quarter Back':
        file_path = "2023/weekly data/QB/"
        df = pd.read_csv(f'{file_path}/combined_clean_qb.csv', sep=',', encoding='utf-8',
        low_memory=False, index_col=0)
    if format_func(option) == 'Running Back':
        file_path = "2023/weekly data/RB/"
        df = pd.read_csv(f'{file_path}/combined_clean_rb.csv', sep=',', encoding='utf-8',
        low_memory=False, index_col=0)
    if format_func(option) == 'Tight End':
        file_path = "2023/weekly data/TE/"
        df = pd.read_csv(f'{file_path}/combined_clean_te.csv', sep=',', encoding='utf-8',
        low_memory=False, index_col=0)
    if format_func(option) == 'Wide Receiver':
        file_path = "2023/weekly data/WR/"
        df = pd.read_csv(f'{file_path}/combined_clean_wr.csv', sep=',', encoding='utf-8',
        low_memory=False, index_col=0)
    return df

CHOICES = {1: "Defense/Special Teams", 2: "Kicker", 3: "Quarter Back", 
           4: "Running Back", 5: "Tight End", 6: "Wide Receiver"}

def format_func(option):
    return CHOICES[option]

#{option} would be #1, 2, etc.
option = st.sidebar.selectbox("Select Position:", options=list(CHOICES.keys()), format_func=format_func)
st.write(f"You have selected {format_func(option)} position")

df1 = load_data(option)

def player_formula(option):
    if format_func(option) == 'Defense/Special Teams':
        formula = ('(Interception *2) + (Def Touchdown *6) + (Sp Touchdown *6) + (Sack *1) + (Fumble Recovered *2) + (Forced Fumbles *1)')
    if format_func(option) == 'Kicker':
        formula = "(1-19 yards *3) + (20-29 yards *3) + (30-39 yards *3) + (40-49 yards *4) + (50 yards+ *5) + (Extra Points * 1) + ((Field Goal Attempted - Field Goal Made) * -1)"
    if format_func(option) == 'Quarter Back':
        formula = "(Passing Yards *0.04) + (Touchdown Pass *4) + (Interceptions *-2) + (Rushing Yards *0.1) + (Fumbles Lost *-2) + (Touchdown Rush *6)"
    if format_func(option) == 'Running Back':
        formula = "(Rushing Yards *0.1) + (Receiving Yards *0.1) + (Each Reception *0.5) + (Touchdown Rush *6) + (Touchdown Reception *6) + (Fumbles Lost *-2)"
    if format_func(option) == 'Tight End':
        formula = "(Rushing Yards *0.1) + (Touchdown Rush *6) + (Each Reception *0.5) + (Fumbles Lost *-2)"
    if format_func(option) == 'Wide Receiver':
        formula = "(Rushing Yards *0.1) + (Touchdown Rush *6) + (Each Reception *0.5) + (Rushing Yards *0.1) + (Fumbles Lost *-2)"
    return formula

st.write(f'{format_func(option)} Actual Points Formula:')
st.latex(fr'''Actual Points = {player_formula(option)}''')

player = df1['Player'].sort_values().unique()

selected_player = st.sidebar.selectbox(f'Select your {format_func(option)}', player)

st.write('You selected:', selected_player)

df2 = df1.loc[(df1['Player'] == selected_player)]
df2 = df2.sort_values(by=['date'])
st.write(f'{format_func(option)} Data:')
df2 = df2.reset_index(drop=True)
st.write(df2)

def player_data(option):
    if format_func(option) == 'Defense/Special Teams':
        ch_data = ['INT', 'def_td', 'spc_td', 'SACK', 'FR', 'FF']
    if format_func(option) == 'Kicker':
        ch_data = ['y40_49', 'y1_19', 'y20_29', 'y30_39', 'y50_plus', 'XPT', 'FGA', 'FG']
    if format_func(option) == 'Quarter Back':
        ch_data = ['YDS', 'TD', 'INT', 'ydr_r', 'FL', 'td_r']
    if format_func(option) == 'Running Back':
        ch_data = ['YDS', 'y_rec', 'REC', 'TD', 'td_rec', 'FL']
    if format_func(option) == 'Tight End':
        ch_data = ['YDS', 'TD', 'REC', 'FL']
    if format_func(option) == 'Wide Receiver':
        ch_data = ['YDS', 'TD', 'REC', 'ydr_rush', 'FL']
    return ch_data

ch_data = player_data(option)

st.line_chart(df2, y=ch_data, x='date')

def player_cs_data(option):
    if format_func(option) == 'Defense/Special Teams':
        cs_data = ['INT_csum', 'def_td_csum', 'spc_td_csum', 'SACK_csum', 'FR_csum', 'FF_csum']
    if format_func(option) == 'Kicker':
        cs_data = ['y40_49_csum', 'y1_19_csum', 'y20_29_csum', 'y30_39_csum', 'y50_plus_csum', 'XPT_csum', 'FGA_csum', 'FG_csum']
    if format_func(option) == 'Quarter Back':
        cs_data = ['YDS_csum', 'TD_csum', 'INT_csum', 'ydr_r_csum', 'FL_csum', 'td_r_csum']
    if format_func(option) == 'Running Back':
        cs_data = ['YDS_csum', 'y_rec_csum', 'REC_csum', 'TD_csum', 'td_rec_csum', 'FL_csum']
    if format_func(option) == 'Tight End':
        cs_data = ['YDS_csum', 'TD_csum', 'REC_csum', 'FL_csum']
    if format_func(option) == 'Wide Receiver':
        cs_data = ['YDS_csum', 'TD_csum', 'REC_csum', 'ydr_rush_csum', 'FL_csum']
    return cs_data

cs_data = player_cs_data(option)

def cum_sum_function(ch_data, df2):
    x=0
    for i in ch_data:
        df2[ch_data[x] + '_csum'] = df2.groupby(['Player'])[ch_data[x]].cumsum()
        x=x+1
    return df2

find_cs = cum_sum_function(ch_data, df2)
find_cs = find_cs.sort_values(by=['date'])
#st.write(find_cs)
st.write(f'Displaying Cumulative Sum Data:')
st.line_chart(find_cs, y=cs_data, x='date')

#training
find_cs_df = find_cs[['date', *cs_data]]
find_cs_df = find_cs_df.set_index('date')
#find_cs_df = find_cs_df.sort_index(inplace=True)
#st.write(find_cs_df)

trainmodel = st.button("Generate Forecast", type="primary")

if trainmodel:
    st.header("Training Maching Learning Model...")
    train = find_cs_df.iloc[:12]
    #st.write(train)
    test = find_cs_df.iloc[12:]
    #st.write(test)
    final_df = pd.DataFrame(index = test.index, columns=[test.columns])
    x=0
    for i in cs_data:

        model = auto_arima(train[cs_data[x]], trace=True, error_action='ignore', suppress_warnings=True)
        mod_fit = model.fit(train[cs_data[x]])
   
        forecast = mod_fit.predict(n_periods=len(test[cs_data[x]]))
        forecast_df = pd.DataFrame(forecast,index = test[cs_data[x]].index,columns=[cs_data[x]])
        final_df = final_df.join(forecast_df, how='left')
        x=x+1
    final_df1 = final_df.dropna(axis=1, how='all')
    final_df1['date'] = final_df1.index
    final_df1 = final_df1.reset_index(drop=True)
    find_cs_df['date'] = find_cs_df.index
    find_cs_df = find_cs_df.reset_index(drop=True)
    data1 = find_cs_df.merge(final_df1, how='outer')
    data2=data1.dropna(how='any')
    data2 = data2[['date', *cs_data]]
    data2 = data2.sort_values(by=['date'])
    data3 = data2.set_index('date').diff()
    data3[[*cs_data]] = data3[[*cs_data]].abs()
    data3[[*cs_data]] = data3[[*cs_data]].round()
    data3.columns = data3.columns.str.rstrip('_csum')
    #st.write(data3)
    data3 = data3.dropna()

    def calc_actual_pts(option, df):
        if format_func(option) == 'Defense/Special Teams':
            df['actual_pts'] = df.apply(lambda row: (row.INT *2) + (row.def_td *6) + (row.spc_td *6) 
                                + (row.SACK *1) + (row.FR *2) + (row.FF *1), axis=1)
        if format_func(option) == 'Kicker':
            df['actual_pts'] = df.apply(lambda row: (row.y40_49 *4) + (row.y1_19 *3) + (row.y20_29 *3) 
                                + (row.y30_39 *3) + (row.y50_pl *5) + (row.XPT * 1)
                                + ((row.FGA - row.FG) * -1), axis=1)
        if format_func(option) == 'Quarter Back':
            df['actual_pts'] = df.apply(lambda row: (row.YDS *0.04) + (row.TD *4) + (row.INT *-2) + (row.ydr_r *0.1) + (row.FL *-2)
                                + (row.td_r *6), axis=1)
        if format_func(option) == 'Running Back':
            df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.y_re *0.1) + (row.REC *0.5) 
                                + (row.TD *6) + (row.td_re *6) + (row.FL *-2), axis=1)
        if format_func(option) == 'Tight End':
            df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.TD *6) + (row.REC *0.5) 
                                + (row.FL *-2), axis=1)
        if format_func(option) == 'Wide Receiver':
            df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.TD *6) + (row.REC *0.5) 
                                + (row.ydr_rush *0.1) + (row.FL *-2), axis=1)
        return df

    df_pts = calc_actual_pts(option, data3)
    meta = df2[['Player', 'week', 'date']]
    df_fin = meta.merge(df_pts, how='right', on='date')
    df_fin = df_fin.reset_index(drop=True)
    p = df_fin.iloc[-1]['actual_pts'].round(1)
    w = df_fin.iloc[-1]['week']
    st.write(f'Forecasted Data:')
    st.write(f'**Your player is forecasted to score: {p} points during week {w}!** :sunglasses:')
    st.write(df_fin)
    st.line_chart(df_fin, y=['actual_pts'], x='date')


image1 = Image.open('2023/ff1.jpg')
st.image(image1)
