for num in range(10, 200):
    try:
        file_path1 = "2023/weekly data/WR/data_temp_backup"
        file_path = "2023/weekly data/WR/"
        df = pd.read_csv(fr'{file_path1}/combined_clean_wr.csv', sep=',', encoding='utf-8',
            low_memory=False, index_col=0).drop(columns=['week', 'team_abbr', 'opp_team'])
        sel_date = df.loc[(df['date'] == '2023-12-07')]
        sel_date["date"] = pd.to_datetime(sel_date["date"], format='%Y-%m-%d', errors='coerce')
        sel_date = sel_date.set_index('date')
        df_1s = df.set_index('date', 'Player')
        #sel_date.dtypes
        Player = df['Player'].unique()
        Player = sorted(Player)
        Player = Player[num]
        print(Player)
        data = df[df['Player'].isin([Player])]
        data = data.sort_values(by=['date'])
        data = data.fillna(0)
        ch_data = ['YDS', 'TD', 'REC', 'ydr_rush', 'FL']
        def cum_sum_function(ch_data, data):
            x=0
            for i in ch_data:
                data[ch_data[x] + '_csum'] = data.groupby(['Player'])[ch_data[x]].cumsum()
                x=x+1
            return data
        find_cs = cum_sum_function(ch_data, data)
        find_cs = find_cs.sort_values(by=['date'])
        cs_data = ('YDS_csum', 'TD_csum', 'REC_csum', 'ydr_rush_csum', 'FL_csum')
        data1 = find_cs[['date', 'Player', *cs_data]]
        data1 = data1.set_index('date', 'Player')
        train = data1.iloc[:13] 
        test = data1.iloc[13:]
        final_df = pd.DataFrame(index = test.index, columns=[test.columns])
        x=0
        for i in cs_data:
            model = auto_arima(train[cs_data[x]], trace=True, error_action='ignore', suppress_warnings=True)
            mod_fit = model.fit(train[cs_data[x]])   
            forecast = mod_fit.predict(n_periods=len(test[cs_data[x]]))
            forecast_df = pd.DataFrame(forecast,index = test.index,columns=[cs_data[x]])
            final_df = final_df.join(forecast_df, how='left')
            x=x+1

        final_df1 = final_df.dropna(axis=1, how='all')
        final_df1['Player'] = Player
        final_df1 = final_df1.set_index(['Player'], append=True)
        data2 = train.merge(final_df1, how='outer', on=['date','Player', *cs_data])
        data2 = data2.set_index(['Player'], append=True)
        data3 = data2.diff()
        data3[[*cs_data]] = data3[[*cs_data]].round(2)
        data3.columns = data3.columns.str.rstrip('_csum')
        data3 = data3.dropna()
        data3 = data3.iloc[[-1]]
        def calc_actual_pts(df):
            df['actual_pts'] = df.apply(lambda row: (row.YDS *0.1) + (row.TD *6) + (row.REC *0.5) 
                                    + (row.ydr_rush *0.1) + (row.FL *-2), axis=1)
            return df

        df_pts = calc_actual_pts(data3)
        df_pts['actual_pts'] = df_pts['actual_pts'].round(1)
        df_pts['Player'] = Player
        df_pts = df_pts[['Player', *ch_data, 'actual_pts']]
        final_forecast_df12 = pd.DataFrame(index = df_1s.index, columns = df_1s.columns)
        final_forecast_df = final_forecast_df12.append(df_pts)
        final_forecast_df1 = final_forecast_df.dropna()
        final_forecast_df1.to_csv(f'{file_path}\{Player}_forecasted_points.csv')
        
    except:
        print("error")
