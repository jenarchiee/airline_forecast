import pandas as pd
import numpy as np

def airlineForecast(trainingDataFileName,validationDataFileName):
    train_data = pd.read_csv(trainingDataFileName, sep=',', header=0)
    valid_data = pd.read_csv(validationDataFileName, sep=',', header=0)

    # Convert the departure_date and booking_date:
    train_data['departure_date'] = pd.to_datetime(train_data['departure_date'])
    train_data['booking_date'] = pd.to_datetime(train_data['booking_date'])
    valid_data['departure_date'] = pd.to_datetime(valid_data['departure_date'])
    valid_data['booking_date'] = pd.to_datetime(valid_data['booking_date'])

    # Calculate days prior
    train_data['days_prior'] = (train_data['departure_date'] - train_data['booking_date']).dt.days
    valid_data['days_prior'] = (valid_data['departure_date'] - valid_data['booking_date']).dt.days

    # Make a week_day column for each data file
    train_data['week_day'] = train_data['departure_date'].dt.weekday_name
    valid_data['week_day'] = valid_data['departure_date'].dt.weekday_name

    # Calculate benchmark
    valid_data['benchmark'] = abs(valid_data['final_demand']-valid_data['naive_forecast'])
    benchmark = valid_data['benchmark'].sum()

    # Calculate final demand (maximum cumulative bookings at given departure date)
    train_data['final_demand'] = train_data['cum_bookings'].groupby(train_data['departure_date']).transform(max)


    # Flat model: Forecast = On-the-book valid_data + (Final Demand Training_data - Cumulative Booking Valid_data)

    avg_final_demand = train_data['final_demand'].groupby(train_data['week_day']).median()
    avg_final_demand = pd.DataFrame({'avg_final_demand': avg_final_demand})
    df_merge1 = valid_data.merge(avg_final_demand, left_on=['week_day'], right_on=['week_day'])
    df_merge1['forecast_demand'] = df_merge1['cum_bookings'] + df_merge1['avg_final_demand'] - df_merge1['cum_bookings']
    df_merge1.loc[df_merge1['cum_bookings'] == df_merge1['final_demand'], 'forecast_demand'] = np.nan
    df_forecast1 = df_merge1[['departure_date', 'booking_date', 'forecast_demand']]

    # Calculate MASE
    df_merge1['error'] = abs(df_merge1['final_demand'] - df_merge1['forecast_demand'])
    error1 = df_merge1['error'].sum()
    MASE1 = round((error1 / benchmark) * 100,2)


    # Additive model: Forecast = On-the-book (valid_data) + Remaining Demand
    # Remaining Demand = Final Demand (Training data) - Cumulative Booking (Training data)

    train_data['remain_demand'] = train_data['final_demand'] - train_data['cum_bookings']
    train_data2 = train_data[['week_day', 'days_prior', 'remain_demand']].groupby(['week_day', 'days_prior']).median()
    df_merge2 = valid_data.merge(train_data2, left_on=['week_day', 'days_prior'], right_on=['week_day', 'days_prior'])
    df_merge2['forecast_demand'] = df_merge2['cum_bookings'] + df_merge2['remain_demand']
    df_merge2.loc[df_merge2['cum_bookings'] == df_merge2['final_demand'], 'forecast_demand'] = np.nan
    df_forecast2 = df_merge2[['departure_date', 'booking_date', 'forecast_demand']]

    # Calculate MASE
    df_merge2['error'] = abs(df_merge2['final_demand'] - df_merge2['forecast_demand'])
    error3 = df_merge2['error'].sum()
    MASE2 = round((error3 / benchmark) * 100,2)


    # Multiplicative model: Forecast = on-the-book (Valid_data)_/ booking ratio (Training_data)
    # Booking ratio = on the-book (Training data) / final demand (Training data set)

    train_data['booking_ratio'] = train_data['cum_bookings'] / train_data['final_demand']
    train_data3 = train_data[['week_day', 'days_prior', 'booking_ratio']].groupby(['days_prior']).mean()
    df_merge3 = valid_data.merge(train_data3, left_on=['days_prior'], right_on=['days_prior'])
    df_merge3 = df_merge3.sort_values(['departure_date', 'booking_date'])
    df_merge3['forecast_demand'] = df_merge3['cum_bookings'] / df_merge3['booking_ratio']
    df_merge3.loc[df_merge3['cum_bookings'] == df_merge3['final_demand'], 'forecast_demand'] = np.nan
    df_forecast3 = df_merge3[['departure_date', 'booking_date', 'forecast_demand']]

    # Calculate MASE
    df_merge3['error'] = abs(df_merge3['final_demand'] - df_merge3['forecast_demand'])
    error3 = df_merge3['error'].sum()
    MASE3 = round((error3 / benchmark) * 100,2)

    return {'forecast1': df_forecast1, 'MASE1': MASE1, 'forecast2': df_forecast2, 'MASE2': MASE2, 'forecast3': df_forecast3, 'MASE3': MASE3}

def main():
    file = airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData.csv')
    print("Forecast error of the Flat model is:",  file['MASE1'])
    print(file['forecast1'], '\n')
    print("Forecast error of the Additive model is:", file['MASE2'])
    print(file['forecast2'], '\n')
    print("Forecast error of the Multiplicative model is:", file['MASE3'])
    print(file['forecast3'])
main()

