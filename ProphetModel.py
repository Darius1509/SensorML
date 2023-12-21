import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv('SensorMLDataset_small.csv')

for column in df.columns[2:]:
    variable_name = column
    df_prophet = df[['Timestamp', variable_name]].rename(columns={'Timestamp': 'ds', variable_name: 'y'})

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=48, freq='H')
    forecast = model.predict(future)

    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Plot predicted vs. actual values
    fig1 = model.plot(forecast, xlabel='Timestamp', ylabel=variable_name, figsize=(10, 4))
    fig1.canvas.manager.set_window_title(f'Prediction for {variable_name}')
    plt.show()

    # fig2 = model.plot_components(forecast, figsize=(10, 4))
    # fig2.canvas.manager.set_window_title(f'Prediction Components for {variable_name}')
    # plt.show()
