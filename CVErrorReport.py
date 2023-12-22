import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

dataset = pd.read_csv("SensorMLDataset_small.csv")
selected_columns = ['pres', 'temp1', 'umid', 'temp2', 'lumina', 'temps1', 'temps2', 'V450', 'B500', 'G550', 'Y570', 'O600', 'R650']
selected_data = dataset[selected_columns]
selected_data['Timestamp'] = dataset['Timestamp']


def cross_validation(data, variable_name):
    df_prophet = data[['Timestamp', variable_name]].rename(columns={'Timestamp': 'ds', variable_name: 'y'})

    model = Prophet()
    model.fit(df_prophet)

    from prophet.diagnostics import cross_validation
    df_cv = cross_validation(model, initial='730 hours', period='24 hours', horizon='48 hours')
    print(f"df_cv for {variable_name}:\n{df_cv}")

    from prophet.diagnostics import performance_metrics
    df_p = performance_metrics(df_cv)

    print(f"df_p for {variable_name}:\n{df_p}")

    from prophet.plot import plot_cross_validation_metric
    metrics = ['mape', 'rmse', 'mse']

    fig, ax = plt.subplots(len(metrics), 1, figsize=(10, len(metrics) * 4))

    if df_cv is not None:
        for i, metric in enumerate(metrics):
            try:
                plot_cross_validation_metric(df_cv, metric=metric, ax=ax[i])
                ax[i].set_title(f'Cross Validation {metric} for {variable_name}')
            except Exception as e:
                print(e)

        plt.show()
        fig.savefig(f'./CrossValidation/CrossValidation-{variable_name}.png')


for column in selected_data.columns[1:]:
    cross_validation(selected_data, column)
