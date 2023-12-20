import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_correlation(data, threshold=0.7):
    #correlation matrix
    correlation_matrix = data.corr()

    #lists for corellated values
    positive_correlation = []
    negative_correlation = []

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2 and col1 < col2:
                correlation_value = correlation_matrix.loc[col1, col2]
                if correlation_value > threshold:
                    positive_correlation.append((col1, col2, correlation_value))
                elif correlation_value < -threshold:
                    negative_correlation.append((col1, col2, correlation_value))

    #Correlation matrix as a heatmap
    plt.rcParams['figure.figsize'] = [10, 10]
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    print("Perechi de variabile cu corelație pozitiva:")
    for pair in positive_correlation:
        print(f"{pair[0]} & {pair[1]}: {pair[2]}")
    print("\nPerechi de variabile cu corelație negativa:")
    for pair in negative_correlation:
        print(f"{pair[0]} & {pair[1]}: {pair[2]}")


dataset = pd.read_csv("SensorMLDataset_small.csv")
selected_columns = ['pres', 'temp1', 'umid', 'temp2', 'lumina', 'temps1', 'temps2', 'V450', 'B500', 'G550', 'Y570', 'O600', 'R650']
selected_data = dataset[selected_columns]

analyze_correlation(selected_data)
