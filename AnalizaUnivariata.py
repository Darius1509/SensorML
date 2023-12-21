import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("SensorMLDataset_small.csv")

columns = df.columns

df['Timestamp'] = df['Timestamp'].apply(lambda x: x.split()[0])

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

pdf_filename = 'MeanPlots.pdf'
with PdfPages(pdf_filename) as pdf:
    for column in columns[1:]:
        df2 = df[['Timestamp', column]]

        df2 = df2.groupby('Timestamp').mean()

        plt.figure(figsize=(20, 10))
        sns.heatmap(df2, cmap='coolwarm')
        plt.title(f"mean {column}")

        pdf.savefig()
        plt.close()

pdf_filename = 'MedianPlots.pdf'
with PdfPages(pdf_filename) as pdf:
    for column in columns[1:]:
        df2 = df[['Timestamp', column]]

        df2 = df2.groupby('Timestamp').median()

        plt.figure(figsize=(20, 10))
        sns.heatmap(df2, cmap='coolwarm')
        plt.title(f"median {column}")

        pdf.savefig()
        plt.close()


pdf_filename = 'BoxPlots.pdf'
with PdfPages(pdf_filename) as pdf:

    for column in columns[1:]:
        df2 = df[['Timestamp', column]]

        plt.figure(figsize=(20, 10))
        sns.boxplot(x=df2[column])
        plt.title(f"boxplot {column}")

        pdf.savefig()
        plt.close()
