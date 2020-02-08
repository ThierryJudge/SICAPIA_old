from matplotlib import pyplot as plt
import pandas as pd
import ntpath
import os

files = ['project/mnist/ConfidenceSamplingStrategy/ConfidenceSamplingStrategy.csv',
         'project/mnist/MarginSamplingStrategy/MarginSamplingStrategy.csv',
         'project/mnist/RandomStrategy/RandomStrategy.csv']

"""
Examples usage 

compare_history.py" --paths path/to/history1.csv path/to/history2.csv path/to/history3.csv
"""
if __name__ == '__main__':
    import argparse

    argparse = argparse.ArgumentParser()
    argparse.add_argument("--paths", type=str, nargs='*', help="Path to history files to compare")
    args = argparse.parse_args()

    dataframes = []
    for _, file in enumerate(args.paths):
        name = os.path.splitext(ntpath.basename(file))[0]
        df = pd.read_csv(file)
        for i, k in enumerate(df.keys()):
            if k != 'num_samples':
                plt.figure(i)
                plt.title(k)
                plt.plot(df['num_samples'], df[k], label=name)
                plt.legend()


    plt.show()






