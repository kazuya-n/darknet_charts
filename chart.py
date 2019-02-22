import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sd_features = ["Region", "IOU", "Class", "Obj", "NoObj", ".5R", ".75R", "count"]
bc_features = ["Batch", "total_loss", "avg_loss", "l_rate", "batch_time", "total_image"]

# Define SubDivision output chunk lines
sd_chunks = 96


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file_path", help="path to redirection of darknet", type=str)
    args = parser.parse_args()
    
    path = args.log_file_path

    with open(path) as f:
        rlines = f.readlines()
    sd_outputs = [line.split(" ") for line in rlines if 'Region' in line]
    bc_outputs = [line.split(" ") for line in rlines if 'avg loss' in line]

    print("Trail Subdivisions")

    # Subdivision Outputs
    sd = pd.DataFrame(sd_outputs)
    sd = sd[[1, 4, 6, 8, 11, 13, 15, 18]]
    sd = sd.apply(lambda x: x.str.replace('nan', '9999999'))
    sd = sd.apply(lambda x: x.str.replace(',', '')).astype(np.float32)
    sd = sd.replace(-9999999, np.NaN)
    sd.columns = sd_features
    sd = sd.astype(float).reset_index()
    # sns_plot = sns.lineplot(x=sd.index/sd_chunks, y="IOU", data=sd)
    # sns_plot.figure.savefig("output.png")

    print("Trail Batchs")

    # Batch Outputs
    bc = pd.DataFrame(bc_outputs)
    bc = bc[[1, 2, 3, 6, 8, 10]]
    bc = bc.apply(lambda x: x.str.replace(':', ''))
    bc = bc.apply(lambda x: x.str.replace(',', '')).astype(np.float32)
    bc.columns = bc_features
    # sns_plot2 = sns.lineplot(x="Batch", y="avg_loss", data=bc)
    # sns_plot2.figure.savefig("output2.png")

    print("Draw plots")

    fig = plt.figure(figsize=(5,5),dpi=200)
    
    ax1 = fig.add_subplot(111)
    ax1.plot(sd.index/sd_chunks, sd.IOU.values)
    ax1.set_ylabel('IOU')

    ax2 = ax1.twinx()
    ax2.plot(bc.Batch.values, bc.avg_loss.values, 'r-')
    ax2.set_ylabel('avg_loss', color='r')
    for tl in ax2.get_yticklabels():
            tl.set_color('r')
    plt.tight_layout()
    plt.savefig('figure.png')

if __name__ == "__main__":
    main()
