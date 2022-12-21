import pandas as pd
import matplotlib.pyplot as plt
import sys, os

def vis_csv_key(path, key="train/total_loss"):
    save_path  = os.path.dirname(path)
    df = pd.read_csv(path,)
    try:
        df[key].plot()
    except KeyError:
        print(key , "not exists")
        exit()
    name = key.split("/")[-1]
    loss_save_path = os.path.join(save_path, f"{name}.jpg")
    plt.savefig(loss_save_path)

if __name__ == "__main__":
    vis_csv_key(sys.argv[1], sys.argv[2])
