import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    result = []
    rewards = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            result.append(row)
    for rows in result:
        rewards.append(float(rows[2]))  # rewards
        rewards.append(float(rows[3].split(":")[1]))
    return rewards

def plot(array, out_path):
    'input: [[ply1, play2, play3, pay4], ...]'
    x = np.linspace(0, len(array), num=len(array))
    y = array
    print("Maximum:", max(y))
    print("Minimum:", min(y))
    plt.xlabel("Time")
    plt.ylabel("Number of games won")
    plt.title("Performance")
    plt.plot(x,y, label="ai")
    plt.legend()
    plt.savefig(out_path+"result.png")
    plt.show()

if __name__ == '__main__':
    rewards = read_file("hallo.txt")
    plot(rewards, "ai")
