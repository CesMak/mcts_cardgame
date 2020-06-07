import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    finished_reward_ra = []
    finished_reward_ai = []
    corr_moves         = []
    games              = []
    batch_size         = []
    result = []
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            result.append(row)
    for row in result:
        games.append(int(row[1]))
        finished_reward_ai.append(float(row[3]))
        finished_reward_ra.append(float(row[5]))
        corr_moves.append(float(row[7]))
        batch_size.append(int(row[13].split("=")[1]))
    return [games, finished_reward_ai, finished_reward_ra, corr_moves, batch_size]

def plot(array, out_path, plot_percentage=False):
    ''
    games, finished_reward_ai, finished_reward_ra, corr_moves, batch_size= array
    ai_index = 1



    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True)
    fig.suptitle('Performance Witches - PPO LINEAR (no LSTM) mit Shuffeln, RANDOM, AI, RANDOM, RANDOM')
    ax1.plot(games, finished_reward_ai, label="ai_reward min="+str(max(finished_reward_ai)))
    ax1.plot(games, finished_reward_ra, label="mean random min="+str(max(finished_reward_ra)))
    ax1.legend()
    ax2.plot(games, corr_moves, label="correct moves max="+str(max(corr_moves)))
    ax2.legend()
    ax3.plot(games, batch_size, label="batch_size max="+str(max(batch_size)))
    ax3.legend()
    plt.show()


if __name__ == '__main__':
    rewards = read_file("logging16_40000_100_44_both2.csv")
    plot(rewards, "ai")
