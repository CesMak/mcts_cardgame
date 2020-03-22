import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    result = []
    games  = []
    mean_reward = []
    mean_inv_moves = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            result.append(row)
    for rows in result:
        games.append(int(rows[0].split(" ")[1]))
        mean_reward.append(float(rows[1].split(" ")[2]))
        mean_inv_moves.append(float(rows[2].split(" ")[2]))
    return [games, mean_reward, mean_inv_moves]

def plot(array, out_path):
    '[games, mean_reward, mean_inv_moves]'
    games, mean_reward, mean_inv_moves = array
    min_moves_idx = mean_inv_moves.index(min(mean_inv_moves))
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig.suptitle('Invalid moves per game')
    ax1.plot(games, mean_inv_moves, label="min="+str(min(mean_inv_moves))+" at Game "+str(games[min_moves_idx])+" mean rw:"+str(mean_reward[min_moves_idx]))
    ax2.plot(games, mean_reward, label="max reward"+str(max(mean_reward)))
    fig.legend()
    plt.show()

    # 21*15 = 315
    # plt.xlabel("Games")
    # plt.ylabel("Invalid moves per game")
    # plt.title("Performance")
    # plt.plot()
    #
    # fig.savefig(out_path+"result.png")
    # plt.show()

if __name__ == '__main__':
    rewards = read_file("hallo.txt")
    plot(rewards, "ai")
