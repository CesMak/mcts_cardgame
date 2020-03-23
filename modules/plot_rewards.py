import csv
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    result         = []
    games          = []
    mean_reward    = []
    mean_inv_moves = []
    won_games      = [[], [], [], []]
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            result.append(row)
    for row in result:
        games.append(int(row[1]))
        mean_reward.append(float(row[3]))
        mean_inv_moves.append(float(row[5]))
        splitted = row[7][1:-1].split(". ")
        for i in range(4):
            won_games[i].append(float(splitted[i]))
    return [games, mean_reward, mean_inv_moves, won_games]

def plot(array, out_path):
    '[games, mean_reward, mean_inv_moves]'
    games, mean_reward, mean_inv_moves, won_games = array
    min_moves_idx = mean_inv_moves.index(min(mean_inv_moves))
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    fig.suptitle('Invalid moves per game')
    ax1.plot(games, mean_inv_moves, label="min="+str(min(mean_inv_moves))+" at Game "+str(games[min_moves_idx])+" mean rw:"+str(mean_reward[min_moves_idx]))
    ax2.plot(games, mean_reward, label="max reward"+str(max(mean_reward)))
    ai_index = 1
    for i in range(len(won_games)):
        if i == ai_index:
            ax3.plot(games, won_games[i], label = 'ai player     %s' %i)
        else:
            ax3.plot(games, won_games[i], label = 'random player %s'%i)
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
    rewards = read_file("logging.txt")
    plot(rewards, "ai")
