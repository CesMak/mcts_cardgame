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

def plot(array, out_path, plot_percentage=False):
    '[games, mean_reward, mean_inv_moves]'
    games, mean_reward, mean_inv_moves, won_games = array
    ai_index = 1

    find_indx = 16.9
    try:
        idx = next(x for x, val in enumerate(mean_inv_moves) if val < find_indx)
        games = games[idx:-1]
        mean_reward = mean_reward[idx:-1]
        mean_inv_moves = mean_inv_moves[idx:-1]
        for i in range(len(won_games)):
            won_games[i] = won_games[i][idx:-1]
    except Exception as e:
        print(e)
    min_moves_idx = mean_inv_moves.index(min(mean_inv_moves))
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    # compute max. win percentage
    maximum = max(won_games[ai_index])
    tmp_idx = won_games[ai_index].index(maximum)
    percentage = 0.0
    try:
        percentage          = round(maximum/(won_games[0][tmp_idx]+won_games[1][tmp_idx]+won_games[2][tmp_idx]+won_games[3][tmp_idx])*100)
    except:
        print("percentage divsion \ 0")
    fig.suptitle('Performance Witches!')
    ax1.plot(games, mean_inv_moves, label="min="+str(min(mean_inv_moves))+" at Game "+str(games[min_moves_idx])+" mean rw:"+str(mean_reward[min_moves_idx]))
    ax1.legend()
    ax2.plot(games, mean_reward, label="max reward"+str(max(mean_reward)))
    ax2.legend()
    percentage_ai = []
    if plot_percentage:
        for i in range(len(won_games[0])):
            sum = won_games[0][i]+won_games[1][i]+won_games[2][i]+won_games[3][i]
            percentage_ai.append(won_games[ai_index][i]/sum*100)

        ax3.plot( games, percentage_ai, label = 'percentage of won games %s' %percentage+" % win rate "+str(maximum))
    else:
        for i in range(len(won_games)):
            if i == ai_index:
                ax3.plot(games, won_games[i], label = 'ai player     %s' %percentage+" % win rate "+str(maximum))
            else:
                ax3.plot(games, won_games[i], label = 'random player %s'%i)
    ax3.legend()
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
