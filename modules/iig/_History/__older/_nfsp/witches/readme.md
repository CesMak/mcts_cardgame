NFSP:
* Test [this](https://github.com/younggyoseo/pytorch-nfsp) implementation for 2-player witches
* RL/SL Loss  RL=Reinforcement Learning, SL = Supervised Learning
* Siehe LaserTag wie werden die Rewards vergeben?!
  * Agents get a unit reward for touching other agent with laser beam
  * Reward is np.array([0, 0]) or np.array([1, 0]) or np.array([0, 1])
  * Es ist ein Nullsummenspiel (daher muss -1 für verloren und +1 für gewonnen als reward gegeben werden??!)
  * Actions of player: FORWARD = 0,  BACKWARD = 1, TURN_RIGHT = 5, BEAM = 8, STAY = 9
  * Lass LaserTag trainieren und schau dir output an: Reward waechst stetig


* Teste test --evaluate function in main
* Welches Model wird genutzt

* Teste state by witches....
* State klappt...


* Teste andere Rewards!
* done erst wenn 1 am ende

```
(mcts_env) markus@markus-pc:~/Documents/06_Software_Projects/mcts/modules/iig/_nfsp/lasertag$ python main.py
                        Options
                        seed: 1122
                        batch_size: 32
                        no_cuda: False
                        max_frames: 1400000
                        buffer_size: 100000
                        update_target: 1000
                        train_freq: 1
                        gamma: 0.99
                        eta: 0.1
                        rl_start: 10000
                        sl_start: 1000
                        dueling: False
                        multi_step: 1
                        env: LaserTag-small2-v0
                        negative: False
                        load_model: None
                        save_model: model
                        evaluate: False
                        render: False
                        evaluation_interval: 10000
                        lr: 0.0001
                        max_tag_interval: 1000
                        eps_start: 1.0
                        eps_final: 0.01
                        eps_decay: 30000
                        cuda: False
                        device: cpu
/home/markus/Documents/06_Software_Projects/mcts/mcts_env/lib/python3.6/site-packages/pycolab/ascii_art.py:318: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.
art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in art)
Frame: 10000    FPS: 132.25 Avg. Tagging Interval Length: 156.28
Player 1 Avg. Reward: 5.00 Avg. RL/SL Loss: 0.00/0.00
Player 2 Avg. Reward: 6.40 Avg. RL/SL Loss: 0.00/0.00
Frame: 20000    FPS: 19.88 Avg. Tagging Interval Length: 103.42
Player 1 Avg. Reward: 9.50 Avg. RL/SL Loss: 0.00/2.21
Player 2 Avg. Reward: 9.10 Avg. RL/SL Loss: 0.00/2.22
Frame: 30000    FPS: 20.57 Avg. Tagging Interval Length: 71.38
Player 1 Avg. Reward: 13.20 Avg. RL/SL Loss: 0.01/2.18
Player 2 Avg. Reward: 14.10 Avg. RL/SL Loss: 0.01/2.15
Frame: 40000    FPS: 20.57 Avg. Tagging Interval Length: 56.81
Player 1 Avg. Reward: 17.10 Avg. RL/SL Loss: 0.01/2.10
Player 2 Avg. Reward: 16.50 Avg. RL/SL Loss: 0.01/2.06
Frame: 50000    FPS: 20.49 Avg. Tagging Interval Length: 58.34
Player 1 Avg. Reward: 18.40 Avg. RL/SL Loss: 0.02/2.01
Player 2 Avg. Reward: 14.60 Avg. RL/SL Loss: 0.02/1.99
Frame: 60000    FPS: 20.45 Avg. Tagging Interval Length: 52.85
Player 1 Avg. Reward: 21.20 Avg. RL/SL Loss: 0.02/1.94
Player 2 Avg. Reward: 16.60 Avg. RL/SL Loss: 0.02/1.90
Frame: 70000    FPS: 20.35 Avg. Tagging Interval Length: 46.97
Player 1 Avg. Reward: 29.10 Avg. RL/SL Loss: 0.03/1.86
Player 2 Avg. Reward: 13.50 Avg. RL/SL Loss: 0.02/1.82
Frame: 80000    FPS: 20.19 Avg. Tagging Interval Length: 48.76
Player 1 Avg. Reward: 25.10 Avg. RL/SL Loss: 0.03/1.80
Player 2 Avg. Reward: 14.50 Avg. RL/SL Loss: 0.03/1.76
Frame: 90000    FPS: 19.97 Avg. Tagging Interval Length: 40.69
Player 1 Avg. Reward: 35.00 Avg. RL/SL Loss: 0.04/1.78
Player 2 Avg. Reward: 13.50 Avg. RL/SL Loss: 0.03/1.72
Frame: 100000   FPS: 19.91 Avg. Tagging Interval Length: 35.48
Player 1 Avg. Reward: 39.10 Avg. RL/SL Loss: 0.04/1.78
Player 2 Avg. Reward: 16.70 Avg. RL/SL Loss: 0.03/1.68
Frame: 110000   FPS: 19.93 Avg. Tagging Interval Length: 31.29
Player 1 Avg. Reward: 48.80 Avg. RL/SL Loss: 0.05/1.76
Player 2 Avg. Reward: 14.70 Avg. RL/SL Loss: 0.04/1.63
```



https://github.com/DoktorDaveJoos/Neural-Ficititious-Self-Play-in-Imperfect-Information-Games
above link geht nicht mehr!!!


https://towardsdatascience.com/neural-fictitious-self-play-800612b4a53f

neural-fictitious-self-play

https://www.groundai.com/project/deep-reinforcement-learning/1

https://drive.google.com/drive/folders/1ru_TDEmMKXjiv4ie3Zl4nYPsHngkZExr

pytorch promising implementation of nfsp:
https://github.com/EricSteinberger/Neural-Fictitous-Self-Play

or see this pytorch implementation:
https://github.com/thomasj02/nfsp-pytorch/blob/master/KuhnPoker/NFSP/Dqn.py

accoring to following nfsp should work for multiple player as well!
https://www.researchgate.net/publication/323173582_Neural_Fictitious_Self-Play_in_Imperfect_Information_Games_with_Many_Players

see: exmamples from the book
https://books.google.de/books?id=5ztMDwAAQBAJ&pg=PA61&lpg=PA61&dq=Neural+Fictitious+Self-Play+in+Imperfect+Information+Games+with+Many+Players+Keigo+Kawamura&source=bl&ots=EAwYfCygXu&sig=ACfU3U3fxAqyRdxBazAU7pX979S7OFRsNg&hl=de&sa=X&ved=2ahUKEwiV1pXk_JfpAhUnx4UKHdBLCrAQ6AEwA3oECAoQAQ#v=onepage&q=Neural%20Fictitious%20Self-Play%20in%20Imperfect%20Information%20Games%20with%20Many%20Players%20Keigo%20Kawamura&f=false


noch besser als nfsp: using monte-carlo:
FSP performs poorly in games with large search spaces and depths. Another shortfall is that in NFSP, the optimal response depends on deep Q-learning calculations, which require a long time until convergence.   
see: ANFSP
