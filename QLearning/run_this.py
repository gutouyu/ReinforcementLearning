
# encoding=utf-8

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in xrange(100):
        # Init observation/state
        observation = env.reset()

        while True:
            # Fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # Update state
            observation = observation_

            # Break while loop when end of this episode
            if done:
                break

    # End of game
    print('Game Over')
    env.destory()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
