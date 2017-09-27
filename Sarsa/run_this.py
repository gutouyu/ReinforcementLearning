
# encoding=utf-8

from maze_env import Maze
from RL_brain import SarsaTable


def update():

    # 有100条命来让Agent学习，如果还没有学会，game over
    for episode in xrange(100):
        # Init observation/state
        observation = env.reset()

        # Sarsa根据observation选取一个action
        action = RL.choose_action(str(observation))

        while True:
            # Fresh env
            env.render()

            # Sarsa执行action,得到下一个observation observation_
            observation_, reward, done = env.step(action)

            # 执行了这一步的action之后，还要再选出下一步要执行的action，才能用于上一步的学习
            action_ = RL.choose_action(str(observation_))

            # 学习
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 更新observation action
            observation = observation_
            action = action_

            # 死掉了（掉进黑块）就重新来
            if done:
                break


    # End of game
    print('Game Over')
    env.destory()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
