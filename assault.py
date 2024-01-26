import numpy as np
import gym

from tpg.trainer import Trainer
from tpg.agent import Agent
from tpg.qlearner import QLearner
from tpg.configurations import DefaultConfiguration

import numpy as np
import pandas as pd
from pprint import pprint

import matplotlib.pyplot as plt
import argparse
import os
import datetime

def update(output_folder, env, generation, teamNum, score, index):
    plt.clf()
    plt.imshow(env.render())
    plt.title(f"Assault Team #{teamNum}, Generation #{generation+1}, Score: {score}")
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"assault_{index}.png"))

def detect_changes(predecessor, successor):
    return [1 if pred != succ else 0 for pred, succ in zip(predecessor, successor)]

def main():
    parser = argparse.ArgumentParser(description="Assault benchmark using TPG")
    parser.add_argument("--teamPopSize", type=int, help="Size of the teams")
    parser.add_argument("--memorySize", type=int, help="Number of global memory registers used")
    parser.add_argument("--numGenerations", type=int, help="Number of generations to run")
    parser.add_argument("--outputDirectory", type=str, help="Output folder for the run video")


    args = parser.parse_args()

    os.makedirs(args.outputDirectory, exist_ok=True)

    customConfig = DefaultConfiguration()
    customConfig.teamPopSize = args.teamPopSize
    customConfig.memorySize = args.memorySize
    numGenerations = args.numGenerations

    qLearner = QLearner(customConfig.memorySize, 7, 0.8, 0.001)
    epsilon = 0.15

    frame_index = 0

    env = gym.make('Assault-v0', render_mode='rgb_array')

    fig = plt.Figure()

    trainer = Trainer(actions=range(7), config=customConfig)

    rewardStats = []

    for generation in range(numGenerations):
        rewards = []
        agents = trainer.getAgents()

        while True:
            teamNum = len(agents)
            agent = agents.pop()

            if agent is None:
                break # no more agents, proceed to next gen

            state = env.reset()[0]

            score = 0

            i = 0

            print(f"Gen #{generation}, Team #{teamNum}")

            isTerminal = False
            isTruncated = False

            while not isTerminal and not isTruncated and i < 500:
                i += 1

                agent.clearMemory()

                programs = agent.getPrograms()

                memory_before_update = agent.getMemory()

                # Update memory
                agent.act(state)

                memory_after_update = agent.getMemory()

                if generation < 2:
                    action = np.random.randint(7)
                else:
                    if np.random.random() < epsilon:
                        action = np.random.randint(7)
                    else:
                        action = qLearner.predict(memory_after_update).argmax()

                update(args.outputDirectory, env, generation, teamNum, score, frame_index)
                frame_index += 1

                state, reward, isTerminated, isTruncated, _ = env.step(action)

                score += reward

                if generation >= 2:
                    """
                    print("Previous state")
                    print(previousState)
                    print("Next state")
                    print(nextState)
                    print(f"Action selected: {action}, Reward: {reward}")
                    """
                    qLearner.train(memory_before_update, memory_after_update, reward, action)

                if isTerminated or isTruncated:
                    break

            if i == 500:
                print("Ran out of turns... giving up")


            agent.reward(score)

            rewards.append(score)

            print(f"Finished after {i} steps with cumulative reward {score}...")

            if len(agents) == 0:
                break

        rewardStats.append((min(rewards), max(rewards), sum(rewards)/len(rewards)))
        trainer.evolve()

    print("Finished run")

if __name__ == "__main__":
    main()
