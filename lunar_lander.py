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
    plt.title(f"Lunar Lander Team #{teamNum}, Generation #{generation+1}, Score: {score}")
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"lunar_lander_{index}.png"))

def detect_changes(predecessor, successor):
    return [1 if pred != succ else 0 for pred, succ in zip(predecessor, successor)]

def main():
    parser = argparse.ArgumentParser(description="Lunar lander benchmark using TPG")
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

    qLearner = QLearner(customConfig.memorySize, 4, 0.8, 0.001)
    epsilon = 0.15

    frame_index = 0

    env = gym.make('LunarLander-v2', render_mode='rgb_array')

    fig = plt.Figure()

    trainer = Trainer(actions=range(4), config=customConfig)

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
                    action = np.random.randint(4)
                else:
                    if np.random.random() < epsilon:
                        action = np.random.randint(4)
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

    rewardInfo = np.array(rewardStats)

    min_rewards = rewardInfo[:, 0]
    max_rewards = rewardInfo[:, 1]
    avg_rewards = rewardInfo[:, 2]

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 6))

    # Plot rewards over time
    episodes = range(1, len(min_rewards) + 1)
    axes.plot(episodes, min_rewards, label='Min Rewards', color='#1f78b4', alpha=0.8)
    axes.plot(episodes, max_rewards, label='Max Rewards', color='#33a02c', alpha=0.8)
    axes.plot(episodes, avg_rewards, label='Avg Rewards', color='#e31a1c', alpha=0.8)
    axes.set_title('Rewards over Time')
    axes.set_xlabel('Generation')
    axes.set_ylabel('Reward')
    axes.legend(loc='upper right')
    axes.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(args.outputDirectory, f"lunar_lander_results.png"))

if __name__ == "__main__":
    main()
