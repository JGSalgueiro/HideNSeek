# Cooperation in an Adversarial Multi-Agent Game

## Introduction

In the rapidly evolving landscape of Artificial Intelligence, the focus has shifted from individual decision-making agents to the collaboration of autonomous agents working together to achieve common goals. This project, conducted in 2022 as part of the Master's Degree program at Instituto Superior Tecnico, delves into the realm of cooperative multi-agent systems, exploring how independent agents can learn to collaborate effectively. The objective is to contribute to the understanding of cooperative behavior and learning mechanisms, particularly in the context of adversarial settings.

## Motivation

In real-world applications, cooperative AI systems are gaining increasing significance, ranging from fully-autonomous driving to the exploration of hazardous environments. Recognizing the potential impact of cooperative artificial intelligence on various domains, this project aims to unravel the dynamics of agent collaboration through the lens of a multi-agent game. The selection of a game environment provides a controlled and structured setting, allowing for precise analysis of strategies and cooperation methods.

## Project Selection

After careful consideration, the chosen game for this exploration is a simplified version of "hide and seek" played on a 2D squared board. This decision is inspired by the success of applying OpenAI to a similar game in the paper titled "Self and Other Modelling in Cooperative Resource Gathering with Multi-agent Reinforcement Learning." The game involves two teams, one hiding and the other seeking, competing against each other. The project introduces dynamic elements such as a dynamic board and team size to study scalability in cooperative behavior and learning. Additionally, obstacles on the board add a layer of complexity, enabling the analysis of how agents cope with a variable environment and collaborate in the face of adversity.

## Project Structure and Roles

The project is structured around the expertise of the team members. David focuses on creating the virtual environment, João is responsible for defining the rules for the environment and the agents, while Rafael applies learning algorithms to the agents.

## Implementation Approach

### Game Environment

The project starts with a simple NxM board, each agent assigned to either the seeking or hiding team. Agents can perceive all teammates but have limited visibility of agents from the opposing team. The game is turn-based, and agents make decisions to move to adjacent cells, with the seeking team aiming to find and eliminate the hiding team.

### Neural Models and Reinforcement Learning

To delve deeper into neural models, the project employs independent neural networks for each agent, trained through Q-Learning and Reinforcement Learning. The neural network outputs the agent's movement action based on neighboring cell information and teammate positions.

## Empirical Evaluation

Performance is measured by the number of turns taken by the seeking team to catch all members of the hiding team. The baseline is established through initial runs with random agent movements. The project's success is determined by the identification of clear cooperative behavior among team members, where actions are directed towards a common, mutually beneficial goal.

## Conclusion

This project not only explores the technical aspects of multi-agent reinforcement learning but also delves into the broader implications of cooperative behavior in artificial intelligence. By utilizing a game environment, the team aims to uncover insights that can be applied to real-world scenarios where collaborative decision-making among autonomous agents is paramount. As the project progresses, the team remains vigilant for unexpected behaviors, ready to adapt rules and strategies to enhance the learning process.

**Keywords:** Multi-Agent, Reinforcement Learning, Cooperation, Hide and Seek
