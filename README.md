# Differentiable Inter Agent Learning to solve the switch riddle

## Paper
https://arxiv.org/pdf/1605.06676.pdf

## DIAL Implementation
This is an implementation of DIAL in python, for a popoular riddle. 
The riddle (as stated in the above paper) is:

*“One hundred prisoners have been newly
ushered into prison. The warden tells them that starting
tomorrow, each of them will be placed in an isolated cell,
unable to communicate amongst each other. Each day,
the warden will choose one of the prisoners uniformly
at random with replacement, and place him in a central
interrogation room containing only a light bulb with a
toggle switch. The prisoner will be able to observe the
current state of the light bulb. If he wishes, he can toggle
the light bulb. He also has the option of announcing that he believes all prisoners have visited the
interrogation room at some point in time. If this announcement is true, then all prisoners are set free,
but if it is false, all prisoners are executed. The warden leaves and the prisoners huddle together to
discuss their fate. Can they agree on a protocol that will guarantee their freedom?*

The above riddle has been solved using DIAL, where agents learning via Q-Learning are forced to learn meaningful messaging protocols, in order to complete a task that requires coordination. This protocol is learned by backpropagating gradients through the message outputs of the neural nets. 
**This implementation does not use parameter sharing.** That is, there are n different agents with their own parameters. There are no shared weights. The given model uses GRUs and dense layers with the relu activation to produce reliable Q-value outputs and messages.

## Installation
1) Clone this repo: git clone https://github.com/sharan-dce/DIAL-switch.git
2) cd DIAL-switch
3) Create a virtual environment: virtualenv env
4) source ./env/bin/activate
5) Install requirements in this environment: sudo pip3 install -r requirements.txt
6) Simply run dial.py: python3 dial.py
7) Deactivate environment when done: deactivate
The above script by default runs for 4 agents - you can change that (in dial.py). The paper also shows results for 3 and 4 agents. This is because the policy space increases exponentially with the number of agents. 
As given in the paper, this reaches optimal policy in about 5k plays, but train runs are carried out for about 4n steps rather than 4n - 6 as in the paper. The paper is super interesting! Do give it a read!