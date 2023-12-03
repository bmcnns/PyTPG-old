from tpg.program import Program
import numpy as np
import random
from tpg.utils import flip

"""
Action  Object has a program to produce a value for the action, program doesn't
run if just a discrete action code.
"""
class ActionObject:

    def __init__(self, actionObj=None, program=None, actionCode=None,
            actionLength=0, teamAction=None, maxProgramLength=128, nRegisters=8):
        if actionObj is not None:
            # clone the other action object
            self.actionCode = actionObj.actionCode
            self.teamAction = actionObj.teamAction
        else:
            # no cloning
            self.actionCode = actionCode
            self.actionLength = actionLength
            self.teamAction = teamAction

        # increase references to team
        if self.teamAction is not None:
            self.teamAction.numLearnersReferencing += 1

    """
    Returns the action code, and if applicable corresponding real action(s).
    """
    def getAction(self, state, visited):
        if self.teamAction is not None:
            # action from team
            return self.teamAction.act(state, visited)
        else:
            # atomic action
            return None


    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isAtomic(self):
        return self.teamAction is None

    """
    Either swap the action, or modify the program, depending on a flag.
    """
    def mutate(self, config, pActAtom, inputs,
               outputs, parentTeam, actionCodes,
               actionLengths, teams, progMutFlag):
        # dereference if old action is team
        if self.teamAction is not None:
            self.teamAction.numLearnersReferencing -= 1
            self.teamAction = None

        # mutate action
        if flip(pActAtom):
            # atomic
            self.actionCode = random.choice(actionCodes)
        else:
            # team action
            self.teamAction = random.choice([t for t in teams
                    if t is not self.teamAction and t is not parentTeam])
            self.teamAction.numLearnersReferencing += 1
