from tpg.program import Program
from tpg.action_object import ActionObject
import numpy as np
from tpg.utils import flip
import random

"""
A team has multiple learners, each learner has a program which is executed to
produce the bid value for this learner's action.
"""
class Learner:

    idCount = 0 # unique learner id

    """
    Create a new learner, either copied from the original or from a program or
    action. Either requires a learner, or a program/action pair.
    """
    def __init__(self, learner=None, program=None, actionObj=None, numRegisters=8):
        if learner is not None:
            self.program = Program(instructions=learner.program.instructions)
            self.actionObj = ActionObject(learner.actionObj)
            self.registers = np.zeros(len(learner.registers), dtype=float)
        elif program is not None and actionObj is not None:
            self.program = program
            self.actionObj = actionObj
            self.registers = np.zeros(numRegisters, dtype=float)

        self.states = []

        self.numTeamsReferencing = 0 # amount of teams with references to this

        self.id = Learner.idCount
        Learner.idCount += 1

    """
    Get the bid value, highest gets its action selected.
    """
    def bid(self, state):
        Program.execute(state, self.registers,
                        self.program.instructions[:,0], self.program.instructions[:,1],
                        self.program.instructions[:,2], self.program.instructions[:,3])

        return self.registers[0]

    """
    Returns the action of this learner, either atomic, or requests the action
    from the action team.
    """
    def getAction(self, state, visited):
        return self.actionObj.getAction(state, visited)


    """
    Returns true if the action is atomic, otherwise the action is a team.
    """
    def isActionAtomic(self):
        return self.actionObj.isAtomic()

    """
    Mutates either the program or the action or both.
    """
    def mutate(self, config, actionCodes, actionLengths,
               teams, parentTeam, progMutFlag,
               inputs=None, outputs=None):
        changed = False
        while not changed:
            # mutate the program
            if flip(config.pMutProg):
                changed = True
                self.program.mutate(config, inputs=inputs, outputs=outputs)

            # mutate the action
            if flip(config.pMutAct):
                changed = True
                self.actionObj.mutate(config, inputs, outputs,
                                      parentTeam, actionCodes, actionLengths,
                                      teams,progMutFlag)

    """
    Saves visited states for mutation uniqueness purposes.
    """
    def saveState(self, state, numStates=50):
        self.states.append(state)
        self.states = self.states[-numStates:]
