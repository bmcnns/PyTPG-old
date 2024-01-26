from tpg.utils import flip
from tpg.learner import Learner
from tpg.memory import Memory
import random

"""
The main building block of TPG. Each team has multiple learning which decide the
action to take in the graph.
"""
class Team:
    def __init__(self):
        self.learners = []
        self.outcomes = {} # scores at various tasks
        self.fitness = None
        self.numLearnersReferencing = 0 # number of learners that reference this
        self.memory = Memory(25)

    """
    Returns an action to use based on the current state.
    """
    def act(self, state, visited=set()):
        visited.add(self) # track visited teams

        topLearner = max([lrnr for lrnr in self.learners
                if lrnr.isActionAtomic() or lrnr.actionObj.teamAction not in visited],
            key=lambda lrnr: lrnr.bid(state, self.memory))

        self.memory.commit(topLearner.program.id)

    """
    Adds learner to the team and updates number of references to that program.
    """
    def addLearner(self, learner=None):
        program = learner.program
        # don't add duplicate program
        if any([lrnr.program == program for lrnr in self.learners]):
            return False

        self.learners.append(learner)
        learner.numTeamsReferencing += 1

        return True

    """
    Removes learner from the team and updates number of references to that program.
    """
    def removeLearner(self, learner):
        if learner in self.learners:
            learner.numTeamsReferencing -= 1
            self.learners.remove(learner)

    """
    Bulk removes learners from teams.
    """
    def removeLearners(self):
        for lrnr in list(self.learners):
            self.removeLearner(lrnr)


    def getPrograms(self):
        programs = []
        for learner in list(self.learners):
            programs.append(learner.program.id)
        return programs

    """
    Number of learners with atomic actions on this team.
    """
    def numAtomicActions(self):
        num = 0
        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                num += 1

        return num

    """
    Mutates the learner set of this team.
    """
    def mutate(self, config, allLearners, actionCodes,
               actionLengths, allTeams, progMutFlag,
               inputs=None, outputs=None):

        # delete some learners
        p = config.pDelLrn
        while flip(p) and len(self.learners) > 2: # must have >= 2 learners
            p *= config.pDelLrn # decrease next chance

            # choose non-atomic learners if only one atomic remaining
            learner = random.choice([l for l in self.learners
                                     if not l.isActionAtomic()
                                        or self.numAtomicActions() > 1])
            self.removeLearner(learner)

        # add some learners
        p = config.pAddLrn
        while flip(p):
            p *= config.pAddLrn # decrease next chance

            learner = random.choice([l for l in allLearners
                                     if l not in self.learners and
                                        l.actionObj.teamAction is not self])
            self.addLearner(learner)

        # give chance to mutate all learners
        oLearners = list(self.learners)

        for learner in oLearners:
            if flip(config.pMutLrn):
                if self.numAtomicActions() == 1 and learner.isActionAtomic():
                    pActAtom0 = 1 # action must be kept atomic if only one
                else:
                    pActAtom0 = config.pActAtom

                # must remove then re-add fresh mutated learner
                self.removeLearner(learner)
                
                newLearner = Learner(learner=learner)

                newLearner.mutate(
                            config, pActAtom0, actionCodes, actionLengths,
                            allTeams, self, progMutFlag,
                            inputs=None, outputs=None)

                self.addLearner(newLearner)
