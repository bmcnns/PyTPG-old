"""
A Team.
"""
class Team:

    def __init__(self, birthGen = 0):
        self.birthGen = birthGen
        self.learners = []

    def addLearner(self, learner):
        if learner not in self.learners:
            learner.teamRefCount += 1
            self.learners.append(learner)
