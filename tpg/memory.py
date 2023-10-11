import numpy as np
import logging
import random


class Memory:
    _instance = None

    def __new__(cls, numRegisters):
        if cls._instance is None:
            cls._instance = super(Memory, cls).__new__(cls)
            cls._instance.initialize(numRegisters)
        return cls._instance

    def initialize(self, numRegisters):
        self.numRegisters = numRegisters
        self.registers = np.zeros(numRegisters)

    def read(self, index):
        if index > self.numRegisters:
            return self.registers[index % self.numRegisters]
        elif index < 0:
            raise RuntimeError("Memory read failed. Provided index is a negative value.")
            return
        else:
            return self.registers[index]

    def write(self, value):
        # randint has inclusive bounds a <= x <= b
        index = random.randint(0, self.numRegisters - 1)
        self.registers[index] = value


def get_memory():
    if Memory._instance is None:
        raise RuntimeError("Memory instance has not been initialized.")
    return Memory._instance
