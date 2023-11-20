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
        self.memoryBuffers = {}

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

    def commit(self, program_id):
        self.registers = self.memoryBuffers[program_id].copy()

    def buffer_write(self, program_id, value):
        if program_id not in self.memoryBuffers:
            self.memoryBuffers[program_id] = self.registers.copy()

        index = random.randint(0, self.numRegisters - 1)
        self.memoryBuffers[program_id][index] = value
        
    def buffer_reset(self):
        self.memoryBuffers = {}

    def memory_reset(self):
        self.registers = np.zeros(self.numRegisters)
        self.memoryBuffers = {}

    def display(self):
        for program_id, buffer in self.memoryBuffers.items():
            print(f"Buffer for program {program_id}")
            print(buffer)

def get_memory():
    if Memory._instance is None:
        raise RuntimeError("Memory instance has not been initialized.")
    return Memory._instance
