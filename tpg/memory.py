import numpy as np
import logging
import random

class Memory:

    def __init__(self, numRegisters):
        self.registers = np.zeros(numRegisters)
        self.buffers = {}
        
    def read(self, index):
        if index < 0:
            raise RuntimeError("Memory read failed. Provided index is a negative value.")
        else:
            return self.registers[index % len(self.registers)]

    def commit(self, program_id):
        #if the program_id isn't in memory buffers, the program didn't alter memory (totally ok)
        if program_id in self.buffers:
            self.registers = self.buffers[program_id].copy()

    def write(self, program_id, register, value):
        if program_id not in self.buffers:
            self.buffers[program_id] = self.registers.copy()
            
        self.buffers[program_id][register] = value

    # Wipe the write-buffers to save memory
    def clear(self):
        self.buffers = {}
