from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self, experience: dict):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
