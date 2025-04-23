from collections import deque
import itertools
import random
import torch

REPLAY_BUFFER_SAVE_PATH = 'replay_buffer.pth'

class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self, experience: dict):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def recemt_prioritized_sample(self, batch_size: int, delayed_update_interval: int):
        # half_batch_size = batch_size // 2
        # recent_batch = list(itertools.islice(self.buffer, len(self.buffer) - half_batch_size, len(self.buffer)))
        # past_buffer = list(itertools.islice(self.buffer, 0, len(self.buffer) - half_batch_size))
        # past_batch = random.sample(past_buffer, half_batch_size)
        recent_buffer = list(itertools.islice(self.buffer, len(self.buffer) - delayed_update_interval, len(self.buffer)))
        past_buffer = list(itertools.islice(self.buffer, 0, len(self.buffer) - delayed_update_interval))
        recent_batch_size = delayed_update_interval // 2
        past_batch_size = batch_size - recent_batch_size
        recent_batch = random.sample(recent_buffer, recent_batch_size)
        past_batch = random.sample(past_buffer, past_batch_size)
        return recent_batch + past_batch
    
    def save(self):
        torch.save(self.buffer, REPLAY_BUFFER_SAVE_PATH)

    def load(self):
        self.buffer = torch.load(REPLAY_BUFFER_SAVE_PATH, weights_only=False)

    def __len__(self):
        return len(self.buffer)
