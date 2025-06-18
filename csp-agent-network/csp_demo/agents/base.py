class BaseAgent:
    def __init__(self, agent_id, memory):
        self.agent_id = agent_id
        self.memory = memory

    def log(self, message):
        print(f"[{self.agent_id}] {message}")