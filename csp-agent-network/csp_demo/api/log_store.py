from datetime import datetime

class CSPLogStore:
    def __init__(self, limit=100):
        self.logs = []
        self.limit = limit

    def log(self, message):
        entry = {
            "time": datetime.utcnow().isoformat(),
            "msg": message
        }
        self.logs.append(entry)
        if len(self.logs) > self.limit:
            self.logs.pop(0)

    def get_logs(self):
        return list(reversed(self.logs))