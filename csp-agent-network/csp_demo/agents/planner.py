from agents.base import BaseAgent
from protocols.csp import create_csp_message

class PlannerAgent(BaseAgent):
    def create_task_message(self, recipient, intent, resource_ref, parameters):
        task = {
            "intent": intent,
            "resource_ref": resource_ref,
            "parameters": parameters,
            "priority": "high"
        }
        return create_csp_message(
            sender=self.agent_id,
            recipient=recipient,
            type_="TASK_REQUEST",
            task=task
        )