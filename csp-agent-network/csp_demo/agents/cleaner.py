from agents.base import BaseAgent
from protocols.csp import create_csp_message

class DataCleanerAgent(BaseAgent):
    def receive_csp_message(self, message):
        task = message.get("task", {})
        intent = task.get("intent")

        if intent == "deduplicate_records":
            result = {
                "success": True,
                "deduplicated_count": 12809,
                "stored_at": "vector://result/cleaned_customers_202406"
            }
        else:
            result = {"error": "Unknown intent"}

        response = create_csp_message(
            sender=self.agent_id,
            recipient=message["sender"],
            type_="TASK_RESULT",
            task=result,
            context_refs=[message["msg_id"]]
        )
        return response