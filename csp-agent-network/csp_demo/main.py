from agents.planner import PlannerAgent
from agents.cleaner import DataCleanerAgent
from memory.chroma_store import ChromaVectorStore
from api.server import run_api

# Create memory and agents
memory = ChromaVectorStore()
planner = PlannerAgent("agent://planner", memory)
cleaner = DataCleanerAgent("agent://data_cleaner", memory)

# Simulate sending a CSP message from planner to cleaner
message = planner.create_task_message(
    recipient="agent://data_cleaner",
    intent="deduplicate_records",
    resource_ref="vector://dataset/profiles",
    parameters={"threshold": 0.95},
)

# Log sending
print("\n--- Sending CSP Message ---")
print(message)

# Deliver and respond
response = cleaner.receive_csp_message(message)

print("\n--- Received Response ---")
print(response)

# Store response in memory
memory.store("task_result", response)

# Optionally run API
if __name__ == "__main__":
    run_api(planner, cleaner, memory)