from orchestration import MainAgent
import uuid


class RunMainAgent:
    def __init__(self, llm=None):
        self.agent = MainAgent(llm=llm)
        self.threads = []
        self.thread_id = None
        self.config = {}

    def new_thread(self, topic: str, pdf_url: str):
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        return self.agent.run(topic, pdf_url)

    def existing_thread(self, topic: str, pdf_url: str):
        if not self.thread_id:
            raise ValueError("No existing thread_id to resume")
        return self.agent.run(topic, pdf_url)

    def get_current_state(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        return self.agent.build_graph().get_state(config)
