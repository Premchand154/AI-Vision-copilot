class Memory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add(self, question, answer):
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        context = ""
        for q, a in self.history:
            context += f"Q: {q}\nA: {a}\n"
        return context

memory = Memory()