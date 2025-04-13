from browsergym.core.action.highlevel import HighLevelActionSet
from webexp.agents.prompt_builders.solver_prompt_builder import SolverPromptBuilder

class InjectablePromptBuilder(SolverPromptBuilder):
    """
    PromptBuilder that allows customization by injecting prompt snippets.
    """

    def __init__(self, action_set: HighLevelActionSet, **kwargs):
        super().__init__(action_set, **kwargs)
        self.snippets = []

    def inject(self, snippets: list[str]) -> None:
        """
        Injects the given prompt snippets into the agent's prompt.
        """
        if not snippets:
            return

        self.snippets = [snippet for snippet in snippets if snippet.strip()]


    def snippets_message(self) -> str:
        text = '\n'.join(self.snippets)
        return {
            "type": "text",
            "text": (
                "# Here are some suggestions that may be helpful for you based on past experiences of interacting with this website:\n"
                f"{text}\n"
            )  
        }
    
    def _build_messages(
        self,
        goal: str,
        thoughts: list[str | None],
        actions: list[str | None],
        axtree: str,
        last_action_error: str | None = None,
        completion_thought: str | None = None,
        completion_action: str | None = None
    ):
        system_messages = {"role": "system", "content": [self.system_message()]}
        user_messages = {
            "role": "user",
            "content": [
                self.goal_message(goal),
                self.axtree_message(axtree),
                self.action_space_message(self.action_set),
                self.action_history_messages(thoughts, actions),
            ]
        }

        if self.snippets:
            user_messages["content"].append(self.snippets_message())

        if last_action_error:
            user_messages["content"].append(self.last_action_error_message(last_action_error))
        
        user_messages["content"].append(self.next_action_request_message())
        
        output = { "prompt": [system_messages, user_messages] }
        
        if completion_thought or completion_action:
            assistant_messages = {
                "role": "assistant",
                "content": [self.completion_message(completion_thought, completion_action)]
            }
            output["completion"] = [assistant_messages]
        
        return output

    