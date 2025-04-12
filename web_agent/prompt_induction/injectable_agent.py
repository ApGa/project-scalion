from browsergym.core.action.highlevel import HighLevelActionSet
from webexp.agents.base_agent import AgentFactory
from webexp.agents.solver_agent import SolverAgent

from injectable_prompt_builder import InjectablePromptBuilder

@AgentFactory.register
class InjectableAgent(SolverAgent):
    """
    Browser Agent that allows customization by injecting prompt snippets.
    """
    def __init__(
            self,
            model_id: str,
            base_url: str | None = None,
            api_key: str | None = None,
            temperature: float = 1.0,
            char_limit: int = -1,
            demo_mode: str = 'off',
    ):
        """
        Initialize the agent.
        """
        super().__init__(model_id, base_url, api_key, temperature, char_limit, demo_mode)

        self.prompt_builder = InjectablePromptBuilder(action_set=self.action_set)


    def inject(self, snippets: list[str]) -> None:
        """
        Injects the given prompt snippets into the agent's prompt.
        """
        if not snippets:
            return
        
        self.prompt_builder.inject(snippets)

    def get_action(self, obs: dict, oracle_action:tuple[str, str] = None, **kwargs) -> tuple[str, dict]:
        """
        Get the action for the given observation.

        Args:
            obs (dict): The observation from the environment.
            oracle_action tuple[str, str]: Tuple of (action, thought) to use if available instead of generating a new one.

        Returns:
            str: The action to take.
        """

        raw_action, misc = super().get_action(obs, oracle_action, **kwargs)

        print(raw_action)

        return raw_action, misc