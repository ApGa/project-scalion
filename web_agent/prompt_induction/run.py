from browsergym.core.env import BrowserEnv
from browsergym.experiments.loop import EnvArgs
from omegaconf import OmegaConf as oc
from dataclasses import dataclass
from webexp.agents.base_agent import Agent,AgentFactory
from webexp.explore.core.agent import wrap_agent_for_callback_protocol
from webexp.explore.core.episode import run_episode
from webexp.explore.core.evaluator import Evaluator
from webexp.explore.core.node import Node
from webexp.explore.core.trajectory import Trajectory
from injectable_agent import InjectableAgent
import argparse
import os

@dataclass
class RunPromptInductionConfig:
    """
    Configuration for running an agent for an episode.
    
    Attributes:
        agent_factory_args (dict): Arguments for the agent factory.
        env_args (dict): Arguments for the environment.
        exp_dir (str): Directory for storing experiment results. Default is "./results".
    """
    agent_factory_args: dict
    env_args: dict
    evaluator: dict
    exp_dir: str
    goals: list[str]
    max_steps: int
    num_rollouts_per_agent: int
    num_iterations: int


def generate_rollouts(goal: str, agents: list[Agent], node: Node, env: BrowserEnv, evaluator: Evaluator, num_rollouts_per_agent: int, max_steps: int) -> list[Trajectory]:
    trajs = []
    for agent in agents:
        for _ in range(num_rollouts_per_agent):
            trajs.append(run_episode(goal, node, env, agent, evaluator, None, max_steps=max_steps))
        agent.reset()
    return trajs

def summarize_experiences(trajs: list[Trajectory]) -> str:
    return "" #TODO

def mutate_agents(agents: list[InjectableAgent], snippets: list[str]) -> list[Agent]:
    #TODO
    for agent in agents:
        agent.inject(snippets)
    return agents

def run_algo(
    goal: str,
    agents: list[Agent],
    env: BrowserEnv,
    evaluator: Evaluator,
    node: Node,
    num_rollouts_per_agent: int,
    max_steps: int,
    num_iterations: int,
) -> list[Agent]:
    for _ in range(num_iterations):
        trajs = generate_rollouts(goal, agents, node, env, evaluator, num_rollouts_per_agent, max_steps)
        snippets = summarize_experiences(trajs)
        agents = mutate_agents(agents, snippets)
    return agents


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run prompt induction.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config: RunPromptInductionConfig = oc.load(args.config)
    oc.resolve(config)
    config_dict = oc.to_container(config)

    os.makedirs(config.exp_dir, exist_ok=True)

    agents = [
        wrap_agent_for_callback_protocol(
            AgentFactory.create_agent(**agent_config),
        )
        for agent_config in config_dict['agents']
    ]

    env: BrowserEnv = EnvArgs(**config_dict['env_args']).make_env(
        action_mapping=lambda x: x,
        exp_dir=config.exp_dir
    )

    env = env.unwrapped
    env.reset()
    root_url = env.page.url

    node = Node(root_url, {}, {}, [], "", [], False, config.exp_dir, misc={})

    evaluator = Evaluator(**config.evaluator)

    # TODO: We might want to mix trajectories from multiple goals at a time to mutate agent.
    for goal in config.goals:
        agents = run_algo(
            goal,
            agents,
            env,
            evaluator,
            node,
            config.num_rollouts_per_agent,
            config.max_steps,
            config.num_iterations
        )

    for agent in agents:
        print(agent.prompt_builder.snippets)

