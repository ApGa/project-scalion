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
import litellm


# %% Rollout Generation
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
            traj = run_episode(goal, node, env, agent, evaluator, None, max_steps=max_steps)
            evaluator.evaluate(traj)
            trajs.append(traj)
        agent.reset()
    return trajs


# %% Summarize Search Strategies

SYS_MSG="""You are a proficient computer user on navigating map websites. Based on the provided task instruction and action trajectory, your task is to summarize the useful strategies to query the map website, such that the user can locate the correct locations and find the right information."""

INSTRUCTION="""# Instruction
You are a proficient computer user on navigating map websites. Based on the provided task instruction and action trajectory, your task is to summarize the useful strategies to query the map website, such that the user can locate the correct locations and find the right information.

## Example
Instruction: Tell me the coordinates of Apple Store near Pitt in DD format
Action trajectory:
```
fill('140', 'Apple Store near Upitt')  # Entered "Apple Store near Upitt" in the search field to search for store location.
click('143')  # Clicked the "Go" button to initiate the search for Apple Store near Upitt.
click('450')  # Searched for "Apple Store near Upitt" but found no matching results.
click('450')  # Zoomed in on the map to look for nearby Apple Store locations.
fill('140', 'Apple Store')  # Entered "Apple Store" in the search bar to perform a broader search.
click('143')  # Clicked the "Go" button to perform a search for "Apple Store".
fill('140', 'Apple Store Pittsburgh')  # Entered "Apple Store" in the search bar to perform a broader search.
click('143')  # Clicked the "Go" button to perform a search for "Apple Store".
```

Search Strategy:
1. Avoid or expand abbreviations in the search query, e.g., remove "Upitt" and only search for "Apple Store".
2. Infer city names from the context, e.g., "Pittsburgh" from "Upitt"; and add them to the search query, e.g, "Apple Store Pittsburgh".
"""

def summarize_experience(trajectory: Trajectory) -> list[str]:
    instruction = trajectory.goal
    steps = '\n'.join([f"{s.action}  # {s.thought}" for s in trajectory.steps])
    test_query = f"##Test Example\nInstruction: {instruction}\n" + f"Action Trajectory: \n```{steps}\n```\n\nSearch Strategy:"
    
    messages = [
        {"role": "system", "content": SYS_MSG},
        {"role": "user", "content": INSTRUCTION},
        {"role": "user", "content": test_query}
    ]
    response = litellm.completion(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://cmu.litellm.ai"),
        model="openai/neulab/claude-3-5-sonnet-20241022",
        messages=messages,
        temperature=0.0,
        n=1,
    )["choices"][0]["message"]["content"]
    print(response)
    cont = input("Continue? (y/n): ")

    return response.split('\n\n')


def summarize_experiences(trajs: list[Trajectory]) -> list[str]:
    snippets = []
    for traj in trajs:
        if traj.misc["evaluation_info"]["reward"] == 1.0:
            snippets.extend(summarize_experience(traj))
    return snippets


# %% Inject Snippets into Agents

def mutate_agents(agents: list[InjectableAgent], snippets: list[str]) -> list[Agent]:
    #TODO
    for agent in agents:
        agent.inject(snippets)
    return agents


# %% Main Algorithm

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
    for i in range(num_iterations):
        trajs = generate_rollouts(goal, agents, node, env, evaluator, num_rollouts_per_agent, max_steps)
        for j, traj in enumerate(trajs):
            traj_save_dir = os.path.join(config.exp_dir, f"{i}_{j}")
            os.makedirs(traj_save_dir, exist_ok=True)
            traj.save(traj_save_dir)
        snippets = summarize_experiences(trajs)
        with open(config.memory_path, 'a+') as fw:
            fw.write('\n\n'.join(snippets))
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

