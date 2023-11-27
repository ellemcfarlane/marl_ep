import pytest
import numpy as np

from offpolicy.runner.rnn.mpe_runner import MPERunner
from offpolicy.envs.mpe.core import Agent
from offpolicy.envs.mpe.scenarios.simple_spread import Scenario

# parameterize
@pytest.mark.parametrize(
    "test_input, expected",
    [
        (([1, 1], np.array([[1.5, 1.5], [1,2], [2,1]])), ([[.5, .5], [0, 1], [1, 0]]))
    ],
)
def test_get_epistemic_priors(test_input, expected):
    priors = MPERunner.get_epistemic_priors(np.array(test_input[0]), np.array(test_input[1]))
    assert np.all(priors == expected), "priors should be equal to expected"

def test_agent_pos_from_joint_obs():
    # obs is np.arary of shape (n_envs, n_agents, obs_dim)
    obs = np.full((1, 3, 18), 1)
    obs[0,1:2,:] = 2
    obs[0,2:,:] = 3
    agent_poses = [MPERunner.agent_pos_from_joint_obs(i, obs) for i in range(3)]
    print('obs', obs)
    exp_agent_poses = [np.array([1,1]), np.array([2,2]), np.array([3,3])]
    for pos in agent_poses:
        assert pos.shape == (2,), f"agent_pos should be of shape (2,), not {pos.shape}"
    for i in range(3):
        assert np.all(agent_poses[i] == np.array(exp_agent_poses[i])), f"agent_pose {agent_poses[i]} != {exp_agent_poses[i]} for agent {i}"

# obs is np.arary of shape (n_envs, n_agents, obs_dim)
def test_add_priors_to_obs():
    agent_poses_in_plan = np.array([[1.5, 1.5], [1,2], [2,1]])
    obs_to_modify = np.array([
        [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
        ]
    ])
    # obs_to_modify = obs_to_modify.reshape((1, 3, 18))
    exp_obs = np.array([
        [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,.5,  .5,  0,  1 ,1, 0],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,-.5, -.5, -1, 0, 0, -1],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,-1.5,-1.5,-2,-1,-1, -2]
        ]
    ])
    assert exp_obs.shape == (1, 3, 24), f"exp_obs should be of shape (1, 3, 24), not {exp_obs.shape}"
    # exp_obs = exp_obs.reshape((1, 3, 24))
    obs_with_priors = MPERunner.add_priors_to_obs(obs_to_modify, agent_poses_in_plan)
    assert obs_with_priors.shape == (1, 3, 24), f"obs_with_priors should be of shape (1, 3, 24), not {obs_with_priors.shape}"
    assert np.all(obs_with_priors == exp_obs), f"obs_with_priors is {obs_with_priors}, but should be {exp_obs}"

def test_joint_pos_in_epistemic_plan():
    # rollout_obss has shape (n_agents, ep_len + 1, n_envs, obs_dim)
    n_agents = 3
    ep_len = 1
    n_envs = 1
    obs_dim = 5
    # rollout_obss = np.array(
    #     [
    #         [
    #             [[1,1,1,1,1]], [[2,2,2,2,2]]
    #         ],
    #         [
    #             [[3,3,3,3,3]], [[4,4,4,4,4]]
    #         ],
    #         [
    #             [[5,5,5,5,5]], [[6,6,6,6,6]]
    #         ]
    #     ]
    # )
    rollout_obss = np.array([
        [[[1,1,1,1,1],[3,3,3,3,3],[5,5,5,5,5]]],
        [[[2,2,2,2,2],[4,4,4,4,4],[6,6,6,6,6]]]
    ])
    assert rollout_obss.shape == (ep_len + 1, n_envs, n_agents, obs_dim), f"rollout_obss should have shape (n_agents, ep_len + 1, n_envs, obs_dim), not {rollout_obss.shape}"
    agent_poses = MPERunner.joint_pos_in_epistemic_plan(rollout_obss, 0, n_agents)
    assert agent_poses.shape == (n_agents, 2), f"agent_poses should have shape (n_agents, 2), not {agent_poses.shape}"
    exp_agent_poses = np.array([[1,1], [3,3], [5,5]])
    assert np.all(agent_poses == exp_agent_poses), f"agent_poses should be {exp_agent_poses}, not {agent_poses}"
    agent_poses1 = MPERunner.joint_pos_in_epistemic_plan(rollout_obss, 1, n_agents)
    assert agent_poses1.shape == (n_agents, 2), f"agent_poses should have shape (n_agents, 2), not {agent_poses.shape}"
    exp_agent_poses1 = np.array([[2,2], [4,4], [6,6]])

@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((-1, (4,4)), (True)),
        ((-1, (2,2)), (True)),
        ((0, (4,4)), (True)),
        ((0, (2,2)), (True)),
        ((1, (0,2)), (False)),
        ((1, (2,2)), (True)),
        ((1, (1,2)), (True)),
        ((1, (2,3)), (True)),
        ((1, (3,2)), (True)),
        ((1, (2,1)), (True)),
    ],
)
def test_within_fov(test_input, expected):
    agent = Agent()
    agent.state.p_pos = np.array([2,2])
    agent.fov = test_input[0]
    other_agent = Agent()
    other_agent.state.p_pos = np.array(test_input[1])
    result = Scenario.within_fov(agent, other_agent)
    assert result == expected, f"within_fov should be {expected}, not {result}"