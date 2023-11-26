import pytest
import numpy as np

from offpolicy.runner.rnn.mpe_runner import MPERunner

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

def test_agent_pos_from_obs():
    # obs is np.arary of shape (n_envs, n_agents, obs_dim)
    obs = np.full((1, 3, 18), 1)
    obs[0,1:2,:] = 2
    obs[0,2:,:] = 3
    agent_poses = [MPERunner.agent_pos_from_obs(i, obs) for i in range(3)]
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
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,.5,.5,0,1,1,0],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,-.5,-.5,-1,0,0,-1],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,-1.5,-1.5,-2,-1,-1,-2]
        ]
    ])
    assert exp_obs.shape == (1, 3, 24), f"exp_obs should be of shape (1, 3, 24), not {exp_obs.shape}"
    # exp_obs = exp_obs.reshape((1, 3, 24))
    obs_with_priors = MPERunner.add_priors_to_obs(obs_to_modify, agent_poses_in_plan)
    assert obs_with_priors.shape == (1, 3, 24), f"obs_with_priors should be of shape (1, 3, 24), not {obs_with_priors.shape}"
    assert np.all(obs_with_priors == exp_obs), f"obs_with_priors is {obs_with_priors}, but should be {exp_obs}"

    