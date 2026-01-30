import sys
import numpy as np


def main() -> None:
    print("python:", sys.executable)

    import numberlink
    from numberlink import NumberLinkRGBVectorEnv
    from deepxube.domains.numberlink import NumberLink

    print("numberlink:", numberlink.__version__)

    # Vector env sanity check (fixed level, no generator)
    env = NumberLinkRGBVectorEnv(num_envs=2, level_id="builtin_5x5_rw_4c")
    obs, info = env.reset()
    print("obs shape:", obs.shape, "action_mask shape:", info["action_mask"].shape)

    mask = info["action_mask"]
    actions = []
    for i in range(mask.shape[0]):
        valid = np.where(mask[i] != 0)[0]
        actions.append(int(valid[0]) if valid.size else 0)
    obs, rewards, terminated, truncated, info = env.step(np.array(actions, dtype=np.int64))
    print("step ok:", rewards.tolist(), terminated.tolist(), truncated.tolist())

    # DeepXube domain sanity check (no training)
    domain = NumberLink(level_id="builtin_5x5_rw_4c")
    states, goals = domain.sample_goal_state_goal_pairs(1)
    print("goal solved:", domain.is_solved(states, goals)[0])
    actions = domain.sample_state_action(states)
    next_states, _ = domain.next_state(states, actions)
    print("next state ok:", isinstance(next_states[0], type(states[0])))


if __name__ == "__main__":
    main()
