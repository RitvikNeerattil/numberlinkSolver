import numpy as np

from numberlink import GeneratorConfig, VariantConfig, NumberLinkRGBVectorEnv


def main() -> None:
    env = NumberLinkRGBVectorEnv(
        num_envs=1,
        generator=GeneratorConfig(width=4, height=4, colors=3, mode="random_walk"),
        variant=VariantConfig(must_fill=True, cell_switching_mode=False),
    )

    _, info = env.reset()
    print("initial solved:", bool(info["solved"][0]))

    sol = env.get_solution()
    print("env.get_solution length:", None if sol is None else len(sol))
    if not sol:
        return

    for step_idx, act in enumerate(sol):
        mask = info["action_mask"][0]
        is_valid = bool(mask[int(act)] != 0)
        print(f"step {step_idx}: act={int(act)} valid={is_valid}")
        obs, reward, terminated, truncated, info = env.step(np.array([int(act)], dtype=np.int64))
        if bool(info["solved"][0]):
            print(f"solved at step {step_idx + 1}")
            break
        if bool(terminated[0]) or bool(truncated[0]):
            print(f"terminated={bool(terminated[0])} truncated={bool(truncated[0])} at step {step_idx + 1}")
            break

    print("final solved:", bool(info["solved"][0]))


if __name__ == "__main__":
    main()
