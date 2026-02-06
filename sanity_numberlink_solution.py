import numpy as np

from numberlink import GeneratorConfig, VariantConfig, NumberLinkRGBVectorEnv
from deepxube.domains.numberlink import NumberLink, NumberLinkAction


def main() -> None:
    env = NumberLinkRGBVectorEnv(
        num_envs=1,
        generator=GeneratorConfig(width=4, height=4, colors=3, mode="random_walk"),
        variant=VariantConfig(must_fill=True, cell_switching_mode=False),
    )

    env.reset()
    sol = env.get_solution()
    print("env.get_solution length:", None if sol is None else len(sol))

    if sol:
        for act in sol:
            env.step(np.array([act], dtype=np.int64))
        print("env solved after replay:", bool(env._compute_solved_mask()[0]))

    dom = NumberLink(width=4, height=4, colors=3, mode="random_walk")
    state = dom._capture_state(env, 0)

    if sol:
        states = [state]
        for act in sol:
            states, _ = dom.next_state(states, [NumberLinkAction(int(act))])
        print("domain is_solved after replay:", dom.is_solved(states, [None])[0])


if __name__ == "__main__":
    main()
