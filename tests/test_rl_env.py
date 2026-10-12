import numpy as np
from stable_baselines3.common.env_checker import check_env

from so101_workcell.rl.env import ACTION_SCALE, CUBE, make_env


def oracle_episode(env, *, verbose: bool = False) -> bool:
    """Scripted controller through the env's action interface: approach, descend, close, lift.

    Steers the *commanded* fingertip target (env._target), not the lagging actual position,
    so it behaves like a sane policy under control latency.
    """
    env.reset()
    rb = env.robot
    phase, k, done = "approach", 0, False
    while not done:
        c = rb.body_pos(CUBE)
        gs = c - env._close_dir * (0.0125 + 0.0009 + 0.002)
        gs[2] = c[2] + 0.003
        cmd = env._target
        if phase == "approach":
            tgt = gs + np.array([0, 0, 0.04])
            a = np.r_[np.clip(0.5 * (tgt - cmd) / ACTION_SCALE, -1, 1), -1]
            if np.linalg.norm(tgt - cmd) < 0.002:
                phase, k = "descend", 0
        elif phase == "descend":
            a = np.r_[np.clip(0.5 * (gs - cmd) / ACTION_SCALE, -1, 1), -1]
            if np.linalg.norm(gs - cmd) < 0.002:
                phase, k = "settle", 0
        elif phase == "settle":
            a = np.array([0, 0, 0, -1.0])
            k += 1
            if k >= 3:
                phase, k = "close", 0
        elif phase == "close":
            a = np.array([0, 0, 0, 1.0])
            k += 1
            if k >= 5:
                phase = "lift"
        else:
            a = np.array([0, 0, 1.0, 1.0])
        _obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if verbose and (env._steps % 8 == 0 or info["success"]):
            print(f"  t={env._steps:3d} {phase:8s} dist={info['dist']*1000:5.1f}mm grasped={info['grasped']} lift={info['lift']:.2f} r={r:.2f} cube z={c[2]:.3f} grip={rb.gripper_opening():.2f}")
    return env.info.success


def test_env_api():
    env = make_env(seed=0, dr=True)
    check_env(env, warn=True)
    env.close()


def test_oracle_succeeds_without_randomization():
    env = make_env(seed=3, dr=False)
    ok = [oracle_episode(env) for _ in range(5)]
    env.close()
    assert sum(ok) >= 4, ok


def test_oracle_mostly_succeeds_with_randomization():
    env = make_env(seed=4, dr=True)
    ok = [oracle_episode(env) for _ in range(8)]
    env.close()
    assert sum(ok) >= 5, ok
