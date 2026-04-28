# Go1 Locomotion Implementation Plan

## Goal

Create a Go1 version of the existing Go2 locomotion sample under `examples/locomotion_go1`.

Target files:

- `go1_env.py`
- `go1_train.py`
- `go1_eval.py`

The first target is not policy quality. The first target is a reliable Go1 training/evaluation loop that can build,
step, train briefly, save checkpoints, and replay a checkpoint.

## Current Feasibility Result

`genesis/assets/urdf/go1_description/urdf/go1.urdf` can be loaded by Genesis.

Observed result from `check_go1_urdf.py`:

- `scene.build(n_envs=1)` succeeds.
- `scene.step()` succeeds for 10 steps.
- `n_dofs = 18`, meaning 6 floating-base DOFs plus 12 motor DOFs.
- Required Go1 motor joints are present:
  - `FR_hip_joint`, `FR_thigh_joint`, `FR_calf_joint`
  - `FL_hip_joint`, `FL_thigh_joint`, `FL_calf_joint`
  - `RR_hip_joint`, `RR_thigh_joint`, `RR_calf_joint`
  - `RL_hip_joint`, `RL_thigh_joint`, `RL_calf_joint`

Known warning:

```text
Neutral robot position (qpos0) exceeds joint limits.
```

This is expected because URDF neutral joint values are zero, while calf joints require negative angles. The locomotion
env should immediately reset to configured `default_joint_angles`, where calf joints use valid negative values. Treat
this as acceptable unless runtime resets or first simulation steps show instability.

## Important Go1 vs Go2 Differences

Static comparison from URDF:

| Item | Go2 | Go1 | Impact |
| --- | ---: | ---: | --- |
| Total mass from URDF inertials | ~15.019 kg | ~13.101 kg | PD gains and reward scale may need retuning. |
| Links before fixed-link merge | 29 | 46 | Go1 has more fixed sensor/rotor links, but Genesis merges them. |
| Joints before fixed-link merge | 28 | 45 | After merge, Go1 runtime check showed 13 joints. |
| Hip limit example | `[-1.0472, 1.0472]` | `[-0.863, 0.863]` | Go1 hip range is narrower. |
| Thigh limit example | `[-1.5708, 3.4907]` | `[-0.686, 4.501]` | Go1 thigh range is shifted. |
| Calf limit example | `[-2.7227, -0.83776]` | `[-2.818, -0.888]` | Similar, but default pose must stay inside limits. |

Go2 policy checkpoints should not be expected to work directly on Go1. The action/observation dimensions match, but
dynamics, limits, mass, and reward targets differ.

## Phase 0 - Keep the URDF Check Script

File:

- `check_go1_urdf.py`

Purpose:

- Keep this as the smoke test for the asset itself.
- Run it before changing env/training code if URDF or mesh files change.

Command:

```bash
python examples/locomotion_go1/check_go1_urdf.py
```

Pass condition:

- Ends with `GO1 URDF load/build/step OK`.
- Prints all 12 required motor joints.

## Phase 1 - Create `go1_env.py`

Start from `examples/locomotion/go2_env.py`, but make the Go1-specific parts explicit.

Required changes:

- Rename class from `Go2Env` to `Go1Env`.
- Set robot URDF to `urdf/go1_description/urdf/go1.urdf`.
- Keep the 12-action interface and `joint_names` contract.
- Keep `actions_dof_idx = torch.argsort(self.motors_dof_idx)` because Go1 runtime DOF order is grouped by joint type:
  hips first, thighs second, calves third.
- Keep `control_dofs_position(..., slice(6, 18))` only after confirming `n_dofs == 18`.
- Add a small validation block after `scene.build()`:
  - assert all configured motor joints exist.
  - assert `robot.n_dofs == 18`.
  - assert configured default joint angles are inside URDF/runtime limits if available.

Initial config choices:

- Start with the existing Go2 defaults:
  - `base_init_pos = [0.0, 0.0, 0.42]`
  - hip default `0.0`
  - front thigh `0.8`
  - rear thigh `1.0`
  - calf `-1.5`
  - `kp = 20.0`
  - `kd = 0.5`
  - `action_scale = 0.25`
- Then tune only if the sanity tests show instability.

Checks after Phase 1:

1. Instantiate `Go1Env(num_envs=1)` on CPU.
2. Call `reset()`.
3. Step with zero actions for 100-300 steps.
4. Confirm:
   - no rigid solver error envs,
   - base does not immediately explode,
   - reset buffers are not constantly true,
   - base height is plausible.

This is the first point where PD gains and base height become real issues rather than guesses.

## Phase 2 - Create `go1_train.py`

Start from `examples/locomotion/go2_train.py`.

Required changes:

- Import `Go1Env` from `go1_env`.
- Default experiment name: `go1-walking`.
- Keep PPO config initially unchanged.
- Keep `num_actions = 12`.
- Keep command shape unchanged:
  - `num_commands = 3`
  - x velocity, y velocity, yaw velocity

Initial training command:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking-smoke -B 64 --max_iterations 2
```

Smoke pass condition:

- Training starts.
- No exception in rollout.
- `logs/go1-walking-smoke/cfgs.pkl` is written.
- At least one iteration completes.

Then scale up:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking -B 4096 --max_iterations 101
```

## Phase 3 - Reward and PD Tuning Checkpoints

Do not tune everything at once. Use staged checks.

### Checkpoint A - Default Pose and PD Hold

When:

- Immediately after `go1_env.py` exists.

Run:

- `num_envs=1`
- zero actions
- no learning

Look for:

- Does Go1 stand near the configured default pose?
- Does it jitter heavily?
- Does it fall before commands matter?

Tune if needed:

- If jittering: reduce `kp`, increase/decrease `kd` carefully.
- If joints lag or collapse: increase `kp` modestly.
- If control is too aggressive: reduce `action_scale` before changing reward.

Initial PD baseline:

- `kp = 20.0`
- `kd = 0.5`

Candidate safe tuning ranges:

- `kp`: 15.0 to 30.0
- `kd`: 0.4 to 1.0
- `action_scale`: 0.15 to 0.25

### Checkpoint B - Base Height Target

When:

- After zero-action hold is stable.

Current Go2 reward target:

- `base_height_target = 0.3`

Question for Go1:

- Is Go1's steady standing base height closer to `0.28`, `0.30`, or another value?

Plan:

- Log or print mean base height during zero-action hold.
- Set `base_height_target` near the observed stable standing height.
- Keep the penalty weight unchanged initially: `base_height = -50.0`.

### Checkpoint C - Short PPO Training

When:

- After PD hold and base height are plausible.

Run:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking-test -B 512 --max_iterations 20
```

Watch:

- reward trend,
- episode length,
- reset frequency,
- `rew_tracking_lin_vel`,
- `rew_base_height`,
- `rew_similar_to_default`,
- solver error resets.

Tune if needed:

- If it crouches or hops to game reward: adjust `base_height_target` or `base_height` penalty.
- If it freezes near default: reduce `similar_to_default` penalty magnitude.
- If actions are noisy: increase `action_rate` penalty magnitude or reduce `action_scale`.
- If it cannot track velocity: check PD first, then tracking reward.

### Checkpoint D - Full Training

When:

- Short PPO run produces non-catastrophic movement.

Run:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking -B 4096 --max_iterations 101
```

Only after this point should policy quality be judged.

## Phase 4 - Create `go1_eval.py`

Start from `examples/locomotion/go2_eval.py`.

Required changes:

- Import `Go1Env`.
- Default experiment name: `go1-walking`.
- Keep checkpoint loading from `logs/<exp_name>/model_<ckpt>.pt`.
- Keep pause/restart/follow viewer controls.

Evaluation command:

```bash
python examples/locomotion_go1/go1_eval.py -e go1-walking --ckpt 100
```

Pass condition:

- Viewer starts.
- Checkpoint loads.
- Policy steps without shape mismatch.
- Go1 moves without immediate solver failure.

## Implementation Order

1. Keep `check_go1_urdf.py` as asset smoke test.
2. Add `go1_env.py` with Go1 URDF and validation.
3. Add a tiny env smoke test command or script if needed.
4. Add `go1_train.py`.
5. Run `-B 64 --max_iterations 2`.
6. Inspect PD/default pose/base height.
7. Run `-B 512 --max_iterations 20`.
8. Tune reward/PD only after observing the short run.
9. Add `go1_eval.py`.
10. Run evaluation against a trained checkpoint.

## Initial Go/No-Go Criteria

Go:

- `check_go1_urdf.py` passes.
- `Go1Env` builds with `n_dofs == 18`.
- Zero-action stepping is stable enough to begin PPO smoke training.
- PPO smoke training completes at least 2 iterations.

No-go / investigate:

- Required joint names cannot be found.
- Runtime DOF count is not 18.
- Default joint pose violates limits.
- Solver error envs trigger immediately under zero actions.
- The robot falls before reward tuning can reasonably matter.

