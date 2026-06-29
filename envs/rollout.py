# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def rollout(
    env: Any,
    agent: Any,
    num_episodes: int,
    ctx: torch.Tensor | None = None,
    record: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[List[List[np.ndarray]]]]:
    observation, info = env.reset()
    agent.reset()
    returns, lengths, infos = [0.0], [0], [[info]]
    ctx = {} if ctx is None else {"z": ctx}
    frames: Optional[List[List[np.ndarray]]] = [[env.render()]] if record else None
    while True:
        input_dict = {"obs": torch.tensor(observation, device=agent.device, dtype=torch.float32)[None], **ctx}
        action = agent.act(**input_dict).cpu().numpy()[0]
        observation, reward, terminated, truncated, info = env.step(action)
        if record:
            frames[-1].append(env.render())
        done = terminated or truncated
        returns[-1] += reward
        lengths[-1] += 1
        infos[-1] += [info]
        if done:
            if len(returns) >= num_episodes:
                break
            observation, info = env.reset()
            agent.reset()
            returns.append(0.0)
            lengths.append(0)
            infos.append([info])
            if record:
                frames.append([env.render()])
    return {"reward": returns, "length": lengths}, infos, frames
