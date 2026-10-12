"""Video recording helper: collects frames on every control tick and writes mp4/gif."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .robot import SO101


class VideoRecorder:
    def __init__(self, camera: str = "front", fps: int = 25, every: int = 2, width: int = 640, height: int = 480):
        self.camera, self.fps, self.every = camera, fps, every
        self.width, self.height = width, height
        self.frames: list[np.ndarray] = []
        self._tick = 0

    def __call__(self, robot: SO101) -> None:
        self._tick += 1
        if self._tick % self.every == 0:
            self.frames.append(robot.render(self.camera, self.width, self.height))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gif":
            imageio.mimsave(path, self.frames, duration=1000 / self.fps, loop=0)
        else:
            imageio.mimsave(path, self.frames, fps=self.fps, macro_block_size=1)
        return path
