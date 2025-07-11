import cv2
import imageio.v2 as imageio
import numpy as np



def save_gif(frames: np.ndarray, filename: str = "tetris.gif", fps: int = 10, resize_to: tuple[int, int] = None):
    """
    Save a sequence of RGB frames as a GIF.

    Args:
        frames (np.ndarray): A NumPy array of shape [t, h, w, 3] and dtype uint8.
        filename (str): Output GIF file path.
        fps (int): Frames per second.
        resize_to (tuple[int, int], optional): Resize each frame to this (width, height).
    """

    t, c, h, w = frames.shape

    frames.reshape((t, h, w, c))

    assert frames.ndim == 4 and frames.shape[-1] == 3, "Expected shape [t, h, w, 3]"
    assert frames.dtype == np.uint8, "Frames must be of dtype uint8"

    if resize_to is not None:
        resized_frames = [cv2.resize(f, resize_to, interpolation=cv2.INTER_NEAREST) for f in frames]
    else:
        resized_frames = frames

    imageio.mimsave(filename, resized_frames, fps=fps)
    print(f"Saved {len(frames)} frames to {filename}")