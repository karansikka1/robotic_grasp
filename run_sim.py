"""Run the simulator locally and save the rendered frames as an MP4."""

import argparse
from pathlib import Path
import tempfile

import av
import numpy as np

from motion_planning.simulator import IMAGE_HEIGHT, IMAGE_WIDTH, Simulator


SIMULATION_STEPS = 250
FRAMES_PER_SECOND = 20
VIDEO_WIDTH = IMAGE_WIDTH * 2


def compose_video_frame(observation: dict) -> np.ndarray:
    """Place the front and wrist RGB observations side by side."""
    front_image = np.flipud(observation["frontview_image"])
    wrist_image = np.flipud(observation["robot0_eye_in_hand_image"])
    return np.ascontiguousarray(
        np.concatenate((front_image, wrist_image), axis=1),
        dtype=np.uint8,
    )


def render_simulation(output_path: Path, steps: int, seed: int) -> None:
    """Run a local headless simulation and atomically write an MP4."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    random = np.random.default_rng(seed)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".mp4",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with av.open(
            str(temporary_path),
            mode="w",
            options={"movflags": "+faststart"},
        ) as container:
            video_stream = container.add_stream(
                "libx264", rate=FRAMES_PER_SECOND
            )
            video_stream.width = VIDEO_WIDTH
            video_stream.height = IMAGE_HEIGHT
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"preset": "fast", "crf": "20"}

            simulator = Simulator(has_renderer=False)
            try:
                simulator.reset()
                action_min, action_max = simulator.action_spec

                for _ in range(steps):
                    action = random.normal(0.0, 0.1, size=action_min.shape)
                    action = np.clip(action, action_min, action_max)
                    observation = simulator.step(action)
                    image = compose_video_frame(observation)
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in video_stream.encode(frame):
                        container.mux(packet)
            finally:
                simulator.close()

            for packet in video_stream.encode():
                container.mux(packet)

        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation.mp4"),
        help="MP4 output path (default: simulation.mp4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=SIMULATION_STEPS,
        help=f"number of simulation steps (default: {SIMULATION_STEPS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    render_simulation(args.output, args.steps, args.seed)
    print(
        f"Saved {args.steps}-step simulation video to "
        f"{args.output.expanduser().resolve()} "
        f"({args.output.expanduser().stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    main()
