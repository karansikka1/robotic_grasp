"""Record the visual BC policy with its actual auxiliary-stage predictions."""
import argparse
from functools import partial
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from bc_data.recording import VideoWriter, compose_video_frame, sha256, write_manifest
from bc_data.state_policy import STAGES
from bc_policy.models import load_policy
from evaluate import evaluate_one

PHASES = ('Approach from above', 'Descend to block', 'Close gripper', 'Lift block',
          'Move above support', 'Lower block', 'Release block', 'Retreat')


def font(size, bold=False):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


class StageVideoWriter(VideoWriter):
    def __init__(self, path, fps, *, predictions):
        super().__init__(path, fps)
        self.predictions = predictions
        self.frame_index = 0
        self.success_steps = 0

    def __enter__(self):
        super().__enter__()
        self.stream.height = 420
        return self

    def add_observation(self, observation):
        # Each post-action frame shows the prediction made for that action.
        canvas = Image.new('RGB', (512, 420), '#122f32')
        canvas.paste(Image.fromarray(compose_video_frame(observation)), (0, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'VISUAL BC / PREDICTED TASK STAGE', font=font(13, True), fill='#e1f0e9')
        draw.text((12, 44), 'FRONT', font=font(10, True), fill='white', stroke_width=1, stroke_fill='#24373b')
        draw.text((268, 44), 'WRIST', font=font(10, True), fill='white', stroke_width=1, stroke_fill='#24373b')
        draw.text((14, 305), 'Model prediction for this action', font=font(12), fill='#adcec3')
        title, color = 'Ready to begin', '#ffffff'
        if self.predictions:
            index = self.predictions[-1]['stage_index']
            if index == 16:
                title = 'Settle the stack'
            else:
                block = 'Green' if index < 8 else 'Blue'
                title = f'{block}: {PHASES[index % 8]}'
                color = '#aee4bc' if index < 8 else '#a9d4ff'
            draw.text((421, 306), f'{index+1} / 17', font=font(12), fill='#adcec3')
        draw.text((14, 327), title, font=font(22, True), fill=color)
        draw.text((14, 365), 'Stage predictions are not fed back as action inputs.', font=font(12), fill='#d1e3dc')
        self.success_steps = self.success_steps + 1 if observation['task_complete'] else 0
        result = ' | Stack complete' if self.success_steps >= 10 else (' | Checking stack' if self.success_steps else '')
        draw.text((14, 391), f'Action {self.frame_index} | {self.frame_index / self.fps:.2f} simulated s{result}',
                  font=font(12), fill='#adcec3')
        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).parent/'checkpoints/best_visual_bc.pt')
    parser.add_argument('--seed', type=int, default=155284722)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    digest = sha256(args.checkpoint)
    policy = load_policy(args.checkpoint, 'rgb_spatial', args.device)
    predictions = []

    def record_prediction(module, inputs, logits):
        # Observe the existing forward pass: no extra inference or memory update.
        values = logits.detach().reshape(-1, len(STAGES))[-1].cpu()
        index = int(values.argmax())
        predictions.append({'action': len(predictions) + 1, 'stage_index': index,
                            'stage': STAGES[index], 'stage_logits': values.tolist()})

    hook = policy.stage_head.register_forward_hook(record_prediction)
    try:
        result = evaluate_one(policy, 'rgb_spatial', args.seed, args.output, video=True,
                              video_writer=partial(StageVideoWriter, predictions=predictions))
    finally:
        hook.remove()
    assert len(predictions) == result['episode_steps']
    assert sha256(args.checkpoint) == digest
    result.update(checkpoint_sha256=digest, policy_type='rgb_spatial',
                  annotation='Unsmoothed argmax of the auxiliary stage head, for the action just executed',
                  predictions=predictions)
    write_manifest(result, args.output/'stage_predictions.json')
    print(f"success={result['success']}, actions={result['episode_steps']}, predictions={len(predictions)}")


if __name__ == '__main__':
    main()
