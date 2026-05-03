import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import websockets


ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SO101 websocket client for native EvoDepth inference.")
    parser.add_argument("--server-uri", default="ws://localhost:9000")
    parser.add_argument("--lerobot-repo", default="/home/zhaobo/evo1_lerobot_evo_depth")
    parser.add_argument("--robot-id", default="5B14113885")
    parser.add_argument(
        "--robot-port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14113885-if00",
    )
    parser.add_argument(
        "--calibration-dir",
        default="/home/zhaobo/datasets/pick_green_block_on_yellow_block_20260427_119eps/calibration/5B14113885",
    )
    parser.add_argument(
        "--right-wrist-index",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0",
        help="OpenCV camera index or stable /dev/v4l/by-path path for right wrist.",
    )
    parser.add_argument(
        "--right-front-index",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0",
        help="OpenCV camera index or stable /dev/v4l/by-path path for right front.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        default=None,
        help="OpenCV FOURCC override. Leave unset to use the camera backend default exposure mode.",
    )
    parser.add_argument("--prompt", default="pick green block on yellow block")
    parser.add_argument("--num-policy-steps", type=int, default=300)
    parser.add_argument("--action-start-index", type=int, default=0)
    parser.add_argument("--action-steps", type=int, default=25)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--hold-hz", type=float, default=10.0)
    parser.add_argument(
        "--hold-last-during-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep resending the last motor target while waiting for the next policy chunk.",
    )
    parser.add_argument(
        "--log-torque-enable",
        action="store_true",
        help="Read and print Torque_Enable after connect and after each chunk.",
    )
    parser.add_argument("--no-action", action="store_true", help="Run perception/inference but do not command motors.")
    parser.add_argument(
        "--flip-right-wrist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip the right wrist image vertically before sending it to the inference server.",
    )
    parser.add_argument(
        "--flip-right-wrist-horizontal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flip the right wrist image horizontally after the optional vertical flip.",
    )
    return parser.parse_args()


def parse_index_or_path(value):
    value = str(value)
    return int(value) if value.isdigit() else value


def parse_fourcc(value):
    if value is None:
        return None
    value = str(value)
    return None if value.lower() in {"", "none", "auto", "default"} else value


def import_lerobot(lerobot_repo: str):
    sys.path.insert(0, lerobot_repo)
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    return ColorMode, OpenCVCameraConfig, SOFollowerRobotConfig, SOFollower


def build_robot(args):
    ColorMode, OpenCVCameraConfig, SOFollowerRobotConfig, SOFollower = import_lerobot(args.lerobot_repo)
    cameras = {
        "right_wrist": OpenCVCameraConfig(
            index_or_path=parse_index_or_path(args.right_wrist_index),
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=parse_fourcc(args.fourcc),
            color_mode=ColorMode.BGR,
        ),
        "right_front": OpenCVCameraConfig(
            index_or_path=parse_index_or_path(args.right_front_index),
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=parse_fourcc(args.fourcc),
            color_mode=ColorMode.BGR,
        ),
    }
    config = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.robot_port,
        calibration_dir=Path(args.calibration_dir),
        cameras=cameras,
    )
    return SOFollower(config)


def build_request(obs: dict, args) -> dict:
    state = np.array([obs[f"{name}.pos"] for name in ACTION_NAMES], dtype=np.float32)
    pad_dim = 24 - state.shape[0]
    if pad_dim < 0:
        raise ValueError(f"state dim {state.shape[0]} exceeds native max_state_dim=24")

    state_padded = np.pad(state, (0, pad_dim), constant_values=0)
    action_mask = [1] * state.shape[0] + [0] * pad_dim

    right_wrist = np.asarray(obs["right_wrist"], dtype=np.uint8)
    if args.flip_right_wrist:
        right_wrist = np.ascontiguousarray(np.flipud(right_wrist))
    if args.flip_right_wrist_horizontal:
        right_wrist = np.ascontiguousarray(np.fliplr(right_wrist))
    right_front = np.asarray(obs["right_front"], dtype=np.uint8)
    dummy = np.zeros_like(right_wrist, dtype=np.uint8)

    return {
        "image": [right_wrist.tolist(), right_front.tolist(), dummy.tolist()],
        "image_mask": [1, 1, 0],
        "state": state_padded.astype(float).tolist(),
        "action_mask": [[int(i) for i in action_mask]],
        "prompt": args.prompt,
    }


def action_dict(action_row) -> dict:
    return {f"{name}.pos": float(action_row[i]) for i, name in enumerate(ACTION_NAMES)}


class HoldLastCommand:
    def __init__(self, robot, hold_hz: float, enabled: bool):
        self.robot = robot
        self.period_s = 1.0 / hold_hz if hold_hz > 0 else 0.0
        self.enabled = enabled and self.period_s > 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._last_cmd = None
        self._last_send_t = 0.0
        self._error = None

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="hold-last-command", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("hold-last command thread failed") from self._error

    def get_observation(self):
        self._raise_if_failed()
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, cmd: dict):
        self._raise_if_failed()
        with self._lock:
            sent = self.robot.send_action(cmd)
            self._last_cmd = dict(sent)
            self._last_send_t = time.monotonic()
            return sent

    def read_torque_enable(self):
        self._raise_if_failed()
        with self._lock:
            return self.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)

    def _run(self):
        while not self._stop.wait(0.01):
            if self._last_cmd is None:
                continue
            if time.monotonic() - self._last_send_t < self.period_s:
                continue
            try:
                with self._lock:
                    if self._last_cmd is None:
                        continue
                    if time.monotonic() - self._last_send_t < self.period_s:
                        continue
                    self.robot.send_action(self._last_cmd)
                    self._last_send_t = time.monotonic()
            except Exception as exc:
                self._error = exc
                self._stop.set()
                print(f"[hold-last error] {exc}", flush=True)


async def run_policy(args):
    robot = build_robot(args)
    hold = HoldLastCommand(robot, args.hold_hz, args.hold_last_during_gaps and not args.no_action)
    robot.connect()
    hold.start()
    if hold.enabled:
        print(f"[hold-last] enabled at {args.hold_hz:.1f} Hz during chunk gaps", flush=True)
    if args.log_torque_enable:
        print(f"[torque_enable after connect] {hold.read_torque_enable()}", flush=True)
    try:
        async with websockets.connect(args.server_uri, max_size=100_000_000) as ws:
            print(f"connected to {args.server_uri}", flush=True)
            for step in range(args.num_policy_steps):
                obs = hold.get_observation()
                request = build_request(obs, args)
                await ws.send(json.dumps(request))
                response = await ws.recv()
                action_chunk = np.asarray(json.loads(response), dtype=np.float32)
                print(f"[policy step {step}] action_chunk shape={action_chunk.shape}", flush=True)

                start = args.action_start_index
                end = min(start + args.action_steps, action_chunk.shape[0])
                if start < 0 or start >= action_chunk.shape[0]:
                    raise ValueError(f"action_start_index={start} outside chunk length {action_chunk.shape[0]}")

                for action_i, action_row in enumerate(action_chunk[start:end], start=start):
                    cmd = action_dict(action_row)
                    if args.no_action:
                        print(f"[no-action chunk_idx={action_i}] {cmd}", flush=True)
                    else:
                        hold.send_action(cmd)
                        print(f"[sent chunk_idx={action_i}] {cmd}", flush=True)
                    time.sleep(1.0 / args.control_hz)
                if args.log_torque_enable:
                    print(f"[torque_enable after chunk step={step}] {hold.read_torque_enable()}", flush=True)
    finally:
        hold.stop()
        robot.disconnect()


if __name__ == "__main__":
    asyncio.run(run_policy(parse_args()))
