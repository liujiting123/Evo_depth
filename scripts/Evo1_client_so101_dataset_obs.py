import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from PIL import Image


ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


DEFAULT_TRAIN_STATE = [
    15.296703,
    -103.69231,
    89.8022,
    76.52747,
    0.835165,
    2.105935,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run native EvoDepth inference on a fixed training-dataset observation and optionally execute on SO101."
    )
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
        "--frame-dir",
        default="/tmp/evodepth_rerun_train_43554",
        help="Either a single-observation dir or an episode dir containing obs_000000/... subdirs.",
    )
    parser.add_argument("--image-1", default="image_1_rgb.png", help="Training right_wrist image filename.")
    parser.add_argument("--image-2", default="image_2_rgb.png", help="Training right_front image filename.")
    parser.add_argument("--prompt", default="pick green block on yellow block")
    parser.add_argument(
        "--state",
        nargs=6,
        type=float,
        default=None,
        help="Override state. In episode dirs, omit this to use each obs_*/state.json.",
    )
    parser.add_argument("--train-resize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-start-index", type=int, default=0)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--hold-hz", type=float, default=10.0)
    parser.add_argument(
        "--hold-last-during-gaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep resending the last motor target while waiting/sleeping between chunks.",
    )
    parser.add_argument(
        "--log-torque-enable",
        action="store_true",
        help="Read and print Torque_Enable after connect and around chunk intervals.",
    )
    parser.add_argument(
        "--chunk-interval-s",
        type=float,
        default=2.0,
        help="Sleep after each executed/inferred chunk before requesting the next chunk.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of inference chunks to request.")
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the robot.")
    return parser.parse_args()


def import_lerobot(lerobot_repo: str):
    sys.path.insert(0, lerobot_repo)
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    return SOFollowerRobotConfig, SOFollower


def build_robot(args):
    SOFollowerRobotConfig, SOFollower = import_lerobot(args.lerobot_repo)
    config = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.robot_port,
        calibration_dir=Path(args.calibration_dir),
        cameras={},
    )
    return SOFollower(config)


def load_image_for_server(path: Path, train_resize: bool) -> np.ndarray:
    if train_resize:
        # Match the training-side visual preprocessing before handing the image
        # to the server's BGR->RGB decode path.
        image = Image.open(path).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        rgb = np.asarray(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return bgr


def discover_episode_obs(frame_dir: Path) -> list[Path]:
    return sorted(path for path in frame_dir.glob("obs_*") if path.is_dir())


def load_state(obs_dir: Path, args) -> np.ndarray:
    if args.state is not None:
        return np.asarray(args.state, dtype=np.float32)

    state_path = obs_dir / "state.json"
    if state_path.exists():
        with state_path.open("r") as f:
            state_data = json.load(f)
        if isinstance(state_data, dict):
            state_data = state_data["state"]
        return np.asarray(state_data[:6], dtype=np.float32)

    return np.asarray(DEFAULT_TRAIN_STATE, dtype=np.float32)


def build_request_for_obs(obs_dir: Path, args) -> tuple[dict, np.ndarray]:
    image_1 = load_image_for_server(obs_dir / args.image_1, args.train_resize)
    image_2 = load_image_for_server(obs_dir / args.image_2, args.train_resize)
    dummy = np.zeros_like(image_1, dtype=np.uint8)

    state = load_state(obs_dir, args)
    pad_dim = 24 - state.shape[0]
    if pad_dim < 0:
        raise ValueError(f"state dim {state.shape[0]} exceeds native max_state_dim=24")

    request = {
            "image": [image_1.tolist(), image_2.tolist(), dummy.tolist()],
            "image_mask": [1, 1, 0],
            "state": np.pad(state, (0, pad_dim), constant_values=0).astype(float).tolist(),
            "action_mask": [[1] * state.shape[0] + [0] * pad_dim],
            "prompt": args.prompt,
        }
    return request, state


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


async def run(args):
    frame_dir = Path(args.frame_dir)
    episode_obs = discover_episode_obs(frame_dir)
    if episode_obs:
        print(f"episode mode: found {len(episode_obs)} observations under {frame_dir}", flush=True)
    else:
        episode_obs = [frame_dir]
        print(f"single-observation mode: {frame_dir}", flush=True)

    execute = args.execute and not args.no_action
    robot = build_robot(args) if execute else None
    hold = HoldLastCommand(robot, args.hold_hz, args.hold_last_during_gaps and execute) if robot else None
    if robot is not None:
        robot.connect()
    if hold is not None:
        hold.start()
        if hold.enabled:
            print(f"[hold-last] enabled at {args.hold_hz:.1f} Hz during chunk gaps", flush=True)
        if args.log_torque_enable:
            print(f"[torque_enable after connect] {hold.read_torque_enable()}", flush=True)
    try:
        async with websockets.connect(args.server_uri, max_size=100_000_000) as ws:
            print(f"connected to {args.server_uri}", flush=True)
            for request_i in range(args.repeat):
                if request_i >= len(episode_obs):
                    print(f"[episode done] requested repeat={args.repeat}, available obs={len(episode_obs)}", flush=True)
                    break
                obs_dir = episode_obs[request_i]
                request, state = build_request_for_obs(obs_dir, args)
                print(f"[request {request_i}] obs_dir={obs_dir}", flush=True)
                print(f"[request {request_i}] state={np.round(state, 6).tolist()}", flush=True)
                await ws.send(json.dumps(request))
                response = await ws.recv()
                action_chunk = np.asarray(json.loads(response), dtype=np.float32)
                print(f"[request {request_i}] action_chunk shape={action_chunk.shape}", flush=True)
                start = args.action_start_index
                end = min(start + args.action_steps, action_chunk.shape[0])
                if start < 0 or start >= action_chunk.shape[0]:
                    raise ValueError(f"action_start_index={start} outside chunk length {action_chunk.shape[0]}")
                for action_i, action_row in enumerate(action_chunk[start:end], start=start):
                    cmd = action_dict(action_row)
                    if execute:
                        hold.send_action(cmd)
                        print(f"[sent dataset chunk_idx={action_i}] {cmd}", flush=True)
                        time.sleep(1.0 / args.control_hz)
                    else:
                        print(f"[no-action dataset chunk_idx={action_i}] {cmd}", flush=True)
                if hold is not None and args.log_torque_enable:
                    print(f"[torque_enable after chunk request={request_i}] {hold.read_torque_enable()}", flush=True)
                if request_i + 1 < args.repeat and args.chunk_interval_s > 0:
                    print(f"[chunk interval] sleep {args.chunk_interval_s:.3f}s", flush=True)
                    time.sleep(args.chunk_interval_s)
                    if hold is not None and args.log_torque_enable:
                        print(
                            f"[torque_enable after interval request={request_i}] {hold.read_torque_enable()}",
                            flush=True,
                        )
    finally:
        if hold is not None:
            hold.stop()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
