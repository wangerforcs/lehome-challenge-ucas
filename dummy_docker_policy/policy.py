"""
LeHome Challenge — OpenPI Policy (Docker).

Loads an openpi checkpoint (pi05_lehome_top_short) and serves it via HTTP.
The server (server.py) handles all HTTP plumbing — this file only implements
the policy logic.
"""

import base64
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class BasePolicyServer:
    """
    Base class for LeHome policy servers.

    Handles all HTTP plumbing. Subclass and override:
        reset()              — called at the start of each episode
        infer(observation)   — called to get the next action chunk

    Observation dict contains:
        "observation.state"           — np.ndarray, shape (12,), float32 (joint angles)
        "observation.images.top_rgb"  — np.ndarray, shape (H, W, 3), uint8
        "observation.images.left_rgb" — np.ndarray, shape (H, W, 3), uint8
        "observation.images.right_rgb"— np.ndarray, shape (H, W, 3), uint8
        "observation.top_depth"       — np.ndarray, shape (H, W), uint16 (depth in mm)
        "action"                      — np.ndarray, shape (12,), float32 (previous action)
    """

    def reset(self) -> None:
        """Called at the start of each episode. Override to clear buffers, etc."""
        pass

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Return a list of actions (action chunk).

        Args:
            observation: Dict of numpy arrays (state, images, previous action).

        Returns:
            List of np.ndarray actions, each shape (action_dim,).
            Return 1 action for per-step control, or N for action chunking.
        """
        raise NotImplementedError("Subclass must implement infer()")

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the HTTP server."""
        policy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                request = json.loads(body)

                if self.path == "/reset":
                    policy.reset()
                    response = {"status": "ok"}

                elif self.path == "/infer":
                    observation = _deserialize_observation(request)
                    actions = policy.infer(observation)
                    response = {"actions": [a.tolist() for a in actions]}

                else:
                    self.send_error(404, f"Unknown endpoint: {self.path}")
                    return

                body_bytes = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

            def log_message(self, format, *args):
                print(f"[{self.command}] {self.path}")

        server = HTTPServer((host, port), Handler)
        print(f"Policy server listening on {host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            server.server_close()


def _deserialize_observation(raw: dict) -> Dict[str, np.ndarray]:
    """Convert JSON observation to numpy arrays."""
    observation = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "base64" in value:
            buf = base64.b64decode(value["base64"])
            observation[key] = np.frombuffer(buf, dtype=value["dtype"]).reshape(value["shape"])
        elif isinstance(value, list):
            observation[key] = np.array(value, dtype=np.float32)
    return observation


class OpenPIPolicy(BasePolicyServer):
    def __init__(self, config_name: str, checkpoint_dir: str,
                 default_prompt: str = "fold the garment on the table",
                 replan_steps: int = 5, port: int = 8080):
        print(f"Loading openpi policy: config={config_name}, checkpoint={checkpoint_dir}")
        self.policy = _policy_config.create_trained_policy(
            _config.get_config(config_name),
            checkpoint_dir,
            default_prompt=default_prompt,
        )
        self.task_description = default_prompt
        self.replan_steps = replan_steps
        self.port = port
        self._action_chunk: List[np.ndarray] = []
        print("OpenPI policy loaded.")

    def reset(self):
        self._action_chunk = []

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        element = {
            "observation/top_image": observation["observation.images.top_rgb"],
            "observation/left_wrist_image": observation["observation.images.left_rgb"],
            "observation/right_wrist_image": observation["observation.images.right_rgb"],
            "observation/state": observation["observation.state"],
            "prompt": self.task_description,
        }
        actions = self.policy.infer(element)["actions"]
        chunk = [np.asarray(a, dtype=np.float32) for a in actions[: self.replan_steps]]
        return chunk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenPI LeHome Policy Server")
    parser.add_argument("--policy.config", dest="config_name", type=str, default="pi05_lehome_top_short",
                        help="Training config name (e.g. pi05_lehome_top_short)")
    parser.add_argument("--policy.dir", dest="checkpoint_dir", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--default-prompt", type=str, default="fold the garment on the table",
                        help="Default task prompt")
    parser.add_argument("--replan-steps", type=int, default=5,
                        help="Number of actions per chunk")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP server port")
    args = parser.parse_args()

    policy = OpenPIPolicy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        replan_steps=args.replan_steps,
        port=args.port,
    )
    policy.run(port=args.port)
