# -*- coding:utf-8 -*-
"""
Record final greedy paths for SARSA, Q-Learning, and Dyna-Q on CliffWalking-v0
and save videos into ./videos using gym video wrappers (RecordVideo if available,
falling back to Monitor). If wrappers are unavailable to capture RGB frames from
this env, a lightweight matplotlib animation fallback is used to produce .mp4.

Expected inputs: the evaluator produced pickles: sarsa_data.pkl, qlearning_data.pkl, dyna_q_data.pkl
containing a dense Q_table matrix.
"""
import os
import pickle
import numpy as np
import gym

from typing import Tuple, Optional


ALGOS = (
	("sarsa", "Sarsa"),
	("qlearning", "Q-Learning"),
	("dyna_q", "Dyna-Q"),
)


def load_q_table(name: str) -> Optional[np.ndarray]:
	# Evaluator saves pickles in the current directory as "{name}_data.pkl"
	pkl = f"data/{name}_data.pkl"
	if not os.path.exists(pkl):
		print(f"[WARN] {pkl} not found. Run evaluate_cliffwalk.py first.")
		return None
	with open(pkl, "rb") as f:
		data = pickle.load(f)
	Q = np.asarray(data.get("Q_table", None))
	if Q is None or Q.ndim != 2:
		print(f"[WARN] Q_table missing or invalid in {pkl}")
		return None
	return Q


def greedy_action(Q: np.ndarray, s: int) -> int:
	"""Select greedy action with random tie-breaking (to match original behavior)."""
	row = Q[int(s)] if int(s) < Q.shape[0] else np.zeros(Q.shape[1], dtype=np.float32)
	m = np.max(row)
	idx = np.where(row == m)[0]
	return int(np.random.choice(idx))


def make_env_with_video(video_dir: str, name_prefix: str):
	"""Create CliffWalking env and wrap with a video recorder.

	Tries RecordVideo (newer gym/gymnasium) first; falls back to Monitor (older gym).
	"""
	os.makedirs(video_dir, exist_ok=True)

	# Try with render_mode="rgb_array" (newer API); fall back to default if it fails
	env = None
	try:
		env = gym.make("CliffWalking-v0", render_mode="rgb_array")
	except TypeError:
		env = gym.make("CliffWalking-v0")

	# Prefer RecordVideo if present
	if hasattr(gym.wrappers, "RecordVideo"):
		print(f"[INFO] Using RecordVideo wrapper for {name_prefix}")
		try:
			env = gym.wrappers.RecordVideo(
				env,
				video_dir,
				episode_trigger=lambda ep: True,
				name_prefix=name_prefix,
			)
			return env
		except Exception as e:
			print(f"[WARN] RecordVideo failed ({e}), trying Monitor fallback…")

	# Fallback: Monitor (older gym)
	if hasattr(gym.wrappers, "Monitor"):
		print(f"[INFO] Using Monitor wrapper for {name_prefix}")
		try:
			env = gym.wrappers.Monitor(
				env,
				video_dir,
				video_callable=lambda ep: True,
				force=True,
			)
			return env
		except Exception as e:
			print(f"[WARN] Monitor fallback failed ({e}).")

	return env  # Unwrapped (we will then use the matplotlib fallback)


def run_episode(env, Q: np.ndarray, limit: int = 100) -> Tuple[float, int]:
	"""Run a single greedy episode using Q and return (total_reward, steps)."""
	# Handle old-vs-new reset signatures
	out = env.reset()
	s = out[0] if isinstance(out, tuple) else out
	total = 0.0
	steps = 0
	for _ in range(limit):
		a = greedy_action(Q, s)
		step_out = env.step(a)
		if len(step_out) == 5:  # new API
			s2, r, terminated, truncated, _ = step_out
			done = terminated or truncated
		else:  # old API
			s2, r, done, _ = step_out
		total += float(r)
		steps += 1
		s = s2
		if done:
			break
	return total, steps


# ---- Fallback matplotlib animator (only used if neither wrapper works) ----
def save_path_animation(Q: np.ndarray, out_mp4: str, grid=(4, 12), limit: int = 60):
	"""Draw a simple grid animation of the greedy path and save as mp4.
	This ensures we can still provide a video even if wrappers can't record frames.
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.animation import FFMpegWriter

	rows, cols = grid

	fig, ax = plt.subplots(figsize=(12, 4.2))
	# grid
	for r in range(rows + 1):
		ax.plot([0, cols], [r, r], "k-", linewidth=1)
	for c in range(cols + 1):
		ax.plot([c, c], [0, rows], "k-", linewidth=1)
	# start/cliff/goal
	ax.add_patch(plt.Rectangle((0, rows - 1), 1, 1, fc="green", alpha=0.3))
	for c in range(1, cols - 1):
		ax.add_patch(plt.Rectangle((c, rows - 1), 1, 1, fc="red", alpha=0.25))
	ax.add_patch(plt.Rectangle((cols - 1, rows - 1), 1, 1, fc="gold", alpha=0.3))
	ax.set_xlim(0, cols)
	ax.set_ylim(0, rows)
	ax.invert_yaxis()
	ax.set_aspect("equal")
	ax.set_xticks([])
	ax.set_yticks([])
	dot, = ax.plot([], [], "bo", ms=10)

	# Derive path using the actual env transitions to be accurate
	env_tmp = gym.make("CliffWalking-v0")
	out = env_tmp.reset()
	s = out[0] if isinstance(out, tuple) else out
	path = [int(s)]
	for _ in range(limit):
		a = greedy_action(Q, s)
		step_out = env_tmp.step(a)
		if len(step_out) == 5:
			s, r, terminated, truncated, _ = step_out
			done = terminated or truncated
		else:
			s, r, done, _ = step_out
		path.append(int(s))
		if done:
			break
	env_tmp.close()

	def s2xy(sid: int):
		r, c = divmod(sid, cols)
		return c + 0.5, r + 0.5

	frames_xy = [s2xy(p) for p in path]

	writer = FFMpegWriter(fps=10, metadata=dict(artist="RL"))
	with writer.saving(fig, out_mp4, dpi=150):
		for (x, y) in frames_xy:
			dot.set_data([x], [y])
			writer.grab_frame()
	plt.close(fig)


def record_algo(name: str, pretty: str, video_dir: str = "videos"):
	Q = load_q_table(name)
	if Q is None:
		return

	# Try wrappers first
	env = make_env_with_video(video_dir, name_prefix=name)
	used_wrapper = (hasattr(env, "episode_id") or hasattr(gym.wrappers, "RecordVideo") or hasattr(gym.wrappers, "Monitor"))
	try:
		total, steps = run_episode(env, Q)
		print(f"[{pretty}] episode finished: reward={total:.1f}, steps={steps}")
	finally:
		try:
			env.close()
		except Exception:
			pass

	# If no video files appeared (e.g., wrappers couldn't capture), create fallback mp4
	# Heuristic: if directory has no file with prefix, generate one.
	files = [f for f in os.listdir(video_dir) if f.startswith(name)] if os.path.isdir(video_dir) else []
	if len(files) == 0:
		out_mp4 = os.path.join(video_dir, f"{name}_path.mp4")
		print(f"[INFO] No wrapper video found; saving fallback animation to {out_mp4}")
		os.makedirs(video_dir, exist_ok=True)
		save_path_animation(Q, out_mp4)


if __name__ == "__main__":
	os.makedirs("videos", exist_ok=True)
	for key, pretty in ALGOS:
		record_algo(key, pretty, video_dir="videos")
	print("Done. Videos saved under ./videos")

