"""Independent, camera-free simulator processes for batched state rollouts."""

import logging
import multiprocessing as mp
import os
import time
import traceback

import numpy as np

from v3.state import state_vector


def _make_simulator(task_config, *, horizon):
    # Construct MuJoCo inside the spawned process, never fork a live simulator
    # or CUDA context from the training process.
    from v3.task import StateGreenLiftSimulator
    return StateGreenLiftSimulator(task_config, horizon=horizon)


def _worker(connection, task_config, max_steps, seed, seed_stride, simulator_factory):
    simulator = None
    try:
        logging.getLogger('robosuite_logs').setLevel(logging.WARNING)
        np.random.seed(seed)
        simulator = simulator_factory(task_config, horizon=max_steps + 1)

        def reset():
            np.random.seed(seed)
            simulator.reset()
            low, _ = simulator.action_spec
            return simulator.step(np.zeros_like(low))

        observation = reset()
        episode_steps = 0
        episode_return = 0.0
        connection.send(('ok', state_vector(observation)))
        while True:
            command, action = connection.recv()
            if command == 'close':
                break
            if command != 'step':
                raise ValueError(f'Unknown worker command: {command}')
            observation = simulator.step(action)
            reward = float(observation['task_reward'])
            success = bool(observation['task_complete'])
            episode_steps += 1
            episode_return += reward
            done = success or episode_steps >= max_steps
            episode = None
            if done:
                episode = {
                    'seed': seed, 'return': episode_return, 'length': episode_steps,
                    'success': success, 'task_metrics': observation['task_metrics'],
                }
                seed += seed_stride
                observation = reset()
                episode_steps = 0
                episode_return = 0.0
            # For done transitions state is the next episode's initial state.
            # The done mask blocks value/advantage propagation across that reset.
            connection.send(('ok', (state_vector(observation), reward, done, episode)))
    except EOFError:
        pass  # Parent closed the pipe.
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if simulator is not None:
                simulator.close()
        finally:
            connection.close()


class ParallelSimulators:
    """Synchronous batched stepping; each simulator owns a separate process/RNG."""

    def __init__(self, num_envs, task_config, max_steps, training_seed, *,
                 simulator_factory=_make_simulator, timeout=120):
        self.connections = []
        self.processes = []
        self.timeout = timeout
        context = mp.get_context('spawn')
        # Prevent each worker from starting a full BLAS/OpenMP thread pool.
        thread_vars = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
        previous = {key: os.environ.get(key) for key in thread_vars}
        try:
            for key in thread_vars:
                os.environ[key] = '1'
            for env_id in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, task_config, max_steps, training_seed + env_id,
                          num_envs, simulator_factory),
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.states = np.stack([self._receive(i) for i in range(num_envs)])
        except BaseException:
            self.close()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _receive(self, env_id):
        connection = self.connections[env_id]
        if not connection.poll(self.timeout):
            raise RuntimeError(f'Simulator {env_id} timed out after {self.timeout}s')
        try:
            status, payload = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f'Simulator {env_id} exited unexpectedly') from error
        if status != 'ok':
            raise RuntimeError(f'Simulator {env_id} failed:\n{payload}')
        return payload

    def step(self, env_ids, actions):
        if len(env_ids) != len(actions) or len(set(env_ids)) != len(env_ids):
            raise ValueError('Expected one action per distinct simulator')
        for env_id, action in zip(env_ids, actions):
            try:
                self.connections[env_id].send(('step', action))
            except (BrokenPipeError, EOFError, OSError) as error:
                raise RuntimeError(f'Simulator {env_id} is unavailable') from error
        results = [self._receive(env_id) for env_id in env_ids]
        for env_id, result in zip(env_ids, results):
            self.states[env_id] = result[0]
        return results

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        deadline = time.monotonic() + 5
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
