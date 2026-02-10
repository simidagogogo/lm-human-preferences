import concurrent.futures
import os
import subprocess
from functools import partial

import cloudpickle
import fire


def launch(name, f, *, namespace='safety', mode='local', mpi=1) -> None:
    print(f"debug. enter launch(). name: {name}, f: {f}")
    """
    debug. enter launch(). 
    name: testdesc-2601022114, 
    f: functools.partial(<function train at 0x7ff46f497050>, 
        HParams(run=RunHParams(seed=1, log_interval=10, save_interval=50, save_dir='/tmp/save/train_reward/testdesc-2601022114'), 
                task=TaskHParams(query_length=64, query_dataset='books', query_prefix='', query_suffix='', start_text='.', end_text='.', response_length=24, truncate_token=13, truncate_after=16, penalty_reward_value=-1, policy=PolicyHParams(temperature=0.7, initial_model='124M')), 
                labels=LabelHParams(type='best_of_4', num_train=4992, source='https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json'), 
                batch_size=32, 
                lr=5e-05, 
                rollout_batch_size=512, 
                normalize_samples=256, 
                debug_normalize=0, 
                normalize_before=True, 
                normalize_after=True)
    )
    """
    if mode == 'local':
        with open('/tmp/pickle_fn', 'wb') as file:
            cloudpickle.dump(f, file)

        # If mpi=1, run directly without mpiexec (for systems without MPI installed)
        if mpi == 1:
            import pickle
            with open('/tmp/pickle_fn', 'rb') as file:
                fn = pickle.loads(file.read())
            fn()
        else:
            subprocess.check_call(['mpiexec', '-n', str(mpi), 'python', '-c', 'import sys; import pickle; pickle.loads(open("/tmp/pickle_fn", "rb").read())()'])
        return
    raise Exception('Other modes unimplemented!')


def parallel(jobs, mode):
    print(f"debug. inter parallel({jobs}, {mode})")
    """
    # debug. inter parallel(
	[functools.partial(
		<function launch at 0x7ff48dbf09e0>, 
		'testdesc-2601022114', 
		functools.partial(
			<function train at 0x7ff46f497050>, 
			HParams(run=RunHParams(seed=1, log_interval=10, save_interval=50, save_dir='/tmp/save/train_reward/testdesc-2601022114'), 
			task=TaskHParams(query_length=64, query_dataset='books', query_prefix='', query_suffix='', start_text='.', end_text='.', response_length=24, truncate_token=13, truncate_after=16, penalty_reward_value=-1, policy=PolicyHParams(temperature=0.7, initial_model='124M')), 
			labels=LabelHParams(type='best_of_4', num_train=4992, source='https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json'), 
			batch_size=32, 
			lr=5e-05, 
			rollout_batch_size=512, 
			normalize_samples=256, 
			debug_normalize=0, 
			normalize_before=True, 
			normalize_after=True)
		), 
		mpi=1, 
		mode='local')
	], 
	local
    )
    """
    if mode == 'local':
        assert len(jobs) == 1, "Cannot run jobs in parallel locally"
        for job in jobs:
            job()
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(job) for job in jobs]
            for f in futures:
                f.result()


def launch_trials(name, fn, trials, hparam_class, *, extra_hparams=None, dry_run=False, mpi=1, mode='local', save_dir=None):
    print(f"debug. enter launch_trials(). name: {name}, fn: {fn}, trials: {trials}, hparam_class: {hparam_class}, dry_run: {dry_run}, mpi: {mpi}, mode: {mode}")
    """
    debug. enter launch_trials(). 
    name: testdesc-2601022114, 
    fn: <function train at 0x7ff46f497050>, 
    trials: [(('task.query_length', 64, {}), ('task.query_dataset', 'books', {}), ('task.response_length', 24, {}), ('task.start_text', '.', {}), ('task.end_text', '.', {}), ('task.truncate_token', 13, {}), ('task.truncate_after', 16, {}), ('task.policy.temperature', 0.7, {}), ('task.policy.initial_model', '124M', {}), ('labels.type', 'best_of_4', {}), ('normalize_after', True, {}), ('normalize_before', True, {}), ('normalize_samples', 256, {}), ('batch_size', 32, {}), ('lr', 5e-05, {}), ('rollout_batch_size', 512, {}), ('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json', {}), ('labels.num_train', 4992, {}), ('run.seed', 1, {}))], 
    hparam_class: <class 'lm_human_preferences.train_reward.HParams'>, 
    dry_run: False, 
    mpi: 1, 
    mode: local    
    """
    jobs = []
    for trial in trials:
        descriptors = []
        kwargs = {}
        for k, v, s in trial:
            if k is not None:
                if k in kwargs:
                    print(f"WARNING: overriding key {k} from {kwargs[k]} to {v}")
                kwargs[k] = v
            if s.get('descriptor'):
                descriptors.append(str(s['descriptor']))
        hparams = hparam_class()
        hparams.override_from_dict(kwargs)
        if extra_hparams:
            hparams.override_from_str_dict(extra_hparams)
        job_name = (name + "/" + "-".join(descriptors)).rstrip("/")
        hparams.validate()
        if dry_run:
            print(f"{job_name}: {kwargs}")
        else:
            if save_dir:
                hparams.run.save_dir = os.path.join(save_dir, job_name)
            trial_fn = partial(fn, hparams)
            jobs.append(partial(launch, job_name, trial_fn, mpi=mpi, mode=mode))

    parallel(jobs, mode=mode)


def main(commands_dict):
    """
    Similar to fire.Fire, but with support for multiple commands without having a class.
    """
    class _Commands:
        def __init__(self):
            for name, cmd in commands_dict.items():
                setattr(self, name, cmd)
    fire.Fire(_Commands)
