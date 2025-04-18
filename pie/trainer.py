import inspect
import os
import uuid
import logging
import time
import collections
import random
import tempfile
import warnings
from typing import ClassVar

import tqdm

import torch
from torch import optim
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
import pytorch_optimizer as ext_optims

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


def get_batch_task(tasks, **kwargs):
    tnames, target = [], None
    for task in tasks:
        tnames.append(task['name'])
        if task.get('target'):
            target = task['name']
    return sample_task(target, tnames, **kwargs)


def sample_task(target, tasks, factor=2):
    # sample target task factor times as many as any other task
    aux = (1 / (len(tasks) - 1)) / factor if len(tasks) > 1 else 0
    trg = 1 - (aux * (len(tasks) - 1))
    weights = [aux if task != target else trg for task in tasks]
    return random.choices(tasks, weights)[0]


def get_target_task(settings):
    for task in settings.tasks:
        if task.get('target'):
            return task['name']
    raise ValueError("No target task?")


class EarlyStopException(Exception):
    def __init__(self, task, loss, state_dict):
        self.task = task
        self.loss = loss
        self.best_state_dict = state_dict


class TaskScheduler(object):
    """
    Track scores
    """
    def __init__(self, settings):
        tasks = {}
        # preprocess tasks
        for task in settings.tasks:
            # add schedule and target
            tasks[task['name']] = task.get('schedule', {})
            tasks[task['name']]['target'] = task.get('target', False)
            # add task data for lm loss
            if settings.include_lm:
                tasks['lm_fwd'] = dict(settings.lm_schedule)
                tasks['lm_bwd'] = dict(settings.lm_schedule)

        for task, tdata in tasks.items():
            # set step counter
            tdata['steps'] = 0
            # set default task mode
            tdata['mode'] = tdata.get('mode', 'max')
            # set initial weight
            tdata['weight'] = tdata.get('weight', 1.0)
            # set initial best
            tdata['best'] = -float('inf') if tdata['mode'] == 'max' else float('inf')

        self.tasks = tasks

        # task schedule
        self.patience = settings.patience
        self.factor = settings.factor
        self.threshold = settings.threshold
        self.min_weight = settings.min_weight
        self.fid = os.path.join(tempfile.gettempdir(), str(uuid.uuid1()))
        self.best_epoch = 0

    def __repr__(self):
        # task scheduler
        output = (
            '<TaskScheduler patience="{}" factor="{}" ' +
            'threshold="{}" min_weight="{}">').format(
                self.patience, self.factor, self.threshold, self.min_weight)

        for task, values in self.tasks.items():
            output += '\n    <Task name="{}" '.format(task)
            output += ' '.join(
                '{}="{}"'.format(key, val) for key, val in values.items())
            output += '/>'
        output += '\n</TaskScheduler>'

        return output

    def is_best(self, task, value):
        threshold = self.tasks[task].get('threshold', self.threshold)
        mode = self.tasks[task]['mode']
        if mode.lower() == 'max':
            return value > (self.tasks[task]['best'] + threshold)
        elif mode.lower() == 'min':
            return value < (self.tasks[task]['best'] - threshold)
        else:
            raise ValueError("Wrong mode value [{}] for task: {}".format(mode, task))

    def step(self, scores, model, epoch):
        """
        Advance schedule step based on dev scores
        """
        for task, score in scores.items():
            if task not in self.tasks:
                # ignore
                continue

            is_target = self.tasks[task].get('target', False)

            # check if we improve
            if self.is_best(task, score):
                self.tasks[task]['best'] = score
                self.tasks[task]['steps'] = 0
                if is_target:
                    # serialize model params
                    torch.save(model.state_dict(), self.fid)
                    self.best_epoch = epoch
            else:
                self.tasks[task]['steps'] += 1

            # check if we need to stop globally or downweight a task loss
            patience = self.tasks[task].get('patience', self.patience)
            if self.tasks[task]['steps'] >= patience:
                # maybe stop entire training
                if is_target:
                    state_dict = torch.load(self.fid)
                    os.remove(self.fid)
                    raise EarlyStopException(task, self.tasks[task]['best'], state_dict)
                # update task weight
                else:
                    factor = self.tasks[task].get('factor', self.factor)
                    new_weight = self.tasks[task]['weight'] * factor
                    min_weight = self.tasks[task].get('min_weight', self.min_weight)
                    self.tasks[task]['weight'] = max(new_weight, min_weight)

    def get_weights(self):
        return {task: self.tasks[task]['weight'] for task in self.tasks}


class LRScheduler(object):
    def __init__(self, optimizer,
                 lr_scheduler='ReduceLROnPlateau',
                 delay=0, **kwargs):

        self.nb_steps: int = 0
        self.delay: int = delay
        lr_scheduler_cls = getattr(optim.lr_scheduler, lr_scheduler)
        params = inspect.signature(lr_scheduler_cls).parameters
        self.lr_scheduler = lr_scheduler_cls(
            optimizer,
            # pick kwargs that fit the selected scheduler
            **{kwarg: val for kwarg, val in kwargs.items() if kwarg in params})

    def step(self, score):
        self.nb_steps += 1

        # apply the step() method of the lr_scheduler when delay is reached
        if self.delay and self.nb_steps <= self.delay:
            return

        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(score)
        else:
            self.lr_scheduler.step()

    def __repr__(self):
        res = '<LrScheduler="{}" lr="{:g}" delay="{}" steps="{}"'.format(
            self.lr_scheduler.__class__.__name__,
            self.lr_scheduler.optimizer.param_groups[0]['lr'],
            self.delay,
            self.nb_steps)
        for key in dir(self.lr_scheduler):
            val = getattr(self.lr_scheduler, key)
            if (not key.startswith('__')) and isinstance(val, (str, float, int)):
                res += ' {}="{}"'.format(key, val)
        res += '/>'
        return res


class Trainer(object):
    """
    Trainer

    Settings
    ========
    optim
    lr (lr_factor, lr_patience, min_lr)
    clip_norm
    weights
    report_freq
    checks_per_epoch
    """
    def __init__(self, settings, model, dataset, num_instances):
        self.target_task = get_target_task(settings)
        self.verbose = settings.verbose
        self.dataset = dataset
        self.model = model
        self.optimizer = self.get_optimizer(settings.optimizer)(
            model.parameters(), lr=settings.lr, **settings.optimizer_params)
        self.clip_norm = settings.clip_norm

        self.report_freq = settings.report_freq
        self.num_batches = num_instances // dataset.batch_size
        if settings.checks_per_epoch == 1:
            self.check_freq = self.num_batches - 1  # check after last batch
        elif settings.checks_per_epoch > self.num_batches:
            self.check_freq = 1  # check after each batch
        elif settings.checks_per_epoch > 1:
            self.check_freq = self.num_batches // settings.checks_per_epoch  # check just
        else:
            self.check_freq = 0  # no checks

        self.current_epoch = 0

        self.task_scheduler = TaskScheduler(settings)
        self.lr_scheduler = LRScheduler(
            self.optimizer,
            lr_scheduler=settings.lr_scheduler,
            delay=settings.lr_scheduler_delay,
            **settings.lr_scheduler_params)

        self.noise_strategies = settings.noise_strategies or {}
        self.noise_strategies = {
            strategy_name: strategy
            for strategy_name, strategy in self.noise_strategies.items()
            if isinstance(strategy, dict) and strategy.get("apply") is True
        }

        if settings.verbose:
            print()
            print("Evaluation check every {}/{} batches".format(
                self.check_freq, self.num_batches))
            print()
            print("::: Task schedules :::")
            print()
            print(self.task_scheduler)
            print()
            print("::: LR schedule :::")
            print()
            print(self.lr_scheduler)
            print()

    @staticmethod
    def get_optimizer(optimizer_name: str) -> ClassVar[Optimizer]:
        """ Allows for getting new optimizers from the torch-optimizer library without
        breaking previous behaviour
        :param optimizer_name: Optimizer Name, eg. Adam, SGD, Ranger
        :return: Optimizer class
        """
        if hasattr(optim, optimizer_name):
            return getattr(optim, optimizer_name)
        elif hasattr(ext_optims, optimizer_name):
            return getattr(ext_optims, optimizer_name)
        else:
            raise ValueError("Unknown optimizer %s " % optimizer_name)

    def weight_loss(self, loss):
        """
        Apply weights to losses and return a single loss number
        """
        weights = self.task_scheduler.get_weights()

        return sum(weights.get(k, 1) * loss[k] for k in loss)

    def evaluate(self, dataset):
        """
        Evaluate objective on held-out data
        """
        total_losses, total_batches = collections.defaultdict(float), 0

        # get all tasks
        tasks = list(self.model.tasks)

        for batch in tqdm.tqdm(dataset.batch_generator()):
            total_batches += 1
            for k, v in self.model.loss(batch, *tasks).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return dict(total_losses)

    def run_check(self, devset):
        """
        Monitor dev loss and eventually early-stop training
        """
        print()
        print("Evaluating model on dev set...")
        print()

        self.model.eval()

        stored_scores = {}

        with torch.no_grad():
            dev_loss = self.evaluate(devset)
            print()
            print("::: Dev losses :::")
            print()
            print('\n'.join('{}: {:.4f}'.format(k, v) for k, v in dev_loss.items()))
            print()
            summary = self.model.evaluate(devset, self.dataset)
            for task_name, scorer in summary.items():
                stored_scores[task_name] = scorer.get_scores()
                scorer.print_summary(scores=stored_scores[task_name])

        self.model.train()
        dev_scores = {}
        for task, scored in stored_scores.items():
            dev_scores[task] = scored['all'][
                self.task_scheduler.tasks[task].get("evaluation", "accuracy")
            ]

        # add lm scores
        if 'lm_fwd' in dev_loss or 'lm_bwd' in dev_loss:
            dev_scores['lm_fwd'] = dev_loss['lm_fwd']
            dev_scores['lm_bwd'] = dev_loss['lm_bwd']

        self.task_scheduler.step(dev_scores, self.model, self.current_epoch)
        self.lr_scheduler.step(dev_scores[self.target_task])

        if self.verbose:
            print(self.task_scheduler)
            print()
            print(self.lr_scheduler)
            print()

        return dev_scores

    def train_epoch(self, devset):
        rep_loss = collections.defaultdict(float)
        rep_batches = collections.defaultdict(int)
        rep_items, rep_start = 0, time.time()
        scores = None

        for b, batch in enumerate(self.dataset.batch_generator(apply_noise=self.noise_strategies)):
            # get loss
            loss = self.model.loss(batch, get_batch_task(self.model.tasks.values()))

            if not loss:
                raise ValueError("Got empty loss, no tasks defined?")

            # optimize
            self.optimizer.zero_grad()
            self.weight_loss(loss).backward()
            if self.clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # accumulate
            rep_items += type(self.dataset).get_nelement(batch)
            for k, v in loss.items():
                rep_batches[k] += 1
                rep_loss[k] += v.item()

            # report
            if b > 0 and b % self.report_freq == 0:
                rep = ""
                for t in sorted(rep_loss):
                    rep += '{}:{:.4f}  '.format(t, rep_loss[t] / rep_batches[t])
                logging.info("Batch [{}/{}] || {} || {:.0f} w/s".format(
                    b, self.num_batches, rep, rep_items / (time.time() - rep_start)))
                rep_loss = collections.defaultdict(float)
                rep_batches = collections.defaultdict(int)
                rep_items, rep_start = 0, time.time()

            if self.check_freq > 0 and b > 0 and b % self.check_freq == 0:
                if devset is not None:
                    rep_start = time.time()
                    scores = self.run_check(devset)
                    logging.info("Evaluation time: {:.0f} sec".format(
                        time.time() - rep_start))
                    rep_start = time.time()

        return scores

    def train_epochs(self, epochs, devset=None):
        """
        Train the model for a number of epochs
        """
        start = time.time()
        scores = None

        try:
            for epoch in range(self.current_epoch + 1, epochs + 1):
                # train epoch
                self.current_epoch += 1
                epoch_start = time.time()
                logging.info("Starting epoch [{}]".format(epoch))
                self.train_epoch(devset)
                epoch_total = time.time() - epoch_start
                logging.info("Finished epoch [{}] in [{:.0f}] secs".format(
                    epoch, epoch_total))

        except EarlyStopException as e:
            logging.info("Early stopping training: "
                         "task [{}] with best score {:.4f}".format(e.task, e.loss))
            
            print(f"Loading best model (epoch {self.task_scheduler.best_epoch}) for target task {self.target_task}")
            self.model.last_state_dict = self.model.state_dict()
            self.model.load_state_dict(e.best_state_dict)
            scores = {e.task: e.loss}
        else:
            # Load best model
            print(f"Loading best model (epoch {self.task_scheduler.best_epoch}) for target task {self.target_task}")
            self.model.last_state_dict = self.model.state_dict()
            if os.path.exists(self.task_scheduler.fid):
                best_state_dict = torch.load(self.task_scheduler.fid)
                self.model.load_state_dict(best_state_dict)
            else:
                warnings.warn(
                    f"Temp path with best model weights doesn't exist ({self.task_scheduler.fid}). "
                    "Maybe the model never improved over training ?"
                )
            scores = {self.target_task: self.task_scheduler.tasks[self.target_task]['best']}

        logging.info("Finished training in [{:.0f}] secs".format(time.time() - start))

        # will be None if no dev test was provided
        return scores
