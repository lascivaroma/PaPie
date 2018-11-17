
import os
import uuid
import logging
import time
import collections

import tqdm

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


class EarlyStopException(Exception):
    def __init__(self, task, loss, state_dict):
        self.task = task
        self.loss = loss
        self.best_state_dict = state_dict


class TaskScheduler(object):
    """
    Track scores
    """
    def __init__(self, tasks, patience, factor, threshold, min_weight,
                 optimizer=None, lr_factor=1, lr_patience=100):
        for task, values in tasks.items():
            tasks[task] = {'steps': 0, **values}
            # set task mode
            if 'mode' not in tasks[task]:
                tasks[task]['mode'] = 'max'
            # set initial weight
            if 'weight' not in tasks[task]:
                tasks[task]['weight'] = 1.0
            # set initial best
            if tasks[task]['mode'] == 'max':
                tasks[task]['best'] = -float('inf')
            else:
                tasks[task]['best'] = float('inf')

        # lr schedule
        self.optimizer = optimizer
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_steps = 0

        # task schedule
        self.tasks = tasks
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.min_weight = min_weight
        self.fid = '/tmp/{}'.format(str(uuid.uuid1()))

    def get_lr(self):
        # assumes single param group
        return float(self.optimizer.param_groups[0]['lr'])

    def set_lr(self, new_lr):
        self.optimizer.param_groups[0]['lr'] = new_lr

    def __repr__(self):
        # task scheduler
        output = (
            '<TaskScheduler patience="{}" factor="{}" ' +
            'threshold="{}" min_weight="{}">').format(
                self.patience, self.factor, self.threshold, self.min_weight)

        for task, values in self.tasks.items():
            output += '\n    <Task name="{}" '.format(task)
            output += ' '.join('{}="{}"'.format(key, val) for key, val in values.items())
            output += '/>'
        output += '\n</TaskScheduler>'

        # lr scheduler
        if self.optimizer is not None:
            output += '\n'
            output += '<LrScheduler lr="{}" lr_steps="{}" lr_patience="{}"/>'.format(
                round(self.get_lr(), 5), self.lr_steps, self.lr_patience)

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

    def step(self, scores, model):
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
                    # lr schedule
                    self.lr_steps = 0
            else:
                self.tasks[task]['steps'] += 1
                # lr schedule
                if is_target:
                    self.lr_steps += 1

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

            # lr schedule
            if is_target and self.lr_steps >= self.lr_patience:
                self.set_lr(self.get_lr() * self.lr_factor)

    def get_weights(self):
        return {task: self.tasks[task]['weight'] for task in self.tasks}


class Trainer(object):
    """
    Trainer

    Settings
    ========
    optim
    lr
    clip_norm
    weights
    report_freq
    checks_per_epoch
    """
    def __init__(self, settings, model, dataset, num_instances):

        self.verbose = settings.verbose
        self.dataset = dataset
        self.model = model
        self.optimizer = getattr(optim, settings.optimizer)(
            model.parameters(), lr=settings.lr)
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

        tasks = {task['name']: task.get('schedule', {}) for task in settings.tasks}
        if settings.include_lm:
            tasks['fwd_lm'] = settings.lm_schedule
            tasks['bwd_lm'] = settings.lm_schedule
        self.task_scheduler = TaskScheduler(
            # task schedule
            tasks, settings.patience, settings.factor, settings.threshold,
            settings.min_weight,
            # lr schedule
            optimizer=self.optimizer,
            lr_factor=settings.lr_factor, lr_patience=settings.lr_patience)

        if settings.verbose:
            print()
            print("Evaluation check every {}/{} batches".format(
                self.check_freq, self.num_batches))
            print()
            print("::: Task schedules :::")
            print()
            print(self.task_scheduler)
            print()

    def weight_loss(self, loss):
        """
        Apply weights to losses and return a single loss number
        """
        weights = self.task_scheduler.get_weights()

        return sum(weights.get(k, 1) * loss[k] for k in loss)

    def evaluate(self, dataset, num_batches=None):
        """
        Evaluate objective on held-out data
        """
        total_losses, total_batches = collections.defaultdict(float), 0

        for batch in tqdm.tqdm(dataset.batch_generator()):
            total_batches += 1
            for k, v in self.model.loss(batch).items():
                total_losses[k] += v.item()

        for k, v in total_losses.items():
            total_losses[k] = v / total_batches

        return dict(total_losses)

    def monitor_batch(self, batch, items, start, nbatches, loss, sep=' '*3):
        """
        Print the report for monitoring
        """
        rep = sep.join('{}:{:.3f}'.format(k, v / nbatches) for k, v in loss.items())
        speed = items / (time.time() - start)
        formatter = "Batch [{}/{}] || {} || {:.0f} w/s"
        logging.info(formatter.format(batch, self.num_batches, rep, speed))

    def run_check(self, dev):
        """
        Monitor dev loss and eventually early-stop training
        """
        print()
        print("Evaluating model on dev set...")
        print()

        self.model.eval()

        with torch.no_grad():
            dev_loss = self.evaluate(dev)
            print()
            print("::: Dev losses :::")
            print()
            print('\n'.join('{}: {:.3f}'.format(k, v) for k, v in dev_loss.items()))
            print()
            summary = self.model.evaluate(dev)
            for task in summary.values():
                task.print_summary()

        self.model.train()
        dev_scores = {t: scorer.get_scores()['accuracy'] for t, scorer in summary.items()}
        # add lm scores
        if 'fwd_lm' in dev_loss or 'bwd_lm' in dev_loss:
            dev_scores['fwd_lm'] = dev_loss['fwd_lm']
            dev_scores['bwd_lm'] = dev_loss['bwd_lm']

        self.task_scheduler.step(dev_scores, self.model)

        if self.verbose:
            print(self.task_scheduler)
            print()

        return dev_scores

    def train_epoch(self, dev):
        rep_loss, rep_items, rep_batches = collections.defaultdict(float), 0, 0
        rep_start = time.time()
        scores = None

        for b, batch in enumerate(self.dataset.batch_generator()):
            # get loss
            loss = self.model.loss(batch)

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
            rep_batches += 1
            for k, v in loss.items():
                rep_loss[k] += v.item()

            # report
            if b > 0 and b % self.report_freq == 0:
                self.monitor_batch(b, rep_items, rep_start, rep_batches, rep_loss)
                rep_loss, rep_items, rep_batches = collections.defaultdict(float), 0, 0
                rep_start = time.time()

            if self.check_freq > 0 and b > 0 and b % self.check_freq == 0:
                if dev is not None:
                    scores = self.run_check(dev)
                    return scores

    def train_epochs(self, epochs, dev=None):
        """
        Train the model for a number of epochs
        """
        start = time.time()
        scores = None

        try:
            for e in range(1, epochs + 1):
                # train epoch
                epoch_start = time.time()
                logging.info("Starting epoch [{}]".format(e))
                self.train_epoch(dev)
                epoch_total = time.time() - epoch_start
                logging.info("Finished epoch [{}] in [{:g}] secs".format(e, epoch_total))

        except EarlyStopException as e:
            logging.info("Early stopping training: "
                         "task [{}] with best score {:.3f}".format(e.task, e.loss))

            self.model.load_state_dict(e.best_state_dict)
            scores = {e.task: e.loss}

        logging.info("Finished training in [{:g}]".format(time.time() - start))

        # will be None if no dev test was provided or the model failed to converge
        return scores
