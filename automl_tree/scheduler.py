"""Persistent, inexpensive UCB-like scheduler of model experiment operators.

Rewards are heuristic marginal ensemble AUC gains normalized by training cost.
OOF-based reward estimates are NOT unbiased generalization measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from math import log, sqrt
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    backend: str
    operator: str


LGB_OPERATORS = (
    'standard', 'extra_trees', 'dart', 'rf',
    'feature_subset', 'feature_ablation',
)


class ExperimentScheduler:
    def __init__(self, work_dir, exploration=0.35, verbose=True):
        self.path = Path(work_dir) / 'experiment_history.json'
        self.exploration = float(exploration)
        self.verbose = verbose
        if self.exploration < 0:
            raise ValueError('exploration must be nonnegative')
        if self.path.exists():
            self.records = json.loads(self.path.read_text())['records']
        else:
            self.records = {}

    def operators(self, backends, allow_ablation):
        actions = []
        for family in backends:
            variants = LGB_OPERATORS if family == 'lgb' else ('standard',)
            for op in variants:
                if op == 'feature_ablation' and not allow_ablation:
                    continue
                actions.append(Experiment(family, op))
        return actions

    @staticmethod
    def label(action):
        return f'{action.backend}:{action.operator}'

    def plan_batch(self, backends, batch_size, allow_ablation=False):
        if batch_size < 0:
            raise ValueError('batch_size must be nonnegative')
        actions = self.operators(backends, allow_ablation)
        if not actions and batch_size:
            raise ValueError('No operators are available')
        counts = {self.label(a): 0 for a in actions}
        rewards = {key: 0.0 for key in counts}
        for record in self.records.values():
            key = record.get('operator_key')
            if key in counts:
                counts[key] += 1
                rewards[key] += float(record['reward'])
        planned = []
        for _ in range(batch_size):
            total = 1 + sum(counts.values())
            def priority(action):
                key = self.label(action)
                n = counts[key]
                if n == 0:
                    return float('inf')
                return rewards[key] / n + self.exploration * sqrt(log(total + 1) / n)
            chosen = max(actions, key=priority)
            planned.append(chosen)
            counts[self.label(chosen)] += 1
        return planned

    def record(self, member, action, gain, seconds, auc=None, baseline_auc=None):
        """Idempotent across resume/restart: one record per candidate trial."""
        if member in self.records:
            return
        seconds = max(0., float(seconds))
        gain = float(gain)
        reward = min(1., max(0., gain) * 1000.) / (1. + seconds / 60.)
        self.records[member] = {
            'operator_key': self.label(action),
            'gain': gain, 'seconds': seconds, 'reward': reward,
            'auc': None if auc is None else float(auc),
            'baseline_auc': None if baseline_auc is None else float(baseline_auc),
        }
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps({'version': 1, 'records': self.records}, indent=2))
        temp.replace(self.path)

    def summary(self):
        output = {}
        for item in self.records.values():
            key = item['operator_key']
            row = output.setdefault(key, {'trials': 0, 'total_gain': 0.,
                                          'total_seconds': 0., 'mean_reward': 0.})
            row['trials'] += 1
            row['total_gain'] += item['gain']
            row['total_seconds'] += item['seconds']
            row['mean_reward'] += item['reward']
        for row in output.values():
            row['mean_reward'] /= row['trials']
        return output
