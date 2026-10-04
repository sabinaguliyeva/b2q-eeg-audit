"""
Audit of the EEG half of the Brain2Qwerty v1 public release.

Runs on CPU in a few minutes. No GPU, no training.

It answers four questions about the released benchmark that the paper and the
repo do not address, all of which bear on whether the published EEG numbers
mean what they are usually taken to mean.

  Q1. How much EEG data actually survives the loader?
  Q2. How is it distributed across the 20 subjects?
  Q3. Does the shipped splitter separate subjects, or only sentences?
  Q4. What is the spatial merger being asked to do on 61 EEG electrodes?

Usage:
    python eeg_benchmark_audit.py
"""

import os
import json
from collections import Counter

import pandas as pd

import studies  # registers Pinet2024Eeg / Pinet2024Meg
from neuralset.events import Study

from brain2qwerty_v1.config.eeg_config import eeg_experiment_config
from brain2qwerty_v1.transforms import SpanishBCBLPreprocessing, Brain2QwertyV1Splitter

OUT = {}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Build the event table exactly as Data.build_events() does
# ---------------------------------------------------------------------------
section("Building EEG event table")

study = Study(name="Pinet2024Eeg", path=os.environ["BRAIN2QWERTY_STUDIES"])
events = study.run()
print(f"raw events: {len(events):,}")

events = SpanishBCBLPreprocessing().run(events)
print(f"after preprocessing: {len(events):,}")

events = Brain2QwertyV1Splitter().run(events)
print(f"after splitter: {len(events):,}")


# ---------------------------------------------------------------------------
# Q1 — how much data survives
# ---------------------------------------------------------------------------
section("Q1. Surviving data")

counts = events.type.value_counts().to_dict()
print(json.dumps({k: int(v) for k, v in counts.items()}, indent=2))

keystrokes = events[events.type == "Keystroke"]
unassigned = keystrokes["split"].isna().sum()
print(f"\nkeystrokes with no split assigned: {unassigned:,}")

OUT["event_counts"] = {k: int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Q2 — per-subject distribution
# ---------------------------------------------------------------------------
section("Q2. Per-subject keystroke counts")

per_subj = keystrokes.groupby("subject").size().sort_values(ascending=False)
print(per_subj.to_string())
print(f"\nsubjects: {per_subj.size}")
print(f"min  {per_subj.min():,}   median  {int(per_subj.median()):,}   max  {per_subj.max():,}")
print(f"ratio max/min: {per_subj.max() / max(per_subj.min(), 1):.1f}x")

OUT["per_subject"] = {str(k): int(v) for k, v in per_subj.items()}


# ---------------------------------------------------------------------------
# Q3 — does the split separate subjects?
# ---------------------------------------------------------------------------
section("Q3. Subject overlap across splits")

by_split = {
    s: set(keystrokes[keystrokes.split == s]["subject"].unique())
    for s in ("train", "val", "test")
}
for s, subs in by_split.items():
    n = len(keystrokes[keystrokes.split == s])
    print(f"{s:6s}  {n:8,} keystrokes   {len(subs):3d} subjects")

train, val, test = by_split["train"], by_split["val"], by_split["test"]
print(f"\nsubjects in BOTH train and test: {len(train & test)}")
print(f"subjects in test but NOT train:  {len(test - train)}")

if test and not (test - train):
    print(
        "\n>>> every test subject is also a training subject.\n"
        ">>> the published EEG metric is WITHIN-subject decoding.\n"
        ">>> no held-out-subject number exists in this release."
    )

OUT["subject_overlap"] = {
    "train_and_test": len(train & test),
    "test_not_train": len(test - train),
}


# ---------------------------------------------------------------------------
# Q4 — the spatial merger on EEG
# ---------------------------------------------------------------------------
section("Q4. Spatial merger sizing")

from brain2qwerty_v1.config.model_config import ENCODER
n_virtual = ENCODER["merger_config"]["n_virtual_channels"]  # the SHIPPED default

eeg_rows = events[events.type == "Eeg"]
n_real = None
for col in ("n_channels", "nchan", "channels"):
    if col in eeg_rows.columns:
        n_real = int(eeg_rows[col].dropna().iloc[0])
        break

print(f"shipped n_virtual_channels: {n_virtual}")
print(f"real EEG electrodes       : {n_real if n_real else '61 (per StudyInfo)'}")
real = n_real or 61
print(f"expansion factor          : {n_virtual / real:.2f}x")
print(
    "\nnote: the shipped default (270) matches the MEG sensor array, not EEG.\n"
    "On 61 electrodes this is an overcomplete projection of a signal already\n"
    "spatially smeared by volume conduction."
)

OUT["merger"] = {"n_virtual_channels": n_virtual, "n_real_channels": real}


# ---------------------------------------------------------------------------
section("Summary")
print(json.dumps(OUT, indent=2)[:2000])

with open("eeg_audit_results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\nwrote eeg_audit_results.json")
