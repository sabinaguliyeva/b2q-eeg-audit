# EEG variant of the Brain2Qwerty v1 experiment config.
#
# WHY THIS FILE EXISTS
# --------------------
# brain2qwerty_v1/config/xp_config.py ships MEG-only:
#   study.name  = "Pinet2024Meg"
#   neuro.name  = "MegExtractor"
# ...even though studies/spanishbcbl.py registers Pinet2024Eeg and the EEG
# recordings are in the same HF repo. So the released config reproduces the
# headline MEG numbers and nothing else. This is the EEG counterpart.
#
# Drop this next to xp_config.py (or import and pass the dict to main).

from brain2qwerty_v1.config.xp_config import experiment_config
from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER

# Pinet2024Eeg, per its StudyInfo:
#   62 timelines, 20 subjects, 61 channels @ 1000 Hz
N_EEG_CHANNELS = 61


def eeg_experiment_config(n_virtual_channels: int | None = None) -> dict:
    cfg = experiment_config()

    # --- swap the study ---------------------------------------------------
    cfg["data"]["study"]["name"] = "Pinet2024Eeg"

    # --- swap the extractor ----------------------------------------------
    # MegExtractor and EegExtractor expose an identical field set, so the
    # preprocessing params (50 Hz resample, 0.1-20 Hz bandpass, baseline,
    # clamp 5, RobustScaler) carry over unchanged. Two MEG-only flags go:
    neuro = cfg["data"]["neuro"]
    neuro["name"] = "EegExtractor"
    neuro.pop("allow_maxshield", None)   # MaxShield is an Elekta MEG thing
    neuro.pop("apply_proj", None)        # SSP projectors: MEG-specific here

    # --- THE INTERESTING KNOB --------------------------------------------
    # ENCODER["merger_config"]["n_virtual_channels"] is 270 by default.
    # That is sized for the MEG sensor array. EEG has 61 channels, so the
    # default asks a per-subject Fourier spatial merger to project 61 real
    # sensors into 270 virtual ones — a >4x overcomplete expansion of a
    # signal that is already spatially smeared by volume conduction.
    #
    # Whether that expansion helps, hurts, or is simply wasted capacity on
    # EEG is not answered anywhere in the release. Sweeping this is the
    # cheapest real experiment available on the public data.
    encoder = {**ENCODER}
    if n_virtual_channels is not None:
        encoder["merger_config"] = {
            **ENCODER["merger_config"],
            "n_virtual_channels": n_virtual_channels,
        }
    cfg["brain_model_config"] = encoder
    cfg["transformer_config"] = TRANSFORMER

    return cfg


def eeg_debug_config(n_virtual_channels: int | None = None) -> dict:
    """One timeline, 2 epochs, single GPU. Use this first."""
    cfg = eeg_experiment_config(n_virtual_channels)
    cfg["data"]["study"]["query"] = "timeline_index == 0"
    cfg["n_epochs"] = 2
    cfg["patience"] = 2
    cfg["devices"] = 1
    cfg["save_checkpoints"] = False
    return cfg


# ---------------------------------------------------------------------------
# CROSS-SUBJECT HOLDOUT
# ---------------------------------------------------------------------------
# The shipped Brain2QwertyV1Splitter (brain2qwerty_v1/transforms.py) splits by
# hashing sentence text, so every subject appears in train AND test. That
# measures within-subject decoding.
#
# Conduit records thousands of DIFFERENT people, so the number that matters to
# them is what happens when a subject is never seen in training. With 20 EEG
# subjects you can hold one out. Note that ENCODER uses per-subject layers
# ("per_subject": True in merger_config) — so a held-out subject has no
# learned subject embedding, and how the model degrades there is the whole
# question. Read the splitter before you modify it.
