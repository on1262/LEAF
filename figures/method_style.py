"""One color and marker per method, shared by the paper's scatter figures (len_explore, len_update).

Colors follow explore_curves.py; GRPO / SDPO / SCoRe / SEED keep its markers. Ours is a star.
"""

# key: (label, color, marker, marker size in points^2)
METHOD_STYLE = {
    "base": ("Base", "#000000", "h", 34),
    "random_sampling": ("Random Sampling", "#8A8B8C", "X", 34),
    "grpo": ("GRPO", "#00AEBB", "o", 30),
    "sdpo": ("SDPO", "#6A51A3", "s", 28),
    "score": ("SCoRe", "#E69F00", "^", 34),
    "seed": ("SEED", "#3A3A3A", "D", 24),
    "sft": ("SFT", "#2E8B57", "P", 38),
    "leaf": ("Ours", "#C00000", "*", 110),
}
