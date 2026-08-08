"""
Pi-tier configuration. NOT IMPLEMENTED YET -- see edge/rpi/README.md.

Two rules when this gets filled in:

1. Credentials come from the gitignored .env at the repository root. Do not add
   a third secret store -- pi_edge_twin.py currently reads private.py and that
   is already one too many.

2. Signal-processing constants (window length, Welch nperseg, bins kept, axis
   combination) MUST be imported from ml/config.py rather than restated here.
   If the deployed feature extractor and the offline one drift apart, the model
   sees a different distribution at inference than it trained on, and nothing
   about the measured performance transfers.
"""
