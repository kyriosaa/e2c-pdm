# Planned

- `sensor_processor.py` — raw int16 counts to engineering units, then windowing
  and Welch PSD features.

Must produce **byte-identical features** to `ml/features.py`, or the deployed
model sees a different distribution than it trained on. Import the offline code
rather than reimplementing it; scale using `accel_fs_g` and the measured ODR, not
datasheet nominals (see `ml/README.md`).
