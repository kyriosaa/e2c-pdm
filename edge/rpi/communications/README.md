# Planned

- `usb_listener.py` — read the framed packet stream from the ESP32 over USB CDC.
  The framing is already implemented host-side in
  `edge/esp32/tools/data_catcher.py`; this is that parser minus the file writing.
- `cloud_offload.py` — ship a window to the cloud tier when the local model is
  not confident. This is the link the bandwidth-savings claim measures.
