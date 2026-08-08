# data/interim/ — generated caches (untracked)

Everything here is derived and safe to delete; it will be rebuilt.

```
<session>/index.json    cached gap map, measured sample rates, loss stats
                        (ml.sessions -- rebuild with --rebuild)
features/<session>/     cached feature windows as .npy + meta.json
                        (ml.features)
```

`index.json` is keyed on the session directory name, so renaming a session
orphans its cache. `ml/sessions.py` bumps `CACHE_VERSION` whenever the index
format or segmentation logic changes, which invalidates stale caches
automatically.
