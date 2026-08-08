"""
Cloud-tier configuration. NOT IMPLEMENTED YET -- see cloud/README.md.

Credentials come from the gitignored .env at the repository root, never from
source. Load them with os.environ, e.g.:

    INFLUXDB_URL      = os.environ["INFLUXDB_URL"]
    INFLUXDB_TOKEN    = os.environ["INFLUXDB_TOKEN"]
    INFLUXDB_DATABASE = os.environ["INFLUXDB_DATABASE"]
"""
