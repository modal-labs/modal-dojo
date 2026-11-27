from modal_dojo._dashboard import app, ensure_creds_secret, fastapi_app  # noqa: F401

# Provision the log-streaming Secret before `modal deploy` loads app dependencies.
ensure_creds_secret()
