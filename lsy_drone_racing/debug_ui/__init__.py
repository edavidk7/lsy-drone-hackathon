"""Optional debug UI: stream controller state to a web dashboard over ZMQ.

This package is *entirely optional* and gated behind the ``DEBUG_UI_ENABLE`` environment variable.
The live controller imports only :func:`lsy_drone_racing.debug_ui.publisher.get_publisher`, which
lazily imports ``zmq`` and returns ``None`` unless the env var is set. Deployments (incl. the real
drone) that do not set the var, or that do not have ``pyzmq``/``fastapi`` installed, are unaffected.

Run the dashboard with::

    python -m lsy_drone_racing.debug_ui.server --config level2.toml

and the publishing controller with::

    DEBUG_UI_ENABLE=1 python scripts/sim.py --config level2.toml --controller nav_rl_controller.py
"""
