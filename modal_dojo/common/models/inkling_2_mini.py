"""Inkling 2 Mini's looped text decoder, trained through Miles/TorchTitan."""

from .base import HFModelConfiguration, parse_inkling_response


class Inkling_2_Mini(HFModelConfiguration):
    """26B total / 5.4B active MoE; Miles reads its native HF architecture."""

    model_name = "thinkingmachines/Inkling-2-Mini"
    revision = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
    response_parser = staticmethod(parse_inkling_response)
