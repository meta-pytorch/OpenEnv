# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for OpenEnv core."""


def convert_to_ws_url(url: str) -> str:
    """
    Convert an HTTP/HTTPS URL to a WS/WSS URL.

    Args:
        url: The URL to convert.

    Returns:
        The converted WebSocket URL.
    """
    ws_url = url.rstrip("/")
    if ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    elif ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
        ws_url = "ws://" + ws_url
    return ws_url
