from __future__ import annotations

import unittest

from vector_gateway.app import _resolved_request_device
from vector_gateway.config import GatewayConfig


class DevicePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        import vector_gateway.app as app_module

        self.app_module = app_module
        self.previous_config = getattr(app_module, "_config", None)
        app_module._config = GatewayConfig.model_validate(
            {
                "queues": {
                    "realtime": {
                        "max_batch_size": 8,
                        "max_wait_ms": 15,
                        "max_concurrent_jobs": 1,
                        "preferred_device": "cuda",
                    },
                    "batch": {
                        "max_batch_size": 128,
                        "max_wait_ms": 200,
                        "max_concurrent_jobs": 1,
                        "preferred_device": "cpu",
                    },
                },
                "routing_rules": [
                    {"caller_pattern": "*", "queue": "realtime", "service_priority": 0, "operation": "query"}
                ],
                "operation_priority": {"query": 0},
                "collections": {"documents": {"vector_size": 1024, "distance": "Cosine", "owner": "default"}},
            }
        )

    def tearDown(self) -> None:
        self.app_module._config = self.previous_config

    def test_queue_preferred_device_is_used(self) -> None:
        self.assertEqual(_resolved_request_device("batch", None), "cpu")

    def test_request_device_overrides_queue_policy(self) -> None:
        self.assertEqual(_resolved_request_device("batch", "cuda"), "cuda")


if __name__ == "__main__":
    unittest.main()
