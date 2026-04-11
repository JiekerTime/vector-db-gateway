"""Caller routing and priority resolution."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

from vector_gateway.config import GatewayConfig, RoutingRule


@dataclass(frozen=True)
class RouteDecision:
    queue_name: str
    service_priority: int
    operation: str
    operation_priority: int


class Router:
    """Resolve a caller and operation into a queue and priorities."""

    def __init__(self, rules: list[RoutingRule], operation_priority: dict[str, int]):
        self._rules = rules
        self._operation_priority = operation_priority

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "Router":
        return cls(config.routing_rules, config.operation_priority)

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def resolve(self, caller: str, operation: str | None = None) -> RouteDecision:
        rule = self._match_rule(caller)
        chosen_operation = operation or rule.operation
        operation_priority = self._operation_priority.get(
            chosen_operation,
            max(self._operation_priority.values(), default=0) + 1,
        )
        return RouteDecision(
            queue_name=rule.queue,
            service_priority=rule.service_priority,
            operation=chosen_operation,
            operation_priority=operation_priority,
        )

    def _match_rule(self, caller: str) -> RoutingRule:
        for rule in self._rules:
            if fnmatch(caller, rule.caller_pattern):
                return rule
        raise ValueError(f"No routing rule matched caller: {caller}")
