"""Event-driven backbone.

In-process pub/sub by default (deterministic, easy to debug in VS Code). If REDIS_URL is set,
every event is also appended to a Redis Stream (`claims.events`) so external consumers
(workers, Kafka bridge, analytics) can subscribe - this is the scale-out seam.
"""
import json
import logging
from collections import defaultdict

log = logging.getLogger("events")


class EventBus:
    def __init__(self, redis_url=""):
        self.handlers = defaultdict(list)
        self.history = []          # last events, for the /api/events debug endpoint
        self.redis = None
        if redis_url:
            try:
                import redis
                self.redis = redis.Redis.from_url(redis_url)
                self.redis.ping()
            except Exception as exc:  # pragma: no cover
                log.warning("Redis unavailable (%s) - using in-process bus only", exc)
                self.redis = None

    def subscribe(self, topic, fn):
        self.handlers[topic].append(fn)

    def publish(self, topic, payload):
        self.history.append({"topic": topic, "claim": payload.get("claim_id")})
        self.history = self.history[-200:]
        if self.redis is not None:
            try:
                self.redis.xadd("claims.events", {"topic": topic, "payload": json.dumps(payload, default=str)},
                                maxlen=100000, approximate=True)
            except Exception as exc:  # pragma: no cover
                log.warning("redis publish failed: %s", exc)
        for fn in self.handlers[topic]:
            fn(payload)
