from ray.tune import Callback


class BroadcastWeightsCallback(Callback):
    def __init__(self, collector):
        self.collector = collector
        self.last_iter = -1

    # Runs on the driver *every time* session.report() fires.
    def handle_result(self, *, result, **kwargs):
        it = result.metrics.get("iteration", -1)
        if it <= self.last_iter:
            return # nothing new
        self.last_iter = it

        ckpt = result.checkpoint
        if ckpt is None:
            return # shouldn’t happen but be safe

        state_dict = ckpt.to_dict()["model"]
        self.collector.update_all_weights(state_dict)
        print(f"[broadcast] pushed iteration {it}")
