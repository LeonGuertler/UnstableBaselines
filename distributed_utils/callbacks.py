from ray.tune import Callback

class BroadcastWeightsCallback(Callback):
    def __init__(self, collector):
        self.collector = collector
        self.last_iter = -1

    def handle_result(self, *, result, **kwargs):
        it = result.metrics.get("iteration", -1)
        if it <= self.last_iter:
            return
        self.last_iter = it

        # if result.checkpoint is None:
        #     return

        # ckpt_data = result.checkpoint.to_dict()
        # state_dict = ckpt_data.get("model", None)
        # if state_dict is None:
        #     return

        weights = result.metrics.get("weights")
        if weights is None:
            print("Warning: Checkpoint has no model state.")
            return


        print("UPDATING ALL WEIGHTS")
        self.collector.update_all_weights(weights)
        print(f"[broadcast] pushed iteration {it}")



# from ray.tune import Callback


# class BroadcastWeightsCallback(Callback):
#     def __init__(self, collector):
#         self.collector = collector
#         self.last_iter = -1

#     # Runs on the driver *every time* session.report() fires.
#     def handle_result(self, *, result, **kwargs):
#         it = result.metrics.get("iteration", -1)
#         if it <= self.last_iter:
#             return # nothing new
#         self.last_iter = it

#         state_dict = result.metrics.get("weights")
#         if state_dict is None:
#             return                      # nothing to send
#         print("UPDATING ALL WEIGHTS")
#         self.collector.update_all_weights(state_dict)
#         print(f"[broadcast] pushed iteration {it}")






#         # ckpt = result.checkpoint
#         # if ckpt is None:
#         #     return # shouldn’t happen but be safe

#         # state_dict = ckpt.to_dict()["model"]
#         # self.collector.update_all_weights(state_dict)
