import erdos
import coverage
import multiprocessing as mp
import os
import signal
from pylot.global_var import COV_FILE_PATH
from pylot.coverage_decorator import coverage_decorator
class TimeToDecisionOperator(erdos.Operator):

    @coverage_decorator
    def __init__(self, pose_stream: erdos.ReadStream,
                 obstacles_stream: erdos.ReadStream,
                 time_to_decision_stream: erdos.WriteStream, flags):
        pose_stream.add_callback(self.on_pose_update,
                                 [time_to_decision_stream])
        obstacles_stream.add_callback(self.on_obstacles_update)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._last_obstacles_msg = None

    @staticmethod
    @coverage_decorator
    def connect(pose_stream: erdos.ReadStream,
                obstacles_stream: erdos.ReadStream):
        return [erdos.WriteStream()]
    @coverage_decorator
    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    @coverage_decorator
    def on_pose_update(self, msg: erdos.Message,
                       time_to_decision_stream: erdos.WriteStream):  
        self._logger.debug('@{}: {} received pose message'.format(
            msg.timestamp, self.config.name))
        ttd = TimeToDecisionOperator.time_to_decision(msg.data.transform,
                                                      msg.data.forward_speed,
                                                      None)
        time_to_decision_stream.send(erdos.Message(msg.timestamp, ttd))
    @coverage_decorator
    def on_obstacles_update(self, msg: erdos.Message):
        self._last_obstacles_msg = msg

    @staticmethod
    @coverage_decorator
    def time_to_decision(pose, forward_speed, obstacles):
        """Computes time to decision (in ms)."""
        # Time to deadline is 400 ms when driving at 10 m/s
        # Deadline decreases by 10ms for every 1m/s of extra speed.
        time_to_deadline = 400 - (forward_speed - 10) * 10
        # TODO: Include other environment information in the calculation.
        return time_to_deadline
