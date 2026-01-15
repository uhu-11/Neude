from pylot.planning.planner import Planner

from rrt_star_planner.RRTStar.rrt_star_wrapper import apply_rrt_star
from pylot.coverage_decorator import coverage_decorator
import numpy as np
class RRTStarPlanner(Planner):
    """Wrapper around the RRT* planner.

    Note:
        Details can be found at `RRT* Planner`_.

    Args:
        world: (:py:class:`~pylot.planning.world.World`): A reference to the
            planning world.
        flags (absl.flags): Object to be used to access absl flags.

    .. _RRT* Planner:
       https://github.com/erdos-project/rrt_star_planner
    """
    @coverage_decorator
    def __init__(self, world, flags, logger):
        super().__init__(world, flags, logger)
        self._hyperparameters = {
            "step_size": flags.step_size,
            "max_iterations": flags.max_iterations,
            "end_dist_threshold": flags.end_dist_threshold,
            "obstacle_clearance": flags.obstacle_clearance_rrt,
            "lane_width": flags.lane_width,
            'seed': flags.random_seed, 
        }
    @coverage_decorator
    def run(self, timestamp, ttd=None):
        """Runs the planner.

        Note:
            The planner assumes that the world is up-to-date.

        Returns:
            :py:class:`~pylot.planning.waypoints.Waypoints`: Waypoints of the
            planned trajectory.
        """
        obstacle_list = self._world.get_obstacle_list()
        # obstacle_list=np.array([[392.25,212,392.75,213.75]])
        # print('obstacle_list........', obstacle_list)
        # obstacle_list=np.array([[395.25001597, 233.87530367, 397.25001597, 235.87530367],
        #     [395.5000387 , 232.37530237, 397.5000387 , 234.37530237],
        #     [395.75006142, 230.87530107, 397.75006142, 232.87530107],
        #     [396.00008415, 229.37529977, 398.00008415, 231.37529977],
        #     [396.25010688, 227.87529848, 398.25010688, 229.87529848],
        #     [396.50012961, 226.37529718, 398.50012961, 228.37529718],
        #     [396.75015233, 224.87529588, 398.75015233, 226.87529588],
        #     [397.00017506, 223.37529458, 399.00017506, 225.37529458],
        #     [397.25019779, 221.87529328, 399.25019779, 223.87529328],
        #     [397.50022052, 220.37529198, 399.50022052, 222.37529198]])
        
        if len(obstacle_list) == 0:
            # Do not use RRT* if there are no obstacles.
            # Do not use Hybrid A* if there are no obstacles.
            output_wps = self._world.follow_waypoints(self._flags.target_speed)
        else:
            # RRT* does not take into account the driveable region.
            # It constructs search space as a top down, minimum bounding
            # rectangle with padding in each dimension.
            self._logger.debug("@{}: Hyperparameters: {}".format(
                timestamp, self._hyperparameters))
            initial_conditions = self._compute_initial_conditions(
                obstacle_list)
            self._logger.debug("@{}: Initial conditions: {}".format(
                timestamp, initial_conditions))
            path_x, path_y, success = apply_rrt_star(initial_conditions,
                                                     self._hyperparameters)
            # print(f'rrt_star结果: path_x, path_y, success',  path_x, path_y, success)
            if success:
                self._logger.debug("@{}: RRT* succeeded".format(timestamp))
                speeds = [self._flags.target_speed] * len(path_x)
                self._logger.debug("@{}: RRT* Path X: {}".format(
                    timestamp, path_x.tolist()))
                self._logger.debug("@{}: RRT* Path Y: {}".format(
                    timestamp, path_y.tolist()))
                self._logger.debug("@{}: RRT* Speeds: {}".format(
                    timestamp, speeds))
                output_wps = self.build_output_waypoints(
                    path_x, path_y, speeds)
            else:
                self._logger.error("@{}: RRT* failed. "
                                   "Sending emergency stop.".format(timestamp))
                output_wps = self._world.follow_waypoints(0)
        return output_wps
    @coverage_decorator
    def _compute_initial_conditions(self, obstacles):
        ego_transform = self._world.ego_transform
        self._world.waypoints.remove_completed(ego_transform.location)
        end_index = min(self._flags.num_waypoints_ahead,
                        len(self._world.waypoints.waypoints) - 1)
        if end_index < 0:
            # If no more waypoints left. Then our location is our end wp.
            self._logger.debug("@{}: No more waypoints left")
            end_wp = ego_transform
        else:
            end_wp = self._world.waypoints.waypoints[end_index]
        initial_conditions = {
            "start": ego_transform.location.as_numpy_array_2D(),
            "end": end_wp.location.as_numpy_array_2D(),
            "obs": obstacles,
        }
        # print('initial_conditions:',initial_conditions)
        return initial_conditions
