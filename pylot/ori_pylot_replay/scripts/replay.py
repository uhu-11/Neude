import time
import re

from absl import app
from absl import flags

from carla import Location, Rotation, Transform

import pylot.flags
from pylot.simulation.utils import get_world

flags.DEFINE_float('replay_start_time', 0.0,
                   'The time at which to start replaying')
flags.DEFINE_float('replay_duration', 0.0,
                   'The duration of the replay run')
flags.DEFINE_integer('replay_id', 0,
                     'The actor id to follow during the replay')
flags.DEFINE_string('replay_file', '', 'Path to the log file')
flags.DEFINE_float('replayer_time_factor', 0.1,
                   'Replay speed factor. 1.0 is realtime, 0.1 is 10x slower.')

FLAGS = flags.FLAGS


def process_images(image):
    game_time = int(image.timestamp * 1000)
    print('Received frame for {}'.format(game_time))
    # frame = pylot.utils.bgra_to_bgr(to_bgra_array(image))
    # cv2.imshow("test", frame)
    # cv2.waitKey(1)


def _extract_total_time(replay_output):
    match = re.search(r'Total time recorded:\s*([0-9.]+)', replay_output)
    if match:
        return float(match.group(1))
    return None


def _wait_for_vehicle(client, replay_id, timeout_sec=20):
    """Waits for replay actors to appear and returns a vehicle actor."""
    deadline = time.time() + timeout_sec
    fallback_vehicle = None
    while time.time() < deadline:
        world = client.get_world()
        actors = world.get_actors()
        if replay_id > 0:
            vehicle = actors.find(replay_id)
            if vehicle is not None:
                return world, vehicle
        vehicles = actors.filter('vehicle.*')
        if vehicles:
            fallback_vehicle = vehicles[0]
            if replay_id <= 0:
                return world, fallback_vehicle
        time.sleep(0.5)
    return client.get_world(), fallback_vehicle


def main(argv):
    client, world = get_world(FLAGS.simulator_host, FLAGS.simulator_port,
                              FLAGS.simulator_timeout)

    # Replayer time factor is only available in > 0.9.5.
    client.set_replayer_time_factor(FLAGS.replayer_time_factor)
    replay_output = client.replay_file(FLAGS.replay_file, FLAGS.replay_start_time,
                                       FLAGS.replay_duration, FLAGS.replay_id)
    print(replay_output)
    total_time = _extract_total_time(replay_output)
    # Wait for the server to load the map and start spawning replay actors.
    world, vehicle = _wait_for_vehicle(client, FLAGS.replay_id)
    if vehicle is None:
        raise ValueError(
            "Could not find any vehicle actor during replay startup. "
            "Please check replay file and start time.")
    if FLAGS.replay_id > 0 and vehicle.id != FLAGS.replay_id:
        print("Requested replay_id {} not found, fallback to vehicle {}".
              format(FLAGS.replay_id, vehicle.id))
    else:
        print("Following vehicle {}".format(vehicle.id))

    # Install the camera.
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x',
                                   str(FLAGS.camera_image_width))
    camera_blueprint.set_attribute('image_size_y',
                                   str(FLAGS.camera_image_height))

    transform = Transform(Location(2.0, 0.0, 1.4),
                          Rotation(pitch=0, yaw=0, roll=0))

    camera = world.spawn_actor(camera_blueprint, transform, attach_to=vehicle)
    camera.listen(process_images)

    wait_sim_seconds = FLAGS.replay_duration
    if wait_sim_seconds <= 0 and total_time is not None:
        wait_sim_seconds = max(total_time - FLAGS.replay_start_time, 0.0)
    if wait_sim_seconds <= 0:
        wait_sim_seconds = 20.0
    wait_real_seconds = (wait_sim_seconds / FLAGS.replayer_time_factor) + 2.0
    print('Waiting {:.1f}s realtime for replay to finish'.format(
        wait_real_seconds))

    try:
        time.sleep(wait_real_seconds)
    finally:
        camera.stop()
        camera.destroy()


if __name__ == '__main__':
    app.run(main)
