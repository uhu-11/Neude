#!/usr/bin/env python3
"""Rebuild one snapshot scene and optionally run Pylot MPC.

This script targets "single-moment scene reconstruction":
- Load world/map/weather/settings from a scene snapshot JSON.
- Spawn ego/NPC/walkers from the snapshot.
- Restore traffic light states from the snapshot.
- Optionally launch pylot.py with MPC in scenario_runner mode so that Pylot
  controls the pre-spawned ego (role_name=hero).
"""

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import carla


def _to_location(data: Dict) -> carla.Location:
    return carla.Location(x=float(data["x"]),
                          y=float(data["y"]),
                          z=float(data["z"]))


def _to_rotation(data: Dict) -> carla.Rotation:
    return carla.Rotation(pitch=float(data["pitch"]),
                          yaw=float(data["yaw"]),
                          roll=float(data["roll"]))


def _to_transform(data: Dict) -> carla.Transform:
    return carla.Transform(_to_location(data["location"]),
                           _to_rotation(data["rotation"]))


def _to_vector(data: Dict) -> carla.Vector3D:
    return carla.Vector3D(x=float(data["x"]),
                          y=float(data["y"]),
                          z=float(data["z"]))


def _distance(tf_a: carla.Transform, tf_b: carla.Transform) -> float:
    loc_a = tf_a.location
    loc_b = tf_b.location
    dx = loc_a.x - loc_b.x
    dy = loc_a.y - loc_b.y
    dz = loc_a.z - loc_b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _copy_transform_with_z_offset(transform: carla.Transform,
                                  z_offset: float) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=transform.location.x,
                       y=transform.location.y,
                       z=transform.location.z + z_offset),
        carla.Rotation(pitch=transform.rotation.pitch,
                       yaw=transform.rotation.yaw,
                       roll=transform.rotation.roll),
    )


def _set_weather(world: carla.World, weather_data: Dict) -> None:
    weather = world.get_weather()
    for key, value in weather_data.items():
        if hasattr(weather, key):
            setattr(weather, key, float(value))
    world.set_weather(weather)


def _weather_readback_dict(world: carla.World) -> Dict[str, float]:
    weather = world.get_weather()
    keys = [
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wetness",
        "fog_density",
        "fog_distance",
        "fog_falloff",
        "scattering_intensity",
        "wind_intensity",
        "sun_altitude_angle",
        "sun_azimuth_angle",
    ]
    result = {}
    for key in keys:
        if hasattr(weather, key):
            result[key] = float(getattr(weather, key))
    return result


def _log_weather_readback(world: carla.World, stage: str) -> None:
    readback = _weather_readback_dict(world)
    parts = ["{}={:.2f}".format(k, v) for k, v in readback.items()]
    print("[INFO] Weather readback ({}): {}".format(stage, ", ".join(parts)))


def _clamp_weather_value(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _clamp_nonnegative(value: float) -> float:
    return max(0.0, float(value))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _camera_exposure_compensation_from_profile(profile: str,
                                               strength: float) -> float:
    targets = {
        "dark": -1.5,
        "very_dark": -3.0,
        "extreme_dark": -5.0,
    }
    return targets[profile] * _clamp_unit(strength)


def _build_weather_from_args(snapshot_weather: Dict, args) -> Dict:
    weather = dict(snapshot_weather)
    if args.weather_profile != "snapshot":
        profiles = {
            "clear": {
                "cloudiness": 10.0,
                "precipitation": 0.0,
                "precipitation_deposits": 0.0,
                "wetness": 0.0,
                "fog_density": 0.0,
                "fog_distance": 100.0,
                "wind_intensity": 5.0,
                "sun_altitude_angle": 45.0,
            },
            "light_rain": {
                "cloudiness": 65.0,
                "precipitation": 35.0,
                "precipitation_deposits": 20.0,
                "wetness": 45.0,
                "fog_density": 12.0,
                "fog_distance": 45.0,
                "wind_intensity": 20.0,
                "sun_altitude_angle": 30.0,
            },
            "heavy_rain_fog": {
                "cloudiness": 95.0,
                "precipitation": 85.0,
                "precipitation_deposits": 85.0,
                "wetness": 95.0,
                "fog_density": 45.0,
                "fog_distance": 20.0,
                "wind_intensity": 45.0,
                "sun_altitude_angle": 12.0,
            },
            "storm_fog": {
                "cloudiness": 100.0,
                "precipitation": 95.0,
                "precipitation_deposits": 90.0,
                "wetness": 100.0,
                "fog_density": 60.0,
                "fog_distance": 12.0,
                "wind_intensity": 70.0,
                "sun_altitude_angle": 8.0,
            },
            # CARLA无真实雪模型，用高湿+高雾+低太阳高度模拟“雪天观感”。
            "snow_like": {
                "cloudiness": 100.0,
                "precipitation": 15.0,
                "precipitation_deposits": 100.0,
                "wetness": 85.0,
                "fog_density": 70.0,
                "fog_distance": 10.0,
                "wind_intensity": 20.0,
                "sun_altitude_angle": 6.0,
            },
            "dense_fog": {
                "cloudiness": 95.0,
                "precipitation": 10.0,
                "precipitation_deposits": 15.0,
                "wetness": 35.0,
                "fog_density": 85.0,
                "fog_distance": 6.0,
                "fog_falloff": 1.8,
                "scattering_intensity": 1.5,
                "wind_intensity": 8.0,
                "sun_altitude_angle": 10.0,
            },
        }
        weather.update(profiles.get(args.weather_profile, {}))

    if args.brightness_profile != "none":
        darkness_targets = {
            "dim": {
                "sun_altitude_angle": 20.0,
                "cloudiness": 65.0,
            },
            "dark": {
                "sun_altitude_angle": 8.0,
                "cloudiness": 85.0,
            },
            "very_dark": {
                "sun_altitude_angle": -6.0,
                "cloudiness": 95.0,
            },
            "night": {
                "sun_altitude_angle": -20.0,
                "cloudiness": 100.0,
            },
        }
        alpha = _clamp_unit(args.brightness_strength)
        target = darkness_targets[args.brightness_profile]
        current_sun = float(weather.get("sun_altitude_angle", 45.0))
        current_cloud = float(weather.get("cloudiness", 30.0))
        weather["sun_altitude_angle"] = \
            current_sun + (target["sun_altitude_angle"] - current_sun) * alpha
        weather["cloudiness"] = \
            current_cloud + (target["cloudiness"] - current_cloud) * alpha

    if args.enable_dense_fog_transform:
        fog_presets = {
            "obvious": {
                "fog_density": 88.0,
                "fog_distance": 5.0,
                "fog_falloff": 2.2,
                "scattering_intensity": 2.0,
                "cloudiness": 100.0,
                "sun_altitude_angle": 8.0,
            },
            "extreme": {
                "fog_density": 98.0,
                "fog_distance": 2.5,
                "fog_falloff": 3.5,
                "scattering_intensity": 3.0,
                "cloudiness": 100.0,
                "sun_altitude_angle": 5.0,
            },
        }
        weather.update(fog_presets[args.dense_fog_level])

    if args.weather_precipitation is not None:
        weather["precipitation"] = _clamp_weather_value(args.weather_precipitation)
    if args.weather_fog_density is not None:
        weather["fog_density"] = _clamp_weather_value(args.weather_fog_density)
    if args.weather_fog_distance is not None:
        weather["fog_distance"] = _clamp_nonnegative(args.weather_fog_distance)
    if args.weather_fog_falloff is not None:
        weather["fog_falloff"] = _clamp_nonnegative(args.weather_fog_falloff)
    if args.weather_scattering_intensity is not None:
        weather["scattering_intensity"] = _clamp_nonnegative(
            args.weather_scattering_intensity)
    if args.weather_wetness is not None:
        weather["wetness"] = _clamp_weather_value(args.weather_wetness)
    if args.weather_cloudiness is not None:
        weather["cloudiness"] = _clamp_weather_value(args.weather_cloudiness)
    if args.weather_wind_intensity is not None:
        weather["wind_intensity"] = _clamp_weather_value(args.weather_wind_intensity)
    if args.weather_sun_altitude is not None:
        weather["sun_altitude_angle"] = float(args.weather_sun_altitude)

    # 可见高斯扰动：对关键天气分量添加N(0, std)扰动。
    if args.weather_gaussian_std > 0:
        if args.weather_noise_seed is not None:
            random.seed(args.weather_noise_seed)
        noisy_keys = [
            "precipitation",
            "fog_density",
            "wetness",
            "cloudiness",
            "wind_intensity",
            "precipitation_deposits",
        ]
        for key in noisy_keys:
            if key in weather:
                weather[key] = _clamp_weather_value(
                    weather[key] + random.gauss(0.0, args.weather_gaussian_std))
    return weather


def _set_world_settings(world: carla.World, settings_data: Dict) -> None:
    settings = world.get_settings()
    if "synchronous_mode" in settings_data:
        settings.synchronous_mode = bool(settings_data["synchronous_mode"])
    if "fixed_delta_seconds" in settings_data:
        fds = settings_data["fixed_delta_seconds"]
        settings.fixed_delta_seconds = float(fds) if fds is not None else None
    if "no_rendering_mode" in settings_data and hasattr(settings,
                                                        "no_rendering_mode"):
        settings.no_rendering_mode = bool(settings_data["no_rendering_mode"])
    world.apply_settings(settings)


def _destroy_dynamic_actors(world: carla.World) -> None:
    actor_list = world.get_actors()
    patterns = [
        "controller.ai.walker",
        "vehicle.*",
        "walker.pedestrian.*",
        "sensor.*",
    ]
    actors_to_destroy = []
    for pattern in patterns:
        actors_to_destroy.extend(actor_list.filter(pattern))
    for actor in actors_to_destroy:
        if actor.is_alive:
            actor.destroy()


def _focus_spectator_on_transform(world: carla.World,
                                  transform: carla.Transform) -> None:
    spectator = world.get_spectator()
    forward = transform.get_forward_vector()
    location = transform.location - carla.Location(x=10.0 * forward.x,
                                                   y=10.0 * forward.y,
                                                   z=0.0)
    spectator_transform = carla.Transform(
        location + carla.Location(z=6.0),
        carla.Rotation(pitch=-20.0, yaw=transform.rotation.yaw, roll=0.0),
    )
    spectator.set_transform(spectator_transform)


def _safe_set_blueprint_attr(bp: carla.ActorBlueprint, key: str, value) -> None:
    if not bp.has_attribute(key):
        return
    try:
        bp.set_attribute(key, str(value))
    except Exception:
        return


def _spawn_vehicle(
        world: carla.World,
        bp_lib: carla.BlueprintLibrary,
        actor_data: Dict,
        ego_actor_id: Optional[int],
        traffic_manager_port: int,
        enable_npc_autopilot: bool = True,
        apply_initial_dynamics_for_npc: bool = True,
) -> Tuple[Optional[carla.Vehicle], Optional[int]]:
    type_id = actor_data["type_id"]
    transform = _to_transform(actor_data["transform"])
    original_id = int(actor_data["actor_id"])
    try:
        blueprint = bp_lib.find(type_id)
    except Exception:
        return None, original_id

    attrs = actor_data.get("attributes", {})
    for key, value in attrs.items():
        _safe_set_blueprint_attr(blueprint, key, value)

    is_ego = ego_actor_id is not None and original_id == ego_actor_id
    if blueprint.has_attribute("role_name"):
        role_name = "hero" if is_ego else attrs.get("role_name", "autopilot")
        _safe_set_blueprint_attr(blueprint, "role_name", role_name)

    actor = None
    # Runtime snapshots can place actors very close. Retry with slight Z lifts.
    for z_offset in [0.0, 0.2, 0.5, 1.0]:
        spawn_tf = _copy_transform_with_z_offset(transform, z_offset)
        candidate = world.try_spawn_actor(blueprint, spawn_tf)
        if candidate is None:
            continue
        if not isinstance(candidate, carla.Vehicle):
            candidate.destroy()
            continue
        actor = candidate
        break
    if actor is None:
        return None, original_id

    # Force actor pose to snapshot pose after successful spawn.
    actor.set_transform(transform)

    if is_ego:
        velocity = actor_data.get("velocity_mps")
        if velocity:
            actor.set_target_velocity(_to_vector(velocity))
        angular_velocity = actor_data.get("angular_velocity_radps")
        if angular_velocity:
            actor.set_target_angular_velocity(_to_vector(angular_velocity))
        control_data = actor_data.get("control")
        if control_data:
            control = carla.VehicleControl()
            control.throttle = float(control_data.get("throttle", 0.0))
            control.steer = float(control_data.get("steer", 0.0))
            control.brake = float(control_data.get("brake", 0.0))
            control.hand_brake = bool(control_data.get("hand_brake", False))
            control.reverse = bool(control_data.get("reverse", False))
            control.manual_gear_shift = bool(
                control_data.get("manual_gear_shift", False))
            control.gear = int(control_data.get("gear", 0))
            actor.apply_control(control)
        actor.set_autopilot(False, traffic_manager_port)
    else:
        if apply_initial_dynamics_for_npc:
            velocity = actor_data.get("velocity_mps")
            if velocity:
                actor.set_target_velocity(_to_vector(velocity))
            angular_velocity = actor_data.get("angular_velocity_radps")
            if angular_velocity:
                actor.set_target_angular_velocity(_to_vector(angular_velocity))
            control_data = actor_data.get("control")
            if control_data:
                control = carla.VehicleControl()
                control.throttle = float(control_data.get("throttle", 0.0))
                control.steer = float(control_data.get("steer", 0.0))
                control.brake = float(control_data.get("brake", 0.0))
                control.hand_brake = bool(control_data.get("hand_brake", False))
                control.reverse = bool(control_data.get("reverse", False))
                control.manual_gear_shift = bool(
                    control_data.get("manual_gear_shift", False))
                control.gear = int(control_data.get("gear", 0))
                actor.apply_control(control)
        actor.set_autopilot(bool(enable_npc_autopilot), traffic_manager_port)
    return actor, original_id


def _spawn_walker(world: carla.World, bp_lib: carla.BlueprintLibrary,
                  actor_data: Dict) -> Tuple[Optional[carla.Walker], Optional[int]]:
    type_id = actor_data["type_id"]
    transform = _to_transform(actor_data["transform"])
    original_id = int(actor_data["actor_id"])
    try:
        blueprint = bp_lib.find(type_id)
    except Exception:
        return None, original_id

    attrs = actor_data.get("attributes", {})
    for key, value in attrs.items():
        _safe_set_blueprint_attr(blueprint, key, value)

    actor = None
    for z_offset in [0.0, 0.2, 0.5]:
        spawn_tf = _copy_transform_with_z_offset(transform, z_offset)
        candidate = world.try_spawn_actor(blueprint, spawn_tf)
        if candidate is None:
            continue
        if not isinstance(candidate, carla.Walker):
            candidate.destroy()
            continue
        actor = candidate
        break
    if actor is None:
        return None, original_id

    actor.set_transform(transform)
    velocity = actor_data.get("velocity_mps")
    if velocity:
        actor.set_target_velocity(_to_vector(velocity))
    return actor, original_id


def _get_walker_speed_mps(walker_data: Dict) -> float:
    control_data = walker_data.get("control", {})
    if "speed" in control_data:
        try:
            speed = float(control_data["speed"])
            if speed >= 0.0:
                return speed
        except (TypeError, ValueError):
            pass
    attrs = walker_data.get("attributes", {})
    if "speed" in attrs:
        try:
            speed = float(attrs["speed"])
            if speed >= 0.0:
                return speed
        except (TypeError, ValueError):
            pass
    return 1.4


def _normalize_direction(direction: carla.Vector3D) -> Optional[carla.Vector3D]:
    magnitude = math.sqrt(direction.x * direction.x + direction.y * direction.y +
                          direction.z * direction.z)
    if magnitude < 1e-6:
        return None
    return carla.Vector3D(x=direction.x / magnitude,
                          y=direction.y / magnitude,
                          z=0.0)


def _extract_walker_initial_direction(walker_data: Dict) -> Optional[carla.Vector3D]:
    # 1) Prefer saved walker control direction.
    control_data = walker_data.get("control", {})
    control_direction = control_data.get("direction")
    if isinstance(control_direction, dict):
        direction = _normalize_direction(_to_vector(control_direction))
        if direction is not None:
            return direction

    # 2) Fall back to velocity direction.
    velocity_data = walker_data.get("velocity_mps")
    if isinstance(velocity_data, dict):
        direction = _normalize_direction(_to_vector(velocity_data))
        if direction is not None:
            return direction

    # 3) Last resort: infer from yaw.
    transform_data = walker_data.get("transform", {})
    rotation_data = transform_data.get("rotation", {})
    yaw = rotation_data.get("yaw")
    if yaw is None:
        return None
    yaw_radians = math.radians(float(yaw))
    direction = carla.Vector3D(x=math.cos(yaw_radians),
                               y=math.sin(yaw_radians),
                               z=0.0)
    return _normalize_direction(direction)


def _init_manual_walker_states(
        world: carla.World,
        walker_entries: List[Tuple[carla.Walker, Dict]],
        keep_initial_direction_seconds: float,
        walker_speed_scale: float,
        walker_max_speed: float,
        fixed_delta_seconds: Optional[float]) -> List[Dict]:
    """Initializes state for manual walker control mode."""
    states = []
    if fixed_delta_seconds is None or fixed_delta_seconds <= 0:
        # Fallback for async mode.
        initial_steps = max(0, int(keep_initial_direction_seconds / 0.05))
    else:
        initial_steps = max(0, int(keep_initial_direction_seconds /
                                   fixed_delta_seconds))
    for walker, walker_data in walker_entries:
        if walker is None or not walker.is_alive:
            continue
        initial_direction = _extract_walker_initial_direction(walker_data)
        target = None if initial_direction is not None \
            else world.get_random_location_from_navigation()
        base_speed = _get_walker_speed_mps(walker_data)
        scaled_speed = base_speed * max(0.0, walker_speed_scale)
        speed = max(0.05, min(scaled_speed, max(0.05, walker_max_speed)))
        states.append({
            "actor": walker,
            "speed": speed,
            "initial_direction": initial_direction,
            "initial_steps_left": initial_steps,
            "target": target,
        })
    return states


def _update_manual_walker_controls(world: carla.World,
                                   walker_states: List[Dict],
                                   reach_threshold: float = 1.5) -> int:
    """Applies WalkerControl every step without AI controller actors."""
    active_count = 0
    for state in walker_states:
        walker = state["actor"]
        if walker is None or not walker.is_alive:
            continue

        direction = None
        if state.get("initial_steps_left", 0) > 0:
            direction = state.get("initial_direction")
            state["initial_steps_left"] -= 1
        else:
            location = walker.get_location()
            target = state.get("target")
            if target is None:
                target = world.get_random_location_from_navigation()
                state["target"] = target
                if target is None:
                    continue

            offset = carla.Vector3D(x=target.x - location.x,
                                    y=target.y - location.y,
                                    z=target.z - location.z)
            distance = math.sqrt(offset.x * offset.x + offset.y * offset.y +
                                 offset.z * offset.z)
            if distance < reach_threshold:
                target = world.get_random_location_from_navigation()
                state["target"] = target
                if target is None:
                    continue
                offset = carla.Vector3D(x=target.x - location.x,
                                        y=target.y - location.y,
                                        z=target.z - location.z)
                distance = math.sqrt(offset.x * offset.x + offset.y * offset.y +
                                     offset.z * offset.z)
                if distance < 1e-3:
                    continue
            direction = _normalize_direction(offset)
            if direction is None:
                continue

        if direction is None:
            continue
        control = carla.WalkerControl(direction=direction,
                                      speed=float(state["speed"]),
                                      jump=False)
        walker.apply_control(control)
        active_count += 1
    return active_count


def _apply_vehicle_freeze(actor: carla.Vehicle, hand_brake: bool = True) -> None:
    if actor is None or not actor.is_alive:
        return
    # Use brake-based freeze only. Forcing target velocity to zero can
    # occasionally prevent TrafficManager from promptly resuming control.
    control = carla.VehicleControl(throttle=0.0,
                                   steer=0.0,
                                   brake=1.0,
                                   hand_brake=hand_brake)
    actor.apply_control(control)


def _apply_walker_freeze(walker_states: List[Dict]) -> None:
    for state in walker_states:
        walker = state.get("actor")
        if walker is None or not walker.is_alive:
            continue
        walker.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(x=0.0, y=0.0, z=0.0),
                                speed=0.0,
                                jump=False))


def _set_npc_physics_lock(npc_actors: List[carla.Vehicle], locked: bool) -> None:
    """Hard-lock NPC physics during startup freeze to prevent rolling."""
    for actor in npc_actors:
        if actor is None or not actor.is_alive:
            continue
        try:
            actor.set_simulate_physics(not locked)
        except RuntimeError:
            continue


def _apply_npc_pose_lock(npc_actors: List[carla.Vehicle],
                         npc_freeze_transforms: Dict[int, carla.Transform]) -> None:
    """Force NPC pose back to snapshot transform during freeze."""
    for actor in npc_actors:
        if actor is None or not actor.is_alive:
            continue
        target_transform = npc_freeze_transforms.get(actor.id)
        if target_transform is None:
            continue
        try:
            actor.set_transform(target_transform)
        except RuntimeError:
            continue


def _extract_npc_initial_dynamic_state(actor_data: Dict) -> Dict:
    velocity = actor_data.get("velocity_mps")
    angular_velocity = actor_data.get("angular_velocity_radps")
    control_data = actor_data.get("control", {})
    throttle = float(control_data.get("throttle", 0.0))
    brake = float(control_data.get("brake", 0.0))
    steer = float(control_data.get("steer", 0.0))
    return {
        "velocity": velocity if isinstance(velocity, dict) else None,
        "angular_velocity": angular_velocity
        if isinstance(angular_velocity, dict) else None,
        "throttle": throttle,
        "brake": brake,
        "steer": steer,
    }


def _apply_vehicle_dynamic_state(actor: carla.Vehicle,
                                 dynamic_state: Optional[Dict],
                                 clear_freeze_control: bool = True) -> bool:
    if actor is None or not actor.is_alive:
        return False
    if clear_freeze_control:
        actor.apply_control(
            carla.VehicleControl(throttle=0.0,
                                 steer=0.0,
                                 brake=0.0,
                                 hand_brake=False))
    if not dynamic_state:
        return False
    velocity = dynamic_state.get("velocity")
    angular_velocity = dynamic_state.get("angular_velocity")
    if velocity:
        actor.set_target_velocity(_to_vector(velocity))
    if angular_velocity:
        actor.set_target_angular_velocity(_to_vector(angular_velocity))
    return bool(velocity or angular_velocity)


def _release_npc_autopilot(npc_actors: List[carla.Vehicle],
                           traffic_manager_port: int,
                           npc_initial_dynamics: Optional[Dict[int, Dict]] = None,
                           apply_snapshot_kick: bool = True
                           ) -> int:
    """Re-enable NPC autopilot after startup freeze."""
    _set_npc_physics_lock(npc_actors, locked=False)
    released = 0
    for actor in npc_actors:
        if actor is None or not actor.is_alive:
            continue
        # Toggle autopilot OFF->ON to force TM re-registration.
        actor.set_autopilot(False, traffic_manager_port)
        # Clear brake/hand_brake state left by freeze stage.
        _apply_vehicle_dynamic_state(actor,
                                     dynamic_state={},
                                     clear_freeze_control=True)
        if npc_initial_dynamics is not None:
            dynamic_state = npc_initial_dynamics.get(actor.id)
        else:
            dynamic_state = None
        if dynamic_state and apply_snapshot_kick:
            _apply_vehicle_dynamic_state(actor,
                                         dynamic_state=dynamic_state,
                                         clear_freeze_control=False)
            # If snapshot indicates positive throttle and no braking, provide
            # a tiny kick before autopilot to reduce post-release deadlock.
            if dynamic_state.get("throttle", 0.0) > 0.05 and \
                    dynamic_state.get("brake", 0.0) < 0.05:
                actor.apply_control(
                    carla.VehicleControl(throttle=min(
                        0.2, dynamic_state.get("throttle", 0.2)),
                        steer=max(-0.3, min(0.3, dynamic_state.get("steer", 0.0))),
                        brake=0.0,
                        hand_brake=False))
        actor.set_autopilot(True, traffic_manager_port)
        released += 1
    return released


def _vehicle_speed_mps(vehicle: carla.Vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y +
                     velocity.z * velocity.z)


def _revive_stalled_npcs(npc_actors: List[carla.Vehicle],
                         traffic_manager_port: int,
                         min_speed_mps: float = 0.2,
                         actor_states: Optional[Dict[int, Dict]] = None,
                         current_frame: Optional[int] = None,
                         max_attempts: int = 2,
                         cooldown_frames: int = 40) -> int:
    """Re-activate stalled NPC autopilot vehicles after release."""
    revived = 0
    for actor in npc_actors:
        if actor is None or not actor.is_alive:
            continue
        state = None
        if actor_states is not None:
            state = actor_states.setdefault(
                actor.id, {
                    "attempts": 0,
                    "last_revive_frame": -10**9,
                })
            if state["attempts"] >= max(0, max_attempts):
                continue
            if current_frame is not None and \
                    (current_frame - state["last_revive_frame"]) < \
                    max(1, cooldown_frames):
                continue
        if _vehicle_speed_mps(actor) >= min_speed_mps:
            continue
        # Small one-frame kick + TM re-register to avoid persistent stall.
        actor.set_autopilot(False, traffic_manager_port)
        actor.apply_control(
            carla.VehicleControl(throttle=0.12,
                                 steer=0.0,
                                 brake=0.0,
                                 hand_brake=False))
        actor.set_autopilot(True, traffic_manager_port)
        if state is not None:
            state["attempts"] += 1
            if current_frame is not None:
                state["last_revive_frame"] = current_frame
        revived += 1
    return revived


def _traffic_light_state_from_str(state: str) -> carla.TrafficLightState:
    value = str(state).lower()
    if "red" in value:
        return carla.TrafficLightState.Red
    if "yellow" in value:
        return carla.TrafficLightState.Yellow
    if "green" in value:
        return carla.TrafficLightState.Green
    return carla.TrafficLightState.Off


def _restore_traffic_lights(world: carla.World, tl_snapshots: List[Dict]) -> None:
    world_tls = list(world.get_actors().filter("traffic.traffic_light*"))
    by_id = {tl.id: tl for tl in world_tls}
    assigned = set()
    for tl_data in tl_snapshots:
        state = _traffic_light_state_from_str(tl_data.get("state", "Off"))
        src_id = int(tl_data["actor_id"])
        target_tl = by_id.get(src_id)
        if target_tl is None:
            src_tf = _to_transform(tl_data["transform"])
            best = None
            best_dist = 1e9
            for world_tl in world_tls:
                if world_tl.id in assigned:
                    continue
                dist = _distance(src_tf, world_tl.get_transform())
                if dist < best_dist:
                    best_dist = dist
                    best = world_tl
            target_tl = best
        if target_tl is None:
            continue
        target_tl.set_state(state)
        assigned.add(target_tl.id)


def _extract_town_name(map_name: str) -> str:
    # Snapshot may contain "Town02" or "Carla/Maps/Town02".
    if "/" in map_name:
        return map_name.split("/")[-1]
    return map_name


def _load_snapshot(path: str) -> Dict:
    with open(path, "r") as infile:
        return json.load(infile)


def _ensure_output_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _build_screenshot_overlay_flags(mode: str) -> List[str]:
    """Builds visualization flags used by screenshot capture.

    Modes:
      - none: pure RGB camera without boxes/trajectory overlays.
      - waypoint_only: only planning trajectory line on camera frame.
      - bbox_only: only detected obstacle boxes on camera frame.
    """
    if mode == "none":
        return [
            "--visualize_rgb_camera",
            "--novisualize_waypoints",
            "--novisualize_detected_obstacles",
            "--novisualize_tracked_obstacles",
            "--novisualize_detected_traffic_lights",
        ]
    if mode == "waypoint_only":
        return [
            "--novisualize_rgb_camera",
            "--visualize_waypoints",
            "--draw_waypoints_on_camera_frames",
            "--nodraw_waypoint_speed_on_camera_frames",
            "--nodraw_waypoint_speed_anchor_on_camera_frames",
            "--nodraw_obstacles_on_waypoint_frames",
            "--novisualize_detected_obstacles",
            "--novisualize_tracked_obstacles",
            "--novisualize_detected_traffic_lights",
        ]
    if mode == "bbox_only":
        return [
            "--novisualize_rgb_camera",
            "--novisualize_waypoints",
            "--visualize_detected_obstacles",
            "--novisualize_tracked_obstacles",
            "--novisualize_detected_traffic_lights",
        ]
    raise ValueError("Unexpected screenshot overlay mode: {}".format(mode))


def _build_pylot_command(args, snapshot: Dict) -> List[str]:
    cmd = [args.python_executable, args.pylot_script]
    if args.pylot_flagfile:
        cmd.append("--flagfile={}".format(args.pylot_flagfile))

    map_name = _extract_town_name(snapshot["carla_world"]["map_name"])
    town_number = 1
    if map_name.lower().startswith("town"):
        try:
            town_number = int(map_name[4:])
        except ValueError:
            town_number = 1

    cmd.extend([
        "--scenario_runner=true",
        "--control=mpc",
        "--simulator_town={}".format(town_number),
        "--simulator_host={}".format(args.host),
        "--simulator_port={}".format(args.port),
        "--carla_traffic_manager_port={}".format(args.traffic_manager_port),
    ])
    if args.enable_localization_gaussian_noise:
        cmd.extend([
            "--accel_noise_stddev_x={}".format(args.accel_noise_stddev_x),
            "--accel_noise_stddev_y={}".format(args.accel_noise_stddev_y),
            "--accel_noise_stddev_z={}".format(args.accel_noise_stddev_z),
            "--gyro_noise_stddev_x={}".format(args.gyro_noise_stddev_x),
            "--gyro_noise_stddev_y={}".format(args.gyro_noise_stddev_y),
            "--gyro_noise_stddev_z={}".format(args.gyro_noise_stddev_z),
            "--gnss_noise_stddev_alt={}".format(args.gnss_noise_stddev_alt),
            "--gnss_noise_stddev_lat={}".format(args.gnss_noise_stddev_lat),
            "--gnss_noise_stddev_lon={}".format(args.gnss_noise_stddev_lon),
        ])
    if args.camera_brightness_profile != "none":
        cmd.extend([
            "--simulator_camera_exposure_mode=histogram",
            "--simulator_camera_exposure_compensation={}".format(
                _camera_exposure_compensation_from_profile(
                    args.camera_brightness_profile,
                    args.camera_brightness_strength)),
        ])
    if args.camera_exposure_compensation is not None:
        cmd.extend([
            "--simulator_camera_exposure_mode=histogram",
            "--simulator_camera_exposure_compensation={}".format(
                args.camera_exposure_compensation),
        ])
    if args.save_pylot_screenshots:
        cmd.extend([
            "--save_visualizer_screenshots",
            "--visualizer_screenshot_path={}".format(args.pylot_screenshot_path),
            "--visualizer_screenshot_every_nth_message={}".format(
                max(1, args.pylot_screenshot_every_nth_message)),
        ])
    if args.pylot_extra_flags:
        cmd.extend(shlex.split(args.pylot_extra_flags))
    cmd.extend(_build_screenshot_overlay_flags(args.screenshot_overlay_mode))
    if args.hide_visualizer_hud:
        cmd.append("--novisualize_hud")
    return cmd


def _spawn_ego_fallback(world: carla.World, bp_lib: carla.BlueprintLibrary,
                        ego_snapshot: Dict,
                        traffic_manager_port: int) -> Optional[carla.Vehicle]:
    if not ego_snapshot:
        return None
    actor_state = ego_snapshot.get("carla_actor_state")
    if not actor_state:
        return None
    type_id = actor_state.get("type_id", "vehicle.lincoln.mkz2017")
    transform = _to_transform(actor_state["transform"])
    try:
        blueprint = bp_lib.find(type_id)
    except Exception:
        return None
    attrs = actor_state.get("attributes", {})
    for key, value in attrs.items():
        _safe_set_blueprint_attr(blueprint, key, value)
    _safe_set_blueprint_attr(blueprint, "role_name", "hero")

    actor = None
    for z_offset in [0.5, 1.0, 2.0]:
        spawn_tf = _copy_transform_with_z_offset(transform, z_offset)
        candidate = world.try_spawn_actor(blueprint, spawn_tf)
        if candidate is None:
            continue
        if not isinstance(candidate, carla.Vehicle):
            candidate.destroy()
            continue
        actor = candidate
        break
    if actor is None:
        return None
    actor.set_transform(transform)
    actor.set_autopilot(False, traffic_manager_port)
    return actor


def _tick_for_stability(world: carla.World,
                        num_ticks: int,
                        sleep_s: float,
                        on_step: Optional[Callable[[], None]] = None):
    settings = world.get_settings()
    for _ in range(max(num_ticks, 0)):
        if on_step is not None:
            on_step()
        if settings.synchronous_mode:
            world.tick()
        else:
            time.sleep(sleep_s)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild snapshot and run Pylot MPC")
    parser.add_argument("--snapshot_json",
                        required=True,
                        help="Path to scene-<timestamp>.json")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout",
                        type=float,
                        default=20.0,
                        help="CARLA client timeout")
    parser.add_argument("--traffic_manager_port",
                        type=int,
                        default=8000,
                        help="CARLA traffic manager port")
    parser.add_argument("--keep_existing_actors",
                        action="store_true",
                        help="Do not destroy existing dynamic actors first")
    parser.add_argument("--setup_only",
                        action="store_true",
                        help="Only setup scene, do not launch Pylot")
    parser.add_argument("--stability_ticks",
                        type=int,
                        default=3,
                        help="Tick world N times after setup")
    parser.add_argument("--python_executable",
                        default="python3",
                        help="Python executable used to run pylot.py")
    parser.add_argument("--pylot_script",
                        default="pylot.py",
                        help="Path to pylot.py")
    parser.add_argument("--pylot_flagfile",
                        default="configs/mpc.conf",
                        help="Pylot flagfile path")
    parser.add_argument("--pylot_extra_flags",
                        default="",
                        help="Additional flags appended to pylot command")
    parser.add_argument(
        "--weather_profile",
        choices=[
            "snapshot", "clear", "light_rain", "heavy_rain_fog", "storm_fog",
            "snow_like", "dense_fog"
        ],
        default="snapshot",
        help="Weather profile override for camera-visible perturbations")
    parser.add_argument("--brightness_profile",
                        choices=["none", "dim", "dark", "very_dark", "night"],
                        default="none",
                        help="Brightness transform profile for darkening")
    parser.add_argument("--brightness_strength",
                        type=float,
                        default=1.0,
                        help="Brightness transform strength in [0,1]")
    parser.add_argument(
        "--camera_brightness_profile",
        choices=["none", "dark", "very_dark", "extreme_dark"],
        default="none",
        help="Darken Pylot camera via exposure compensation")
    parser.add_argument("--camera_brightness_strength",
                        type=float,
                        default=1.0,
                        help="Camera brightness transform strength in [0,1]")
    parser.add_argument("--camera_exposure_compensation",
                        type=float,
                        default=None,
                        help="Direct exposure compensation override for Pylot camera")
    parser.add_argument("--save_pylot_screenshots",
                        action="store_true",
                        default=True,
                        help="Save Pylot visualizer screenshots with overlays")
    parser.add_argument("--no_save_pylot_screenshots",
                        action="store_false",
                        dest="save_pylot_screenshots",
                        help="Disable saving Pylot visualizer screenshots")
    parser.add_argument(
        "--pylot_screenshot_path",
        default="/media/lzq/D/lzq/ori_pylot/pylot-master/pylot_screenshot_save/pylot_screenshot",
        help="Directory used for saving Pylot visualizer screenshots")
    parser.add_argument("--pylot_screenshot_every_nth_message",
                        type=int,
                        default=1,
                        help="Save one screenshot every N visualizer updates")
    parser.add_argument(
        "--screenshot_overlay_mode",
        choices=["none", "waypoint_only", "bbox_only"],
        default="none",
        help=("Overlay style for saved screenshots: none (default, pure RGB), "
              "waypoint_only, or bbox_only"))
    parser.add_argument(
        "--hide_visualizer_hud",
        action="store_true",
        default=False,
        help=("Hide Pylot visualizer top-left HUD overlay in live window "
              "and saved screenshots."))
    parser.add_argument("--enable_dense_fog_transform",
                        action="store_true",
                        default=False,
                        help="Enable an obvious fog transform for camera view")
    parser.add_argument("--dense_fog_level",
                        choices=["obvious", "extreme"],
                        default="obvious",
                        help="Fog intensity level when dense fog transform is enabled")
    parser.add_argument("--weather_precipitation",
                        type=float,
                        default=None,
                        help="Override precipitation [0,100]")
    parser.add_argument("--weather_fog_density",
                        type=float,
                        default=None,
                        help="Override fog density [0,100]")
    parser.add_argument("--weather_fog_distance",
                        type=float,
                        default=None,
                        help="Override fog distance (meters, lower means thicker)")
    parser.add_argument("--weather_fog_falloff",
                        type=float,
                        default=None,
                        help="Override fog falloff (higher means denser nearby fog)")
    parser.add_argument("--weather_scattering_intensity",
                        type=float,
                        default=None,
                        help="Override volumetric scattering intensity")
    parser.add_argument("--weather_wetness",
                        type=float,
                        default=None,
                        help="Override wetness [0,100]")
    parser.add_argument("--weather_cloudiness",
                        type=float,
                        default=None,
                        help="Override cloudiness [0,100]")
    parser.add_argument("--weather_wind_intensity",
                        type=float,
                        default=None,
                        help="Override wind intensity [0,100]")
    parser.add_argument("--weather_sun_altitude",
                        type=float,
                        default=None,
                        help="Override sun altitude angle")
    parser.add_argument(
        "--weather_gaussian_std",
        type=float,
        default=0.0,
        help="Gaussian stddev applied to weather components for visible noise")
    parser.add_argument("--weather_noise_seed",
                        type=int,
                        default=None,
                        help="Random seed for weather gaussian perturbation")
    parser.add_argument("--keep_seconds",
                        type=int,
                        default=0,
                        help="Keep process alive N seconds after setup")
    parser.add_argument("--manual_walker_control",
                        action="store_true",
                        default=True,
                        help="Use manual WalkerControl each tick")
    parser.add_argument("--manual_walker_interval",
                        type=float,
                        default=0.05,
                        help="Manual walker control update interval in seconds")
    parser.add_argument(
        "--walker_keep_initial_direction_seconds",
        type=float,
        default=3.0,
        help="Keep snapshot direction before switching to random navigation")
    parser.add_argument("--walker_speed_scale",
                        type=float,
                        default=0.5,
                        help="Global scale for walker speeds")
    parser.add_argument("--walker_max_speed",
                        type=float,
                        default=1.0,
                        help="Hard max walker speed in m/s")
    parser.add_argument(
        "--startup_mode",
        choices=["immediate", "freeze_then_go"],
        default="freeze_then_go",
        help="How to start simulation after Pylot launch")
    parser.add_argument("--startup_freeze_seconds",
                        type=float,
                        default=2.0,
                        help="Freeze duration before coordinated release")
    parser.add_argument("--freeze_npc_during_startup",
                        action="store_true",
                        default=True,
                        help="Freeze NPC vehicles during startup freeze")
    parser.add_argument("--no_freeze_npc_during_startup",
                        action="store_false",
                        dest="freeze_npc_during_startup",
                        help="Do not freeze NPC vehicles during startup freeze")
    parser.add_argument("--npc_revival_monitor_seconds",
                        type=float,
                        default=6.0,
                        help="Monitor and revive stalled NPCs after release")
    parser.add_argument("--npc_revival_check_interval_frames",
                        type=int,
                        default=10,
                        help="Frame interval for stalled NPC checks")
    parser.add_argument("--npc_revival_max_attempts",
                        type=int,
                        default=2,
                        help="Maximum revive attempts per NPC")
    parser.add_argument("--npc_revival_cooldown_frames",
                        type=int,
                        default=40,
                        help="Minimum frame gap between revive attempts")
    parser.add_argument("--npc_revival_grace_seconds",
                        type=float,
                        default=1.5,
                        help="Wait time after release before first revive check")
    parser.add_argument("--npc_stall_min_speed_mps",
                        type=float,
                        default=0.2,
                        help="Speed threshold to consider NPC stalled")
    parser.add_argument(
        "--npc_snapshot_hold_seconds",
        type=float,
        default=1.0,
        help="After startup release, keep NPCs on snapshot dynamics for N seconds before enabling autopilot")
    parser.add_argument(
        "--npc_release_without_snapshot_kick",
        action="store_true",
        default=False,
        help=("Skip snapshot velocity/tiny-throttle kick when releasing NPCs "
              "to autopilot"))
    parser.add_argument("--enable_localization_gaussian_noise",
                        action="store_true",
                        default=False,
                        help="Enable IMU/GNSS gaussian noise in Pylot (not camera-visible)")
    parser.add_argument("--accel_noise_stddev_x", type=float, default=0.001)
    parser.add_argument("--accel_noise_stddev_y", type=float, default=0.001)
    parser.add_argument("--accel_noise_stddev_z", type=float, default=0.015)
    parser.add_argument("--gyro_noise_stddev_x", type=float, default=0.001)
    parser.add_argument("--gyro_noise_stddev_y", type=float, default=0.001)
    parser.add_argument("--gyro_noise_stddev_z", type=float, default=0.001)
    parser.add_argument("--gnss_noise_stddev_alt", type=float, default=0.000005)
    parser.add_argument("--gnss_noise_stddev_lat", type=float, default=0.000005)
    parser.add_argument("--gnss_noise_stddev_lon", type=float, default=0.000005)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.snapshot_json):
        raise FileNotFoundError("Snapshot file not found: {}".format(
            args.snapshot_json))
    snapshot = _load_snapshot(args.snapshot_json)
    if args.save_pylot_screenshots:
        _ensure_output_dir(args.pylot_screenshot_path)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    map_name = _extract_town_name(snapshot["carla_world"]["map_name"])
    print("[INFO] Loading world {}".format(map_name))
    world = client.load_world(map_name)
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)

    settings_data = snapshot["carla_world"]["settings"]
    _set_world_settings(world, settings_data)
    if hasattr(traffic_manager, "set_synchronous_mode"):
        traffic_manager.set_synchronous_mode(
            bool(settings_data.get("synchronous_mode", True)))

    weather_data = _build_weather_from_args(snapshot["carla_world"]["weather"],
                                            args)
    _set_weather(world, weather_data)
    print("[INFO] Weather profile: {}".format(args.weather_profile))
    print("[INFO] Brightness profile: {} (strength {:.2f})".format(
        args.brightness_profile, _clamp_unit(args.brightness_strength)))
    if args.camera_brightness_profile != "none":
        print("[INFO] Camera brightness profile: {} (strength {:.2f})".format(
            args.camera_brightness_profile,
            _clamp_unit(args.camera_brightness_strength)))
    if args.camera_exposure_compensation is not None:
        print("[INFO] Camera exposure compensation override: {:.2f}".format(
            args.camera_exposure_compensation))
    if args.save_pylot_screenshots:
        print("[INFO] Pylot screenshots: enabled; path={}, every_nth={}".format(
            args.pylot_screenshot_path,
            max(1, args.pylot_screenshot_every_nth_message)))
        print("[INFO] Screenshot overlay mode: {}".format(
            args.screenshot_overlay_mode))
        print("[INFO] Visualizer HUD: {}".format(
            "hidden" if args.hide_visualizer_hud else "shown"))
    else:
        print("[INFO] Pylot screenshots: disabled")
    print("[INFO] Applied weather: precipitation={:.2f}, fog_density={:.2f}, "
          "fog_distance={:.2f}, fog_falloff={:.2f}, "
          "scattering_intensity={:.2f}, wetness={:.2f}, cloudiness={:.2f}, "
          "wind_intensity={:.2f}".format(
              float(weather_data.get("precipitation", 0.0)),
              float(weather_data.get("fog_density", 0.0)),
              float(weather_data.get("fog_distance", 0.0)),
              float(weather_data.get("fog_falloff", 0.0)),
              float(weather_data.get("scattering_intensity", 0.0)),
              float(weather_data.get("wetness", 0.0)),
              float(weather_data.get("cloudiness", 0.0)),
              float(weather_data.get("wind_intensity", 0.0))))
    _log_weather_readback(world, "after_set_weather")
    if not args.keep_existing_actors:
        _destroy_dynamic_actors(world)

    bp_lib = world.get_blueprint_library()
    actors_data = snapshot["actors"]
    ego_actor_id = snapshot.get("ego_vehicle", {}).get("actor_id")

    vehicle_id_map = {}
    spawned_vehicle_actors = []
    npc_initial_dynamics: Dict[int, Dict] = {}
    ego_initial_dynamic: Optional[Dict] = None
    npc_freeze_transforms: Dict[int, carla.Transform] = {}
    ego_actor = None
    spawned_vehicles = 0
    npc_autopilot_on_spawn = args.startup_mode == "immediate"
    npc_apply_initial_dynamics = args.startup_mode == "immediate"
    vehicles_data = actors_data.get("vehicles", [])
    # Always try ego vehicle first to maximize chance for hero restoration.
    vehicles_data = sorted(
        vehicles_data,
        key=lambda item: 0 if int(item["actor_id"]) == int(ego_actor_id) else 1
        if ego_actor_id is not None else 1)
    for vehicle_data in vehicles_data:
        actor, original_id = _spawn_vehicle(world, bp_lib, vehicle_data,
                                            ego_actor_id,
                                            args.traffic_manager_port,
                                            npc_autopilot_on_spawn,
                                            npc_apply_initial_dynamics)
        if actor is None:
            print("[WARN] Failed to spawn vehicle from original actor {}".format(
                original_id))
            continue
        vehicle_id_map[original_id] = actor.id
        spawned_vehicle_actors.append(actor)
        if ego_actor_id is not None and int(original_id) == int(ego_actor_id):
            ego_actor = actor
            ego_initial_dynamic = _extract_npc_initial_dynamic_state(
                vehicle_data)
        else:
            npc_initial_dynamics[actor.id] = \
                _extract_npc_initial_dynamic_state(vehicle_data)
            npc_freeze_transforms[actor.id] = _to_transform(
                vehicle_data["transform"])
        spawned_vehicles += 1

    if ego_actor_id is not None and ego_actor_id not in vehicle_id_map:
        fallback_ego = _spawn_ego_fallback(world, bp_lib,
                                           snapshot.get("ego_vehicle", {}),
                                           args.traffic_manager_port)
        if fallback_ego is not None:
            vehicle_id_map[int(ego_actor_id)] = fallback_ego.id
            spawned_vehicle_actors.append(fallback_ego)
            ego_actor = fallback_ego
            spawned_vehicles += 1
            print("[INFO] Ego restored using fallback spawn. "
                  "original_id={} new_id={}".format(ego_actor_id,
                                                    fallback_ego.id))
            fallback_ego_state = snapshot.get("ego_vehicle",
                                              {}).get("carla_actor_state", {})
            ego_initial_dynamic = _extract_npc_initial_dynamic_state(
                fallback_ego_state)

    spawned_walkers = 0
    walker_entries = []
    for walker_data in actors_data.get("walkers", []):
        actor, original_id = _spawn_walker(world, bp_lib, walker_data)
        if actor is None:
            print("[WARN] Failed to spawn walker from original actor {}".format(
                original_id))
            continue
        spawned_walkers += 1
        walker_entries.append((actor, walker_data))

    walker_states = _init_manual_walker_states(
        world, walker_entries, args.walker_keep_initial_direction_seconds,
        args.walker_speed_scale, args.walker_max_speed,
        settings_data.get("fixed_delta_seconds")
    ) if args.manual_walker_control else []

    npc_actors = []
    # Ensure freeze mode is effective before stabilization ticks.
    if args.startup_mode == "freeze_then_go":
        if ego_actor is not None:
            for actor in spawned_vehicle_actors:
                if actor.id != ego_actor.id:
                    npc_actors.append(actor)
        else:
            npc_actors = spawned_vehicle_actors
        _set_npc_physics_lock(npc_actors, locked=True)

    def _apply_startup_freeze_controls():
        if ego_actor is not None:
            _apply_vehicle_freeze(ego_actor, hand_brake=True)
        if args.freeze_npc_during_startup:
            for npc_actor in npc_actors:
                _apply_vehicle_freeze(npc_actor, hand_brake=False)
            _apply_npc_pose_lock(npc_actors, npc_freeze_transforms)
        if args.manual_walker_control:
            _apply_walker_freeze(walker_states)

    if args.startup_mode == "freeze_then_go":
        # Apply once immediately, then keep applying on each stabilization tick.
        _apply_startup_freeze_controls()
    active_manual_walkers = 0
    if args.manual_walker_control:
        active_manual_walkers = _update_manual_walker_controls(
            world, walker_states)

    _restore_traffic_lights(world, actors_data.get("traffic_lights", []))
    _tick_for_stability(
        world,
        args.stability_ticks,
        0.05,
        on_step=_apply_startup_freeze_controls
        if args.startup_mode == "freeze_then_go" else None)

    print("[INFO] Scene setup done.")
    print("[INFO] Spawned vehicles: {}".format(spawned_vehicles))
    print("[INFO] Spawned walkers: {}".format(spawned_walkers))
    print("[INFO] Manual walker mode active: {}".format(
        args.manual_walker_control))
    print("[INFO] Active manual walkers: {}".format(active_manual_walkers))
    if ego_actor_id in vehicle_id_map:
        print("[INFO] Ego restored. original_id={} new_id={}".format(
            ego_actor_id, vehicle_id_map[ego_actor_id]))
        if ego_actor is None:
            ego_actor = world.get_actors().find(vehicle_id_map[ego_actor_id])
        if ego_actor is not None:
            _focus_spectator_on_transform(world, ego_actor.get_transform())
    else:
        print("[WARN] Ego actor was not restored; Pylot may block waiting hero.")

    if args.keep_seconds > 0:
        print("[INFO] Keeping scene alive for {} seconds ...".format(
            args.keep_seconds))
        settings = world.get_settings()
        end_time = time.time() + args.keep_seconds
        while time.time() < end_time:
            if args.manual_walker_control:
                _update_manual_walker_controls(world, walker_states)
            if settings.synchronous_mode:
                world.tick()
            else:
                time.sleep(0.05)
            if ego_actor_id in vehicle_id_map:
                ego_actor = world.get_actors().find(vehicle_id_map[ego_actor_id])
                if ego_actor is not None:
                    _focus_spectator_on_transform(world, ego_actor.get_transform())

    if args.setup_only:
        return

    cmd = _build_pylot_command(args, snapshot)
    print("[INFO] Launching Pylot command:\n{}".format(" ".join(cmd)))
    # Run Pylot while continuously updating manual walker controls.
    process = subprocess.Popen(cmd)
    _log_weather_readback(world, "after_pylot_launch")
    try:
        settings = world.get_settings()
        last_frame = None
        last_async_update_ts = 0.0
        startup_released = args.startup_mode == "immediate"
        npc_autopilot_released = args.startup_mode == "immediate"
        npc_hold_active = False
        npc_hold_start_frame = None
        npc_hold_start_time = None
        npc_hold_duration_seconds = max(0.0, args.npc_snapshot_hold_seconds)
        npc_hold_duration_frames = 0
        if settings.synchronous_mode and npc_hold_duration_seconds > 0.0:
            npc_hold_duration_frames = int(
                math.ceil(npc_hold_duration_seconds /
                          max(settings.fixed_delta_seconds or 0.05, 1e-3)))
        freeze_start_frame = None
        freeze_duration_frames = 0
        if (not startup_released) and settings.synchronous_mode:
            freeze_duration_frames = max(
                0,
                int(max(0.0, args.startup_freeze_seconds) /
                    max(settings.fixed_delta_seconds or 0.05, 1e-3)))
        startup_release_at = None if startup_released or settings.synchronous_mode else \
            (time.monotonic() + max(0.0, args.startup_freeze_seconds))
        startup_logged = False
        release_frame = None
        npc_actor_states: Dict[int, Dict] = {}
        # Freeze immediately after subprocess launch to avoid startup window.
        if not startup_released:
            _apply_startup_freeze_controls()
        while process.poll() is None:
            frame_advanced = False
            if settings.synchronous_mode:
                # In synchronous mode, avoid flooding CARLA with control RPCs
                # before next world tick; update once per frame.
                try:
                    frame = world.get_snapshot().frame
                except RuntimeError:
                    time.sleep(0.02)
                    continue
                if frame != last_frame:
                    frame_advanced = True
                    last_frame = frame
            else:
                now = time.monotonic()
                if now - last_async_update_ts >= max(args.manual_walker_interval,
                                                     0.01):
                    frame_advanced = True
                    last_async_update_ts = now

            if frame_advanced and not startup_released:
                should_release = False
                if settings.synchronous_mode:
                    if freeze_start_frame is None and last_frame is not None:
                        freeze_start_frame = last_frame
                    if freeze_start_frame is not None and last_frame is not None:
                        if (last_frame - freeze_start_frame) >= freeze_duration_frames:
                            should_release = True
                else:
                    should_release = (time.monotonic() >= startup_release_at)
                if should_release:
                    startup_released = True
                    ego_reapplied = False
                    if ego_actor is not None and ego_initial_dynamic is not None:
                        ego_reapplied = _apply_vehicle_dynamic_state(
                            ego_actor,
                            dynamic_state=ego_initial_dynamic,
                            clear_freeze_control=True)
                    if npc_hold_duration_seconds > 0.0 and len(npc_actors) > 0:
                        _set_npc_physics_lock(npc_actors, locked=False)
                        held_npcs = 0
                        for actor in npc_actors:
                            if actor is None or not actor.is_alive:
                                continue
                            actor.set_autopilot(False, args.traffic_manager_port)
                            dynamic_state = npc_initial_dynamics.get(actor.id)
                            applied = _apply_vehicle_dynamic_state(
                                actor,
                                dynamic_state=dynamic_state,
                                clear_freeze_control=True)
                            if applied:
                                held_npcs += 1
                        npc_hold_active = True
                        npc_hold_start_frame = last_frame
                        npc_hold_start_time = time.monotonic()
                        released_npcs = 0
                        release_frame = None
                    else:
                        released_npcs = _release_npc_autopilot(
                            npc_actors,
                            args.traffic_manager_port,
                            npc_initial_dynamics=npc_initial_dynamics,
                            apply_snapshot_kick=(
                                not args.npc_release_without_snapshot_kick))
                        npc_autopilot_released = True
                        release_frame = last_frame
                    if not startup_logged:
                        if npc_hold_active:
                            print("[INFO] Startup freeze released; ego resumed. "
                                  "NPC snapshot hold active for {:.2f}s "
                                  "({} frames in sync mode), held NPCs with "
                                  "snapshot dynamics: {}; "
                                  "ego snapshot dynamics applied: {}".format(
                                      npc_hold_duration_seconds,
                                      npc_hold_duration_frames,
                                      held_npcs,
                                      ego_reapplied))
                        else:
                            print("[INFO] Startup freeze released; actors resume. "
                                  "NPC autopilot resumed: {}; "
                                  "ego snapshot dynamics applied: {}".format(
                                      released_npcs, ego_reapplied))
                        startup_logged = True

            if frame_advanced:
                if startup_released:
                    if npc_hold_active:
                        for actor in npc_actors:
                            if actor is None or not actor.is_alive:
                                continue
                            dynamic_state = npc_initial_dynamics.get(actor.id)
                            _apply_vehicle_dynamic_state(
                                actor,
                                dynamic_state=dynamic_state,
                                clear_freeze_control=False)
                        hold_finished = False
                        if settings.synchronous_mode:
                            if npc_hold_start_frame is not None and \
                                    last_frame is not None and \
                                    (last_frame - npc_hold_start_frame) >= \
                                    npc_hold_duration_frames:
                                hold_finished = True
                        else:
                            hold_finished = (npc_hold_start_time is not None and
                                             (time.monotonic() - npc_hold_start_time) >=
                                             npc_hold_duration_seconds)
                        if hold_finished:
                            released_npcs = _release_npc_autopilot(
                                npc_actors,
                                args.traffic_manager_port,
                                npc_initial_dynamics=npc_initial_dynamics,
                                apply_snapshot_kick=(
                                    not args.npc_release_without_snapshot_kick))
                            npc_hold_active = False
                            npc_autopilot_released = True
                            release_frame = last_frame
                            print("[INFO] NPC snapshot hold finished; "
                                  "autopilot resumed for {} NPCs.".format(
                                      released_npcs))
                    if settings.synchronous_mode and \
                            npc_autopilot_released and \
                            release_frame is not None:
                        monitor_frames = max(
                            0,
                            int(args.npc_revival_monitor_seconds /
                                max(settings.fixed_delta_seconds or 0.05, 1e-3)))
                        grace_frames = max(
                            0,
                            int(args.npc_revival_grace_seconds /
                                max(settings.fixed_delta_seconds or 0.05, 1e-3)))
                        if (last_frame - release_frame) <= monitor_frames and \
                                (last_frame - release_frame) >= grace_frames and \
                                args.npc_revival_check_interval_frames > 0 and \
                                ((last_frame - release_frame) %
                                 args.npc_revival_check_interval_frames == 0):
                            revived_npcs = _revive_stalled_npcs(
                                npc_actors,
                                args.traffic_manager_port,
                                min_speed_mps=args.npc_stall_min_speed_mps,
                                actor_states=npc_actor_states,
                                current_frame=last_frame,
                                max_attempts=args.npc_revival_max_attempts,
                                cooldown_frames=args.npc_revival_cooldown_frames)
                            if revived_npcs > 0:
                                print("[INFO] Revived stalled NPCs: {}".format(
                                    revived_npcs))
                    if args.manual_walker_control:
                        _update_manual_walker_controls(world, walker_states)
                else:
                    _apply_startup_freeze_controls()
            time.sleep(0.01)
    except KeyboardInterrupt:
        process.terminate()
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as exc:
        print("[ERROR] {}".format(exc))
        sys.exit(1)
