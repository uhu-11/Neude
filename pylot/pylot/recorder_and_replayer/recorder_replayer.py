import pickle
import carla
import erdos
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.traffic_light import TrafficLight, TrafficLightColor
from pylot.perception.detection.speed_limit_sign import SpeedLimitSign
from pylot.perception.detection.stop_sign import StopSign
from pylot.perception.messages import (ObstaclesMessage, TrafficLightsMessage,
                                     SpeedSignsMessage, StopSignsMessage)
from pylot.perception.detection.utils import BoundingBox2D, BoundingBox3D
import pylot.utils
from carla.libcarla import WeatherParameters

class SceneDataRecorder:
    def __init__(self, world, save_path: str = "recorded_scenes.pkl"):
        """初始化场景数据记录器
        
        Args:
            save_path: 保存记录数据的文件路径
        """
        self.save_path = save_path
        self.recorded_scenes = {}
        self.start_timestamp: Optional[erdos.Timestamp] = None
        self.world = world
        
    def record_scene(self, timestamp: erdos.Timestamp, vehicles, people, traffic_lights, speed_limits, traffic_stops) -> None:
        """记录一个时间戳的场景数据"""
        # 设置起始时间戳
        if self.start_timestamp is None:
            self.start_timestamp = timestamp.coordinates[0]
            
        # 计算相对时间（毫秒）
        relative_time = timestamp.coordinates[0] - self.start_timestamp

        print(f'记录时的{timestamp}第一个车辆信息:',vehicles[0])
        # 创建序列化的场景数据
        weather = self.world.get_weather()
        serialized_weather = {
            'cloudiness': weather.cloudiness,
            'precipitation': weather.precipitation,
            'precipitation_deposits': weather.precipitation_deposits,
            'wind_intensity': weather.wind_intensity,
            'sun_azimuth_angle': weather.sun_azimuth_angle,
            'sun_altitude_angle': weather.sun_altitude_angle,
        }
        scene = {
            'vehicles': vehicles,
            'people': people,
            'traffic_lights': traffic_lights,
            'speed_limits': speed_limits,
            'traffic_stops': traffic_stops,
            'weather': serialized_weather,
            'relative_time': relative_time
        }
        print(f'记录时的{timestamp, relative_time}第一个转换后的车辆信息:',scene['vehicles'][0])
        
        # 存储场景数据
        self.recorded_scenes[relative_time] = scene
        
        # 保存当前场景数据到文件
        self.save_recorded_scenes()
        
    
    def save_recorded_scenes(self):
        """保存记录的场景数据到文件"""
        print('保存记录的场景数据到文件.',len(self.recorded_scenes))
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.recorded_scenes, f)


class SceneDataReplayer:
    def __init__(self, world, record_file_path: str = "recorded_scenes.pkl"):
        """初始化场景数据回放器
        
        Args:
            record_file_path: 记录数据的文件路径
        """
        self.world = world
        self.record_file_path = record_file_path
        self.start_timestamp: Optional[erdos.Timestamp] = None
        
        # 加载记录的场景数据
        with open(self.record_file_path, 'rb') as f:
            self.scenes: Dict[int, Dict] = pickle.load(f)
        
        # print('dadwadscadcswadadasssssssssss:',self.scenes[0]['vehicles'][0])

    def get_scene_data(self, current_timestamp: erdos.Timestamp, seed_ind) -> Optional[Tuple]:
        """获取当前时间戳对应的场景数据"""
        if self.start_timestamp is None:
            self.start_timestamp = current_timestamp.coordinates[0]
            
        relative_time = current_timestamp.coordinates[0] + seed_ind*50 - self.start_timestamp
        print('relative_time:',relative_time)
        # print('timestamps:',self.scenes.keys())
        
        nearest_time = self._find_nearest_timestamp(relative_time)
        if nearest_time is None:
            return None
            
        scene = self.scenes[nearest_time]
        
        weather_data = scene['weather']
        weather = WeatherParameters(
            cloudiness=weather_data['cloudiness'],
            precipitation=weather_data['precipitation'],
            precipitation_deposits=weather_data['precipitation_deposits'],
            wind_intensity=weather_data['wind_intensity'],
            sun_azimuth_angle=weather_data['sun_azimuth_angle'],
            sun_altitude_angle=weather_data['sun_altitude_angle'],
        )
        self.world.set_weather(weather)

        # print('重放时的第一个车辆信息:',scene['vehicles'][0])
        return scene['vehicles'], scene['people'], scene['traffic_lights'], scene['speed_limits'], scene['traffic_stops']

    def _find_nearest_timestamp(self, target_time: float) -> Optional[float]:
        """找到最接近目标时间的记录时间戳"""
        if not self.scenes:
            return None
        times = list(self.scenes.keys())
        return min(times, key=lambda x: abs(x - target_time))


class HeroVehicleDataRecorder:
    def __init__(self, save_path: str = "hero_vehicle_data_recorded_scenes.pkl"):
        """初始化场景数据记录器
        
        Args:
            save_path: 保存记录数据的文件路径
        """
        self.save_path = save_path
        self.recorded_scenes = {}
        self.start_timestamp: Optional[erdos.Timestamp] = None

        
    def record_scene(self, timestamp: erdos.Timestamp, data) -> None:
        """记录一个时间戳的场景数据"""
        # 设置起始时间戳
        if self.start_timestamp is None:
            self.start_timestamp = timestamp.coordinates[0]
            
        # 计算相对时间（毫秒）
        relative_time = timestamp.coordinates[0] - self.start_timestamp

        # print(f'记录时的{timestamp}第一个车辆信息:',vehicles[0])
        # 创建序列化的场景数据
        
        # 存储场景数据
        self.recorded_scenes[relative_time] = data
        
        # 保存当前场景数据到文件
        self.save_recorded_scenes()
        
    
    def save_recorded_scenes(self):
        """保存记录的场景数据到文件"""
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.recorded_scenes, f)


class HeroVehicleDataReplayer:
    def __init__(self, record_file_path: str = "hero_vehicle_data_recorded_scenes.pkl"):
        """初始化场景数据回放器
        
        Args:
            record_file_path: 记录数据的文件路径
        """
        self.record_file_path = record_file_path
        self.start_timestamp: Optional[erdos.Timestamp] = None
        
        # 加载记录的场景数据
        with open(self.record_file_path, 'rb') as f:
            self.scenes: Dict[int, Dict] = pickle.load(f)
        
        # print('dadwadscadcswadadasssssssssss:',self.scenes[0]['vehicles'][0])

    def get_scene_data(self, current_timestamp: erdos.Timestamp, seed_ind):
        """获取当前时间戳对应的场景数据"""
        if self.start_timestamp is None:
            self.start_timestamp = current_timestamp.coordinates[0]
            
        relative_time = current_timestamp.coordinates[0] - self.start_timestamp + seed_ind*50
        # print('timestamps:',self.scenes.keys())
        
        nearest_time = self._find_nearest_timestamp(relative_time)
        if nearest_time is None:
            return None
            
        scene = self.scenes[nearest_time]
        # print('重放时的第一个车辆信息:',scene['vehicles'][0])
        return scene

    def _find_nearest_timestamp(self, target_time: float):
        """找到最接近目标时间的记录时间戳"""
        if not self.scenes:
            return None
        times = list(self.scenes.keys())
        return min(times, key=lambda x: abs(x - target_time))


class WorldDataRecorder:
    def __init__(self, save_path: str = "world_data_recorded_scenes.pkl"):
        """初始化场景数据记录器
        
        Args:
            save_path: 保存记录数据的文件路径
        """
        self.save_path = save_path


        
    def record_scene(self, data) -> None:
        """记录一个时间戳的场景数据"""
        # 设置起始时间戳
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
        



class WorldDataReplayer:
    def __init__(self, record_file_path: str = "world_data_recorded_scenes.pkl"):
        """初始化场景数据回放器
        
        Args:
            record_file_path: 记录数据的文件路径
        """
        self.record_file_path = record_file_path

        
        # 加载记录的场景数据
        with open(self.record_file_path, 'rb') as f:
            self.scenes = pickle.load(f)

        # print('dadwadscadcswadadasssssssssss:',self.scenes[0]['vehicles'][0])

    def get_scene_data(self):
        """获取当前时间戳对应的场景数据"""
        return self.scenes



class PointCloudRecorder:
    def __init__(self, save_path: str = "point_cloud_data_recorded_scenes.pkl"):
        """初始化场景数据记录器
        
        Args:
            save_path: 保存记录数据的文件路径
        """
        self.save_path = save_path
        self.recorded_scenes = {}
        self.start_timestamp: Optional[erdos.Timestamp] = None

        
    def record_scene(self, timestamp: erdos.Timestamp, data) -> None:
        """记录一个时间戳的场景数据"""
        # 设置起始时间戳
        if self.start_timestamp is None:
            self.start_timestamp = timestamp.coordinates[0]
            
        # 计算相对时间（毫秒）
        relative_time = timestamp.coordinates[0] - self.start_timestamp

        # print(f'记录时的{timestamp}第一个车辆信息:',vehicles[0])
        # 创建序列化的场景数据
        
        # 存储场景数据
        self.recorded_scenes[relative_time] = data
        
        # 保存当前场景数据到文件
        self.save_recorded_scenes()
        
    
    def save_recorded_scenes(self):
        """保存记录的场景数据到文件"""
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.recorded_scenes, f)


class PointCloudReplayer:
    def __init__(self, record_file_path: str = "point_cloud_data_recorded_scenes.pkl"):
        """初始化场景数据回放器
        
        Args:
            record_file_path: 记录数据的文件路径
        """
        self.record_file_path = record_file_path
        self.start_timestamp: Optional[erdos.Timestamp] = None
        
        # 加载记录的场景数据
        with open(self.record_file_path, 'rb') as f:
            self.scenes: Dict[int, Dict] = pickle.load(f)
        
        # print('dadwadscadcswadadasssssssssss:',self.scenes[0]['vehicles'][0])

    def get_scene_data(self, current_timestamp: erdos.Timestamp, seed_ind):
        """获取当前时间戳对应的场景数据"""
        if self.start_timestamp is None:
            self.start_timestamp = current_timestamp.coordinates[0]
            
        relative_time = current_timestamp.coordinates[0] - self.start_timestamp + seed_ind*50
        # print('timestamps:',self.scenes.keys())
        
        nearest_time = self._find_nearest_timestamp(relative_time)
        if nearest_time is None:
            return None
            
        scene = self.scenes[nearest_time]
        # print('重放时的第一个车辆信息:',scene['vehicles'][0])
        return scene

    def _find_nearest_timestamp(self, target_time: float):
        """找到最接近目标时间的记录时间戳"""
        if not self.scenes:
            return None
        times = list(self.scenes.keys())
        return min(times, key=lambda x: abs(x - target_time))



'''

记录时的[5788]第一个车辆信息: 
Obstacle(id: 9707, label: vehicle, confidence: 1.0, bbox: BoundingBox3D(transform: Transform(location: Location(x=0.0040429686196148396, y=2.115964825577521e-08, z=0.7188605070114136), rotation: Rotation(pitch=0, yaw=0, roll=0)), extent: Vector3D(x=2.4508416652679443, y=1.0641621351242065, z=0.7553732395172119))) at Transform(location: Location(x=92.10997772216797, y=105.66152954101562, z=0.0908435583114624), rotation: Rotation(pitch=0.0, yaw=-90.00029754638672, roll=0.0))
记录时的([5788], 0)第一个转换后的车辆信息: 
Obstacle(id: 9707, label: vehicle, confidence: 1.0, bbox: BoundingBox3D(transform: Transform(location: Location(x=0.0040429686196148396, y=2.115964825577521e-08, z=0.7188605070114136), rotation: Rotation(pitch=0, yaw=0, roll=0)), extent: Vector3D(x=2.4508416652679443, y=1.0641621351242065, z=0.7553732395172119))) at Transform(location: Location(x=92.10997772216797, y=105.66152954101562, z=0.0908435583114624), rotation: Rotation(pitch=0.0, yaw=-90.00029754638672, roll=0.0))
重放时的第一个车辆信息: 
Obstacle(id: 9707, label: vehicle, confidence: 1.0, bbox: BoundingBox3D(transform: Transform(location: Location(x=0.0040429686196148396, y=2.115964825577521e-08, z=0.7188605070114136), rotation: Rotation(pitch=0, yaw=0, roll=0)), extent: Vector3D(x=2.4508416652679443, y=1.0641621351242065, z=0.7553732395172119))) at Transform(location: Location(x=92.10997772216797, y=105.66152954101562, z=0.0908435583114624), rotation: Rotation(pitch=0.0, yaw=-90.00029754638672, roll=0.0))

'''

