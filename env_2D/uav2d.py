import os
import random
import time
import math

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from astar_planner import plan_path
from matplotlib.patches import Rectangle


class UAV2D(gym.Env):
    """2D空间中的无人机仿真环境。"""

    def __init__(self, config: dict = None):
        super(UAV2D, self).__init__()
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self.config = config or {}

        # Parameters
        self.grid_size = self.config.get("grid_size", 100)
        self.uav_size = self.config.get("uav_size", 1)
        self.uav_view_range = self.config.get("uav_view_range", 5)
        self.num_buildings = self.config.get("num_buildings", 10)
        self.buildings_min_size = self.config.get("buildings_min_size", 3)
        self.buildings_max_size = self.config.get("buildings_max_size", 5)
        self.num_dynamic_obstacles = self.config.get("num_dynamic_obstacles", 3)
        self.max_speed = self.config.get("max_speed", 2)
        self.min_speed = self.config.get("min_speed", 1)
        self.max_steps = self.config.get("max_steps", 200)
        self.random_goal = self.config.get("random_start_goal", True)

        # Fields
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.buildings, self.buildings_len, self.buildings_wid = (
            self.generate_buildings()
        )
        self.dynamic_obstacles = self.generate_dynamic_obstacles()
        self.uav_position = self.config.get("uav_pos", np.array([0, 0]))
        self.uav_orientation = np.array([1, 0])
        self.target_position = self.config.get("target_pos", np.array([1, 1]))
        self.path = []

        state = self.get_state()
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32
        )

        # Episodic values
        self.step_cnt = 0
        self.save = 0

        # Rewards
        self.colli = 0
        self.reach = 0
        self.time_penalty = 0
        self.distance_reward = 0
        self.obstacle_reward = 0
        self.angle_reward = 0

        self.time_penalty_list = []
        self.distance_reward_list = []
        self.obstacle_reward_list = []
        self.closest_obs = self.grid_size
        self.angle_reward_list = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #  重置奖励曲线
        self.colli = 0
        self.reach = 0
        self.time_penalty = 0
        self.distance_reward = 0
        self.obstacle_reward = 0
        self.angle_reward = 0
        self.time_penalty_list = []
        self.distance_reward_list = []
        self.obstacle_reward_list = []
        self.closest_obs = self.grid_size
        self.angle_reward_list = []

        self.step_cnt = 0
        self.path.clear()
        self.uav_orientation = np.array([random.randint(0, 1), random.randint(0, 1)])

        self.dynamic_obstacles = self.generate_dynamic_obstacles()
        self.buildings, self.buildings_len, self.buildings_wid = (
            self.generate_buildings()
        )

        # 定义坐标轴范围
        all_coords = set(
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        )
        occupied_coords = set()
        for i, (x, y) in enumerate(self.buildings):
            length = self.buildings_len[i]
            width = self.buildings_wid[i]
            # 添加建筑物区域内的所有坐标
            for dx in range(length + self.uav_size * 4):
                for dy in range(width + self.uav_size * 4):
                    occupied_coords.add(
                        (x - self.uav_size * 2 + dx, y - self.uav_size * 2 + dy)
                    )
        for obs in self.dynamic_obstacles:
            a1, a2 = obs["pos"]
            occupied_coords.add((a1, a2))

        if self.random_goal:
            existing_coords = list(occupied_coords)
            # 从所有坐标中排除现有坐标
            valid_coords = all_coords - set(existing_coords)

            # 从剩余的坐标中随机选择一个
            if valid_coords:
                dist, attempt = 0, 100
                while dist < self.grid_size // 2 and attempt > 0:
                    ran1 = random.choice(list(valid_coords))
                    ran2 = np.array(random.choice(list(valid_coords)))
                    self.uav_position = np.array([ran1[0], ran1[1]])
                    self.target_position = np.array([ran2[0], ran2[1]])
                    dist = np.linalg.norm(self.uav_position - self.target_position)

                if attempt <= 0:
                    return self.reset()
            else:
                raise ValueError("没有可用的坐标!")
        else:
            pass

        self.path.append(self.uav_position.copy())
        self.total_distance = np.linalg.norm(self.uav_position - self.target_position)
        self.closest_distance = self.total_distance

        self.pre_uav = self.uav_position
        self.pre_pre_uav = self.pre_uav

        # 全局地图
        # 将建筑物标记为障碍物
        pos = 0
        for x, y in self.buildings:
            # 计算索引范围，确保它们不会超出 grid_size
            start_x = max(0, x - self.uav_size)
            end_x = min(self.grid_size, x + self.buildings_len[pos] + self.uav_size)

            start_y = max(0, y - self.uav_size)
            end_y = min(self.grid_size, y + self.buildings_wid[pos] + self.uav_size)

            self.grid[start_x:end_x, start_y:end_y] = 1.0

            pos += 1

        return self.get_state(), {}

    def step(self, action):
        action_choice = [
            [1, 0],
            [-1, 0],
            [0, -1],
            [0, 1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]
        action = action_choice[action]
        last_position = self.uav_position.copy()
        self.uav_position += action
        self.uav_orientation = self.uav_position - last_position
        self.path.append(self.uav_position.copy())

        # 更新动态障碍物的位置和方向
        for obs in self.dynamic_obstacles:
            if np.random.rand() < 0.05:  # 5%的概率改变方向
                dist_to_uav = self.uav_position - obs["pos"]
                dist_to_goal = self.target_position - obs["pos"]
                dist_to = np.sign(dist_to_uav + dist_to_goal)
                alpha = np.random.choice([0, 1], size=2)
                obs["vel"] = alpha * dist_to

            obs["vel"] = [int(obs["vel"][0]), int(obs["vel"][1])]

            obs["pos"] += obs["vel"]
            obs["path"].append(obs["pos"].copy())

        # 检查是否碰撞
        collision = self.check_collision()

        # 检查是否到达目标点
        goal_reached = np.linalg.norm(self.uav_position - self.target_position) <= 2

        done = collision or goal_reached

        reward = self.compute_reward(goal_reached, collision)

        self.step_cnt += 1

        # 检查是否越界
        if ((self.uav_position < 0).any()) or (
            (self.uav_position >= self.grid_size).any()
        ):
            self.colli = -100
            return (
                self.get_state(),
                reward - 100,
                True,
                False,
                {"info": "UAV out of bounds"},
            )

        return (
            self.get_state(),
            reward,
            done,
            self.step_cnt >= self.max_steps,
            {"collision": collision, "goal_reached": goal_reached},
        )

    def get_distance_to_obstacles(self):
        """计算到可见障碍物的距离。"""
        distances = []
        # dynamic_pos=0
        # building_pos=0
        pos = []
        # min_pos = []
        for obs in self.dynamic_obstacles:
            distance = np.linalg.norm(obs["pos"] - self.uav_position)
            if (
                distance <= 5 and np.abs(obs["pos"][1] - self.uav_position[1]) <= 5
            ):  # 视野半径为5个单位
                distances.append(distance)
                pos.append(obs["pos"])
            # dynamic_pos+=1

        for ob in self.buildings:
            distance = np.linalg.norm(ob - self.uav_position)
            if distance <= 5 and np.abs(ob[1] - self.uav_position[1]) <= 5:
                distances.append(distance)
                pos.append(ob)

        # Ensure we always return 3 distances (pad with large values if necessary)
        while len(distances) < 3:
            distances.append(1000.0)  # A large value to represent "no obstacle"
            pos.append([0, 0])
        sorted_with_indices = sorted(enumerate(distances), key=lambda x: x[1])

        # 提取最小的三个数字的下标
        min_three_indices = [idx for idx, _ in sorted_with_indices[:3]]
        min_pos1 = pos[min_three_indices[0]]
        min_pos2 = pos[min_three_indices[1]]
        return np.array(min_pos1), np.array(min_pos2)

    def get_state(self):
        """获取环境的当前状态。"""
        s0 = self.uav_position / self.grid_size
        s1 = self.target_position / self.grid_size
        s2, s3 = self.get_distance_to_obstacles()
        s2 = s2 / self.grid_size
        s3 = s3 / self.grid_size
        s4 = (s2 - self.uav_position) / self.grid_size
        s5 = (s3 - self.uav_position) / self.grid_size
        s6 = (self.target_position - self.uav_position) / self.grid_size
        return np.concatenate((s0, s1, s2, s3, s4, s5, s6)).astype(np.float32)

    def get_local_map(self, map):
        view_range = self.uav_view_range

        # Calculate the start and end indices for each dimension
        start_x = self.uav_position[0] - view_range // 2
        start_y = self.uav_position[1] - view_range // 2

        # Create a 5x5 array filled with 1s (not movable)
        local_map = np.ones((view_range, view_range), dtype=map.dtype)

        for i in range(view_range):
            for j in range(view_range):
                x = start_x + i
                y = start_y + j

                # Check if the current position is within the full map bounds
                if (0 <= x < self.grid_size) and (0 <= y < self.grid_size):
                    local_map[i, j] = map[x, y]

        return local_map

    def check_collision(self):
        # 检查与静态建筑物的碰撞
        check = 0
        pos = 0
        for x, y in self.buildings:
            if (
                (x - self.uav_size)
                <= self.uav_position[0]
                <= (x + self.buildings_len[pos] + self.uav_size)
            ) and (
                (y - self.uav_size)
                <= self.uav_position[1]
                <= (y + self.buildings_wid[pos] + self.uav_size)
            ):
                check += 1
            pos += 1
        # 检查与动态障碍物的碰撞
        for obs in self.dynamic_obstacles:
            # print('obsssss',obs)
            if (
                self.uav_position[0] == obs["pos"][0]
                and self.uav_position[1] == obs["pos"][1]
            ):
                check += 1
                # return True
        if check > 0:
            return True
        else:
            return False

    def calculate_angle(self, vec1, vec2):
        # 检查是否有零向量
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0, "无法计算旋转角度（至少一个向量为零向量）"

        # 计算向量的模长
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # 计算点积（用于计算角度）
        dot_product = np.dot(vec1, vec2)

        # 计算角度（弧度制）
        angle_rad = np.arccos(dot_product / (norm_vec1 * norm_vec2))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def compute_reward(self, goal_reached, collision):
        colli, reach = 0, 0
        # 碰撞惩罚
        if collision:
            colli = -100
        # 到达奖励
        if goal_reached:
            reach = 300

        current_dist_to_goal = np.linalg.norm(self.uav_position - self.target_position)

        # 时间惩罚
        time_penalty = -0.1

        # 距离奖励（势函数）
        # 目标
        distance_reward = 0
        obstacle_reward = 0
        dis = self.closest_distance - current_dist_to_goal  # 靠近是负的
        if dis == 0:
            pass
        else:
            distance_reward = ((dis) / abs(dis)) * math.exp(dis)
        # 障碍

        dis_obs_pos = self.get_distance_to_obstacles()
        dis_obs = np.linalg.norm(self.uav_position - dis_obs_pos)

        if dis_obs < self.uav_view_range:
            if self.closest_obs < self.uav_view_range:
                dis_o = dis_obs - self.closest_obs
                if dis_o == 0:
                    pass
                else:
                    obstacle_reward = ((dis_o) / abs(dis_o)) * math.exp(dis_o)
        else:
            obstacle_reward = distance_reward

        self.closest_obs = dis_obs
        self.clo_closest_distance = self.closest_distance
        self.closest_distance = current_dist_to_goal

        # 转角惩罚
        angle_reward = 0
        angle = self.uav_position - self.pre_uav
        pre_angle = self.pre_uav - self.pre_pre_uav
        if self.calculate_angle(pre_angle, angle) == 0 or 0.0:
            angle_reward = 1
        else:
            angle_reward = -0.1

        v_reward = 0
        self.colli = colli
        self.reach = reach
        self.time_penalty += time_penalty
        self.time_penalty_list.append(self.time_penalty)
        self.distance_reward += distance_reward
        self.obstacle_reward += obstacle_reward
        self.distance_reward_list.append(self.distance_reward)
        self.obstacle_reward_list.append(obstacle_reward)
        self.angle_reward += angle_reward
        self.angle_reward_list.append(self.angle_reward)

        return (
            colli
            + reach
            + time_penalty
            + distance_reward
            + angle_reward
            + v_reward
            + obstacle_reward
        )

    def generate_buildings(self):
        """生成建筑物（障碍物）的随机位置。"""
        buildings = []
        buildings_len = []
        buildings_wid = []
        for _ in range(self.num_buildings):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            length = random.randint(self.buildings_min_size, self.buildings_max_size)
            width = random.randint(self.buildings_min_size, self.buildings_max_size)
            buildings.append((x, y))
            buildings_len.append(length)
            buildings_wid.append(width)
        return buildings, buildings_len, buildings_wid

    def generate_dynamic_obstacles(self):
        """生成动态障碍物的位置和速度。"""
        obstacles = []
        for _ in range(self.num_dynamic_obstacles):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            velocity = np.array(
                [
                    random.uniform(-self.max_speed, self.max_speed),
                    random.uniform(-self.max_speed, self.max_speed),
                ]
            )
            obstacles.append({"pos": np.array([x, y]), "vel": velocity, "path": []})
        return obstacles

    def visualize(self):
        """可视化环境。"""
        fig = plt.figure()
        ax = fig.add_subplot()
        self.save += 1
        # self.phi = 0
        # 获取起点和终点
        path = np.array(self.path)
        start_point = path[0]
        end_point = path[-1]
        # 固定坐标轴范围在起点和终点之间
        ax.set_xlim(
            min(start_point[0], end_point[0]) - 5, max(start_point[0], end_point[0]) + 5
        )
        ax.set_ylim(
            min(start_point[1], end_point[1]) - 5, max(start_point[1], end_point[1]) + 5
        )

        def update(num):
            ax.clear()  # 清除前一帧的数据，避免内存累积
            ax.set_title(f"Step: {num}")

            buildings_pos = 0
            # 绘制静态障碍物
            for ob in self.buildings:
                # 创建矩形
                rect = Rectangle(
                    (ob[0], ob[1]),
                    self.buildings_len[buildings_pos],
                    self.buildings_wid[buildings_pos],
                    fill=True,
                    edgecolor="black",
                    facecolor="black",
                    linewidth=2,
                )
                # 将矩形添加到坐标轴上
                ax.add_patch(rect)
                buildings_pos += 1

            # 绘制动态障碍物
            for obs in self.dynamic_obstacles:
                pos = np.array(obs["path"])
                if len(pos) > 0:
                    end_idx = min(num, len(pos) - 1)
                    ax.plot(pos[: end_idx + 1, 0], pos[: end_idx + 1, 1], "yellow")
                    ax.scatter(pos[end_idx, 0], pos[end_idx, 1], color="yellow")

            # 绘制无人机路径
            path = np.array(self.path)
            if num < len(path):
                ax.plot(path[: num + 1, 0], path[: num + 1, 1], "blue")
                ax.scatter(path[num, 0], path[num, 1], color="red")
            ax.scatter(self.target_position[0], self.target_position[1], color="green")

        ani = animation.FuncAnimation(fig, update, frames=len(self.path), repeat=False)
        folder_path = os.path.join(os.path.dirname(__file__), self.timestamp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        full_file_path = os.path.join(folder_path, str(self.save))
        ani.save(full_file_path + ".gif", writer="pillow")
        plt.close(fig)
        print(f"动画已保存到 {full_file_path}")

    def analyze(self):
        """分析当前回合的飞行数据并生成报告"""
        path = np.array(self.path)
        steps = len(path)

        # 1. 计算与障碍物的最小距离曲线
        min_obstacle_distances = []
        for position in path:
            distances = []
            # 检查与静态建筑物的距离
            for building in self.buildings:
                dist = np.linalg.norm(position[:1] - building[:1])  # 只考虑xy平面的距离

            # 检查与动态障碍物的距离
            for obs in self.dynamic_obstacles:
                dist = np.linalg.norm(
                    position
                    - obs["path"][
                        min(len(obs["path"]) - 1, len(min_obstacle_distances))
                    ]
                )
                distances.append(dist)
            min_obstacle_distances.append(min(distances) if distances else float("inf"))

        # 2. 计算与终点的距离曲线
        target_distances = [np.linalg.norm(pos - self.target_position) for pos in path]

        # 3. 计算飞行平稳度（使用速度变化）
        velocities = np.linalg.norm(np.diff(path, axis=0), axis=1)
        smoothness = np.std(velocities)  # 速度标准差作为平稳度指标

        # 4. 计算速度曲线
        speed_curve = np.concatenate(([0], velocities))  # 添加初始速度0

        # 绘图
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        # 障碍物距离图
        axs[0, 0].plot(min_obstacle_distances)
        axs[0, 0].set_title("Minimum Distance to Obstacles")
        axs[0, 0].set_xlabel("Step")
        axs[0, 0].set_ylabel("Distance")

        # 终点距离图
        axs[0, 1].plot(target_distances)
        axs[0, 1].set_title("Distance to Target")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("Distance")

        # 速度曲线
        axs[1, 0].plot(speed_curve)
        axs[1, 0].set_title("Speed Curve")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Speed")

        # 3D路径图
        axs[1, 1].plot(path[:, 0], path[:, 1])
        axs[1, 1].scatter(
            self.target_position[0],
            self.target_position[1],
            c="r",
            marker="*",
            s=200,
        )
        axs[1, 1].set_title("3D Flight Path")
        axs[1, 1].set_xlabel("X")
        axs[1, 1].set_ylabel("Y")

        plt.tight_layout()
        folder_path = os.path.join(os.path.dirname(__file__), self.timestamp)
        file_path = os.path.join(folder_path, f"flight_analysis_{self.save}.png")
        plt.savefig(file_path)
        plt.close()

        plt.figure()
        plt.plot(
            self.distance_reward_list,
            linestyle="-",
            color="red",
            label="distance_reward",
        )
        plt.plot(
            self.time_penalty_list, linestyle="--", color="blue", label="time_penalty"
        )
        plt.plot(
            self.obstacle_reward_list,
            linestyle=":",
            color="green",
            label="obstacle_reward",
        )
        plt.plot(
            self.angle_reward_list,
            linestyle=":",
            color="black",
            label="angle_reward",
        )
        plt.scatter(
            len(self.distance_reward_list),
            self.colli,
            label="colli:{}".format(self.colli),
        )
        plt.scatter(
            len(self.distance_reward_list),
            self.reach,
            label="reach:{}".format(self.reach),
        )
        plt.legend()
        file_path = os.path.join(folder_path, f"reward_analysis_{self.save}.png")
        plt.savefig(file_path)
        plt.close()

        # 打印分析结果
        print(f"Flight Analysis for Episode {self.save}:")
        print(f"Total steps: {steps}")
        print(f"Final distance to target: {target_distances[-1]:.2f}")
        print(f"Minimum distance to obstacles: {min(min_obstacle_distances):.2f}")
        print(f"Average speed: {np.mean(speed_curve):.2f}")
        print(f"Flight smoothness (lower is smoother): {smoothness:.2f}")
        print(f"Analysis graph saved as 'flight_analysis_{self.save}.png'")
        print(
            f"reward_analyse,distance_reward:{self.distance_reward},time_penalty:{self.time_penalty}"
        )


# Example of usage
if __name__ == "__main__":
    config = {
        "grid_size": 50,
        "uav_size": 1,
        "uav_view_range": 5,
        "num_buildings": 5,
        "buildings_min_size": 2,
        "buildings_max_size": 4,
        "num_dynamic_obstacles": 3,
        "max_speed": 2,
        "min_speed": 1,
        "max_steps": 500,
        "random_start_goal": True,
    }
    env = UAV2D(config)
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Sample random action
        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)
        env.visualize()
        env.analyze()
