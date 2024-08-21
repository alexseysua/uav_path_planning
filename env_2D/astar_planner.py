import heapq
import numpy as np


def astar(grid, start, goal):
    """A*路径规划算法

    参数:
    grid -- 3D栅格地图, 通常由0和1组成, 其中1代表障碍物
    start -- 起点坐标, 格式为(x, y, z)
    goal -- 终点坐标, 格式为(x, y, z)

    返回:
    path -- 从起点到终点的路径, 如果找不到路径则返回空列表
    """
    # 获取栅格地图的大小 (行数, 列数, 高度)
    rows, cols, heights = grid.shape

    # 定义3D空间中可能的移动方向
    directions = []
    for dx in [-1, 0, 1]:  # x方向上的偏移量
        for dy in [-1, 0, 1]:  # y方向上的偏移量
            for dz in [-1, 0, 1]:  # z方向上的偏移量
                # 排除当前位置 (0, 0, 0)
                if not (dx == 0 and dy == 0 and dz == 0):
                    directions.append([dx, dy, dz])

    def heuristic(a, b):
        """启发函数, 使用欧氏距离来估算从当前点到目标点的代价"""
        return np.linalg.norm(np.array(a) - np.array(b))

    # 创建优先队列 (open_list) 并将起点添加进去, 格式为 (f值, g值, 节点)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))

    # 记录从起点到达每个节点的最优路径
    came_from = {}

    # 记录从起点到达每个节点的实际代价
    g_score = {start: 0}

    # 当open_list非空时, 继续搜索
    while open_list:
        # 从open_list中取出f值最小的节点
        _, current_g, current = heapq.heappop(open_list)

        # 如果当前节点是目标节点, 回溯路径并返回
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()  # 反转路径以得到从起点到终点的顺序
            return path

        # 考虑当前节点的每个相邻节点
        for direction in directions:
            neighbor = tuple(np.array(current) + np.array(direction))

            # 检查邻居节点是否在网格范围内
            if (0 <= neighbor[0] < rows) and (0 <= neighbor[1] < cols) and (
                    0 <= neighbor[2] < heights):
                # 如果邻居节点不是障碍物, 则计算g值
                # print(grid[neighbor])
                if grid[neighbor] == 0.0:  # 检查邻居节点是否为障碍物
                    tentative_g_score = current_g + 1  # 假设从当前节点到邻居节点的g值

                    # 如果邻居节点尚未探索, 或者通过当前节点到达邻居节点的路径更短
                    if neighbor not in g_score or tentative_g_score < g_score[
                        neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list,
                                       (f_score, tentative_g_score, neighbor))
                        came_from[neighbor] = current  # 记录路径

    # 如果open_list为空且未找到目标节点, 返回空路径
    return []


def plan_path(grid, start, goal):
    """给定栅格地图、起点和终点, 规划路径

    参数:
    grid -- 3D栅格地图
    start -- 起点坐标
    goal -- 终点坐标

    返回:
    path -- 最优路径, 或为空
    """
    path = astar(grid, start, goal)
    if not path:
        return []
    return path
