import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
from deap import base, creator, tools, algorithms  # 使用 DEAP 库进行遗传算法优化
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import os
from ament_index_python.packages import get_package_share_directory

class CartesianImpedanceController(Node):

    def __init__(self):
        super().__init__('cartesian_impedance_controller')

        # 订阅 franka_robot_state_broadcaster/current_pose 话题
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.current_pose_callback,
            10
        )

        # 订阅 franka_robot_state_broadcaster/desired_end_effector_twist 话题
        self.twist_subscription = self.create_subscription(
            TwistStamped,
            '/franka_robot_state_broadcaster/desired_end_effector_twist',
            self.desired_twist_callback,
            10
        )

        # 订阅 /joint_states 话题
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 发布优化的 K 矩阵
        self.k_matrix_publisher = self.create_publisher(Float64MultiArray, 'optimized_k_matrix', 10)

        # 初始化目标位置、欧拉角、线速度和角速度
        self.position_d_target_ = np.zeros(3)
        self.orientation_euler_d_target_ = np.zeros(3)  # 用于欧拉角
        self.linear_velocity_d_target_ = np.zeros(3)  # 用于线速度
        self.angular_velocity_d_target_ = np.zeros(3)  # 用于角速度

        # 控制器的 K 矩阵 (6x6 矩阵，作为遗传算法优化变量)，对角矩阵
        self.K_matrix = np.eye(6) * 150  # 初始设为对角线元素为 150

        # 目标位置和角度
        self.desired_state = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])  # x, y, z, roll, pitch, yaw

        # 阻尼矩阵
        self.D_matrix = np.array([[35, 0, 0, 0, 0, 0],[0, 35, 0, 0, 0, 0],[0, 0, 35, 0, 0, 0],[0, 0, 0, 25, 0, 0],[0, 0, 0, 0, 25, 0],[0, 0, 0, 0, 0, 6]])  # 阻尼矩阵

        # 时间步长
        self.dt = 0.01  # 假设时间步长为 0.01s
        self.n_steps = 10  # 预测的时间步长

        # 初始化遗传算法框架
        self.setup_genetic_algorithm()

        # 初始化 Pinocchio 模型
        package_share_directory = get_package_share_directory('franka_description')
        urdf_path = os.path.join(package_share_directory, 'robots', 'fr3', 'fr3.urdf')
        self.model = RobotWrapper.BuildFromURDF(urdf_path, [os.path.join(package_share_directory, 'robots', 'fr3')])
        self.data = self.model.model.createData()

        # 初始化关节位置
        self.q = np.zeros(self.model.model.nq)  # 确保关节位置向量的长度匹配模型

    def setup_genetic_algorithm(self):
        # 初始化遗传算法的框架
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, 0, 250)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=6)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_K_matrix)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=125, sigma=20, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_K_matrix(self, individual):
        individual = [max(0, x) for x in individual]
        K_matrix = np.diag(individual)

        current_state = np.hstack((self.position_d_target_, self.orientation_euler_d_target_))
        current_velocity = np.hstack((self.linear_velocity_d_target_, self.angular_velocity_d_target_))
        desired_state = self.desired_state

        total_cost = 0
        predicted_states = [current_state]
        predicted_velocities = [current_velocity]

        for k in range(self.n_steps):
            state_error = predicted_states[k] - desired_state
            M_cartesian = self.compute_cartesian_mass_matrix()  # 计算由 Pinocchio 获取的 M 矩阵
            M_inv = np.linalg.inv(M_cartesian)
            acceleration = M_inv @ (K_matrix @ state_error + self.D_matrix @ predicted_velocities[k])
            next_velocity = predicted_velocities[k] + acceleration * self.dt
            next_state = predicted_states[k] + predicted_velocities[k] * self.dt
            predicted_velocities.append(next_velocity)
            predicted_states.append(next_state)
            total_cost += np.sum(state_error ** 2)

        self.get_logger().info(f"Evaluating K Matrix with cost: {total_cost}")
        return total_cost,

    def optimize_K_matrix_genetic(self):
        population = self.toolbox.population(n=50)
        cxpb, mutpb, ngen = 0.5, 0.2, 40
        population, logbook = algorithms.eaSimple(population, self.toolbox, cxpb, mutpb, ngen, stats=None, verbose=False)

        best_individual = tools.selBest(population, 1)[0]
        self.K_matrix = np.diag([max(0, x) for x in best_individual])
        self.get_logger().info(f"Optimized Full K Matrix:\n{self.K_matrix}")

        # 发布优化的 K 矩阵
        self.publish_k_matrix()

    def publish_k_matrix(self):
        msg = Float64MultiArray()
        msg.data = [float(item) for row in self.K_matrix for item in row]
        self.k_matrix_publisher.publish(msg)
        self.get_logger().info(f"Published optimized K matrix: {msg.data}")

    def current_pose_callback(self, msg):
        self.position_d_target_ = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        orientation_quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        self.orientation_euler_d_target_ = R.from_quat(orientation_quat).as_euler('xyz')
        current_state = np.hstack((self.position_d_target_, self.orientation_euler_d_target_))
        self.get_logger().info(f"Current State (Position + Orientation): {current_state}")
        self.optimize_K_matrix_genetic()

    def desired_twist_callback(self, msg):
        self.linear_velocity_d_target_ = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.angular_velocity_d_target_ = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        current_velocity = np.hstack((self.linear_velocity_d_target_, self.angular_velocity_d_target_))
        self.get_logger().info(f"Current Velocity (Linear + Angular): {current_velocity}")

    def joint_state_callback(self, msg):
        # 只获取模型所需数量的关节位置，确保向量大小正确
        num_joints = self.model.model.nq
        self.q = np.array(msg.position[:num_joints])
        self.get_logger().info(f"Updated joint positions: {self.q}")

    def compute_cartesian_mass_matrix(self):
        # 确保关节配置向量的大小是正确的
        if len(self.q) != self.model.model.nq:
            self.get_logger().error("Joint position vector q has incorrect size.")
            return np.eye(6)  # 返回一个默认值以防止程序崩溃

        # 计算关节空间质量矩阵
        M_joint_space = pin.crba(self.model.model, self.data, self.q)

        # 获取末端执行器的帧 ID
        end_effector_frame = 'fr3_hand'  # 请确认你的末端执行器名称，这里可能需要替换
        frame_id = self.model.model.getFrameId(end_effector_frame)

        # 前向运动学，用于更新数据
        pin.forwardKinematics(self.model.model, self.data, self.q)
        pin.updateFramePlacement(self.model.model, self.data, frame_id)

        # 计算雅可比矩阵，使用 LOCAL_WORLD_ALIGNED 参考框架
        J = pin.computeFrameJacobian(self.model.model, self.data, self.q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # 使用伪逆雅可比矩阵来进行转换
        J_pseudo_inv = np.linalg.pinv(J)
        M_cartesian = J_pseudo_inv.T @ M_joint_space @ J_pseudo_inv

        # 打印计算得到的笛卡尔质量矩阵
        self.get_logger().info(f"Computed Cartesian Mass Matrix:\n{M_cartesian}")

        return M_cartesian

def main(args=None):
    rclpy.init(args=args)
    node = CartesianImpedanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
