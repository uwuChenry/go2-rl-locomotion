# Go2 RL Locomotion Model - Technical Architecture

## Overview

The go2-rl-locomotion project implements a comprehensive quadruped locomotion system using reinforcement learning in Isaac Lab. This document details the model architecture, input/output specifications, and reward function design for training a Unitree Go2 robot to perform natural locomotion with height-aware navigation and proper gait patterns.

This project targets **Isaac Sim 6.0.1** and **Isaac Lab 3.0**, ported from an earlier Isaac Lab 2.x / Isaac Sim 4.5-5.0 version of the same project. See "Usage Examples" below for the updated invocation.

## Model Architecture

### Environment Framework
- **Base Environment**: `ManagerBasedRLEnv` from Isaac Lab
- **Algorithm**: Proximal Policy Optimization (PPO) via RSL-RL
- **Physics Engine**: NVIDIA PhysX with GPU acceleration
- **Simulation Frequency**: 200 Hz (dt = 0.005s)
- **Control Frequency**: 50 Hz (decimation = 4)

### Robot Configuration
- **Robot Model**: Unitree Go2 (simplified URDF)
- **Degrees of Freedom**: 12 (3 per leg: hip, thigh, calf joints)
- **Actuator Type**: DC Motor with position control
- **Joint Control**: Position targets with 35% scaling factor

### Neural Network Architecture
```
Policy Network (Actor):
├── Input Layer: 48 dimensions (observations)
├── Hidden Layer 1: 128 neurons (ELU activation)
├── Hidden Layer 2: 128 neurons (ELU activation) 
├── Hidden Layer 3: 128 neurons (ELU activation)
└── Output Layer: 12 dimensions (joint position targets)

Value Network (Critic):
├── Input Layer: 48 dimensions (observations)
├── Hidden Layer 1: 128 neurons (ELU activation)
├── Hidden Layer 2: 128 neurons (ELU activation)
├── Hidden Layer 3: 128 neurons (ELU activation)
└── Output Layer: 1 dimension (state value estimate)
```

### Training Configuration
- **Episode Length**: 20 seconds (4000 steps at 200Hz)
- **Steps per Environment**: 24
- **Number of Environments**: 32,000 (training) / 50 (play)
- **Learning Rate**: 1e-3 with adaptive scheduling
- **Batch Size**: 4 mini-batches
- **Training Epochs**: 5 per iteration

## Inputs and Outputs

### Observation Space (48 dimensions)

| Component | Dimensions | Description | Noise Range |
|-----------|------------|-------------|-------------|
| `base_lin_vel` | 3 | Robot base linear velocity in body frame | ±0.1 m/s |
| `base_ang_vel` | 3 | Robot base angular velocity in body frame | ±0.2 rad/s |
| `projected_gravity` | 3 | Gravity vector projected to body frame | ±0.05 |
| `velocity_commands` | 3 | Commanded velocities (lin_x, lin_y, ang_z) | No noise |
| `joint_pos` | 12 | Joint positions relative to default | ±0.01 rad |
| `joint_vel` | 12 | Joint velocities relative to default | ±1.5 rad/s |
| `actions` | 12 | Previous action values (memory) | No noise |

**Total**: 48 observations per environment

### Action Space (12 dimensions)
- **Type**: Continuous joint position targets
- **Range**: [-1.0, 1.0] (scaled to ±35% of joint limits)
- **Joints**: 
  ```
  FL_hip_joint, FL_thigh_joint, FL_calf_joint    # Front Left
  FR_hip_joint, FR_thigh_joint, FR_calf_joint    # Front Right  
  RL_hip_joint, RL_thigh_joint, RL_calf_joint    # Rear Left
  RR_hip_joint, RR_thigh_joint, RR_calf_joint    # Rear Right
  ```

### Command Interface
- **Linear Velocity**: (-2.0, 2.5) m/s in X, (-1.0, 1.0) m/s in Y
- **Angular Velocity**: (-1.0, 1.0) rad/s around Z-axis
- **Heading Control**: Full 360° with stiffness control

### Sensor Systems

#### Contact Sensors
- **Location**: All 4 feet (`*_foot` bodies)
- **Data**: Force vectors, contact timing, air time
- **Update Rate**: 200 Hz (every simulation step)

#### Height Scanner (RayCaster)
- **Pattern**: 16×10 grid (160 rays total)
- **Coverage**: 1.6m × 1.0m area in front of robot
- **Range**: 3.0m maximum distance
- **Offset**: 0.1m forward from robot base
- **Update Rate**: 50 Hz (every control step)

## Reward Function Architecture

The reward system implements a multi-component approach to encourage natural locomotion behaviors:

### 1. Velocity Tracking Rewards
```python
lin_vel_z_l2: -2.0      # Penalize vertical motion
ang_vel_xy_l2: -0.05    # Penalize roll/pitch rotation  
lin_vel_xy_exp: 1.5     # Track commanded X-Y velocity
```
**Purpose**: Encourage following velocity commands while maintaining stability.

### 2. Energy Efficiency Rewards
```python
dof_torques_l2: -0.0002    # Minimize joint torques
action_rate_l2: -0.01      # Smooth action changes
dof_acc_l2: -2.5e-7        # Penalize rapid accelerations
```
**Purpose**: Promote energy-efficient, smooth movements.

### 3. Height-Aware Navigation
```python
def reward_height_scan_above_threshold(min_height=0.25, max_height=0.30):
    """Linear reward for terrain height awareness"""
    height_data = sensor_height - ray_hits_z
    mean_height = height_data.mean(dim=1)
    return torch.clamp((mean_height - min_height) / (max_height - min_height), 0.0, 1.0)
```
**Purpose**: Encourage navigation over elevated terrain (25-30cm height range).

### 4. Foot Clearance Reward
```python
def foot_clearance_reward(target_height=0.06, std=0.05):
    """Reward proper foot lifting during swing phase"""
    foot_height_error = torch.square(foot_z - target_height)
    velocity_factor = torch.tanh(foot_velocity_norm)
    return torch.exp(-torch.sum(foot_height_error * velocity_factor) / std)
```
**Purpose**: Encourage 6cm foot clearance during swing phase to prevent dragging.

### 5. Gait Pattern Enforcement
```python
def gait_reward(synced_pairs=[["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]]):
    """Enforce trotting gait with diagonal coordination"""
    # Synchronize diagonal pairs (FL+RR, FR+RL)
    sync_reward = diagonal_pair_synchronization()
    # Anti-synchronize between pairs  
    async_reward = diagonal_pair_opposition()
    return sync_reward * async_reward
```
**Purpose**: Enforce natural trotting gait with diagonal foot coordination.

### 6. Safety and Stability Rewards
```python
undesired_contacts: -0.1      # Penalize thigh ground contact
flat_orientation_l2: -2.5     # Maintain upright posture
base_contact: TERMINATION     # End episode on base contact
```
**Purpose**: Prevent unsafe configurations and maintain stability.


## Training Dynamics

### Randomization
- **Mass Randomization**: Base mass ±1-3 kg
- **Center of Mass**: ±5cm in X-Y, ±1cm in Z
- **Surface Properties**: Static friction 0.8, dynamic 0.6
- **Initial Conditions**: Random base pose within limits

### Performance Metrics
- **Success Criteria**: Stable locomotion with commanded velocity tracking
- **Episode Termination**: Base contact or 20-second timeout
- **Key Metrics**: 
  - Average episode reward
  - Velocity tracking error
  - Gait consistency
  - Foot clearance statistics

### Hardware Requirements
- **GPU Memory**: 20gb+ recommended for 32K environments 12gb+ for 4k

## Usage Examples

### Training
```bash
# From this project directory (legacy per-library entrypoint, still supported)
./IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=go2-rl-locomotion --headless

# Isaac Lab 3.0 unified entrypoint (run from the IsaacLab checkout, with this
# package pip-installed into the same environment)
./isaaclab.sh train --rl_library rsl_rl --task go2-rl-locomotion --headless
```

### Evaluation
```bash
./IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=go2-rl-locomotion --num_envs=32

# or, unified entrypoint
./isaaclab.sh play --rl_library rsl_rl --task go2-rl-locomotion --num_envs=32
```

### Configuration Files
- Main Config: `source/go2_rl_locomotion/go2_rl_locomotion/tasks/manager_based/go2/go2_env_cfg.py`
- Robot Model: `source/go2_rl_locomotion/go2_rl_locomotion/tasks/manager_based/go2/go2ArticulationCFG.py`
- Training Config: `source/go2_rl_locomotion/go2_rl_locomotion/tasks/manager_based/go2/agents/rsl_rl_ppo_cfg.py`

## Future Enhancements

1. **Terrain Adaptation**: Curriculum learning on varied terrains
2. **Dynamic Gait Selection**: Adaptive gait switching based on speed
3. **Vision Integration**: Camera-based navigation and obstacle avoidance
4. **Robustness Testing**: Disturbance rejection and recovery behaviors
5. **Real Robot Transfer**: Sim-to-real domain adaptation techniques
