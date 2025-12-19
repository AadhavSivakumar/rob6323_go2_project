# ROB6323 Go2 Project — Isaac Lab
## By Nate Smurthwaite, Dana Sy-Ching, and Aadhav Sivakumar
### Part 1 - Penalizing Action Rate

By viewing the output of the base code, we can see that the robot is purely optimizing its adherence to the change in yaw and the linear velocity, meaning that it will take any actions to achieve this. It results in the robot taking jerky microsteps, actions that would not be considered “walking” in real life. To combat this, we can penalize the robot’s rate of action, ensuring that it takes larger sweeping motions that look more natural.

Calculation of the linear velocity and yaw error which is eventually mapped appropriately
```python
lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])

```

### Part 2 - PD controller

Instead of letting the simulator control the torques, we can take direct control by implementing our own controller. Using a PD controller, we can manually tune the gains to get the most desired outcome. It also lets us set manual torque limits, so if these controls were applied to a real life robot, it wouldn’t surpass its limits.

Calculation of torque factoring in friction with imposed max torque of 100.
```python
torques = torch.clip(
            (
                self.Kp  (
                    self.desiredjointpos 
                    - self.robot.data.jointpos
                )
                - self.Kd  self.robot.data.jointvel
            ),
            -self.torquelimits,
            self.torquelimits,
        )

```


### Part 3 - Discarding failed runs

In order to make the training more efficient, we can discard runs that fail. Episodes in which the robot falls down are not useful, because that sequence of controls led to failure. We can determine if a robot falls down by tracking its body height, and if it goes below a certain threshold that run can terminated.

Comparison of base height against threshold and other termination conditions.
```python
…
cstr_base_height_min = base_height < self.cfg.base_height_min
died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
```

### Part 4 - Raibert Heuristic

Proper gaits follow a cycle, so if the robot’s footfalls aren’t following a cycle, it can be seen that the gait looks irregular. The Raibert Heuristic allows the robot to quickly adopt a structured walking cycle, by utilizing appropriate phase calculations for each leg instead of spending many runs to discover the same thing by chance.

These parameters can be changed to modify the gait. There are also associated calculations that combine the properties of the robot to get the actual offsets.
```python
desired_stance_width = 0.25 
desired_stance_length = 0.45
phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
frequencies = torch.tensor([3.0], device=self.device)
```

### Part 5 - Reward Function Refining

There are some other undesired behaviors that occur during the robot’s movement, and those can be eliminated by adding their contributions to the reward function as well. Things like excessive body tilt, vertical bouncing, or high joint velocities can be systematically eliminated by tuning their associated scales.

Calculating penalties for the above mentioned behaviors
```python
…
rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
```

### Part 6 - Foot Interaction

All of the previous parts have been focused on how the dog’s body and legs are reacting, but there hasn’t been much focus on the feet themselves. In order to make the gait even better, we can penalize not lifting the feet up enough, and we can reward strong contact forces with the ground, which would reduce slipping.

Calculations for feet clearance reward factoring in actual and target foot height along with calculation of forces upon contact
```python
…
rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states) 
rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1)  
…
for i in range(4):
rew_tracking_contacts_shaped_force += - (1 - desired_contact[:, i]) * (1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.)) 
rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force / 4
```

### Part 7 - Friction (bonus)

Implementing static and viscous friction torque will allow us to have a more accurate and realistic model that would improve outcomes on a real robot. We utilize the joint torques to calculate these forces and implement them into the torque caclulation seen above (Part 2)

Stiction and viscous coefficientis randomized according to a uniform distribution per episode
```python
n = len(env_ids) 
self.rand_stict[env_ids] = sample_uniform(0.0, 2.5, (n,), device=self.device) self.rand_visc[env_ids] = sample_uniform(0.00, 0.3, (n,), device=self.device)
```

Implementation of static and viscous friction formulas
```python
stiction_torque = self.rand_stict.unsqueeze(1) * torch.tanh(self.robot.data.joint_vel / 0.1) viscous_torque = self.rand_visc.unsqueeze(1) * self.robot.data.joint_vel
```

Modified torque calculation
```python
torques = torch.clip(
            (
                self.Kp  (
                    self.desiredjointpos 
                    - self.robot.data.jointpos
                )
                - self.Kd  self.robot.data.jointvel
                - (stictiontorque + viscoustorque)   # subtract friction
            ),
            -self.torquelimits,
            self.torquelimits,
        )

```


## Video Output
[Here](https://drive.google.com/file/d/1OQzUCm4y6LDllz53u_WXwhSlS3K-UTh-/view?usp=sharing) you can see the final video output 

You can see the result of the added part 7 with no modifications to the previous reward scales [here](https://drive.google.com/file/d/1KPHd-W2sX720DV_xDN4l9nRK_7CbXZ-S/view?usp=sharing)
(slight modifications to feet clearance and contact force offer minor improvements)
