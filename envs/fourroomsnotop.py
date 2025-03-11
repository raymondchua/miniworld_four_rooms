import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

from CRLMSF.environments.gym_miniworld.miniworld import *


class FourRoomsNoTop(MiniWorldEnv):
    """
    Modification of the classic four rooms without connecting room in the bottom environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(
        self,
        max_episode_steps: int,
        scale: float = 1.0,
        agent_pos=[-6.5, 0, -6.5],
        coverage_plot=False,
        **kwargs
    ):
        self.agent_pos = agent_pos
        self.coverage_plot = coverage_plot
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)
        self._actions_name = ["Turn left", "Turn right", "Forward", "Move back"]
        self.action_space = spaces.Discrete(len(self._actions_name))  # 0-indexed

        # random direction for the agent
        self._possible_directions = np.asarray([0, 90, 180, 270])
        self._agent_dir_choice = np.random.choice(self._possible_directions)

        # convert agent_dir to radians
        # agent_dir_radians = self._agent_dir_choice * math.pi / 180.0

        # self.agent_pos = [agent_x, 0.0, agent_z]
        # self.agent_pos = None
        self.agent.pos = [self.agent_pos[0], 0.0, self.agent_pos[2]]
        self._scale = scale

    def _gen_world(self):
        """
        Generate the world with random agent position and direction.
        Returns
        -------

        """
        # Bottom-left room
        room0 = self.add_rect_room(
            min_x=-7 * self._scale,
            max_x=-1 * self._scale,
            min_z=1 * self._scale,
            max_z=7 * self._scale,
            wall_tex="cardboard",
        )
        # Bottom-right room
        room1 = self.add_rect_room(
            min_x=1 * self._scale,
            max_x=7 * self._scale,
            min_z=1 * self._scale,
            max_z=7 * self._scale,
            wall_tex="marble",
        )
        # Top-right room
        room2 = self.add_rect_room(
            min_x=1 * self._scale,
            max_x=7 * self._scale,
            min_z=-7 * self._scale,
            max_z=-1 * self._scale,
            wall_tex="metal_grill",
        )
        # Top-left room
        room3 = self.add_rect_room(
            min_x=-7 * self._scale,
            max_x=-1 * self._scale,
            min_z=-7 * self._scale,
            max_z=-1 * self._scale,
            wall_tex="stucco",
        )

        # Add openings to connect the rooms together
        self.connect_rooms(
            room0, room1, min_z=3 * self._scale, max_z=5 * self._scale, max_y=2.2
        )
        self.connect_rooms(
            room1, room2, min_x=3 * self._scale, max_x=5 * self._scale, max_y=2.2
        )
        self.connect_rooms(
            room3, room0, min_x=-5 * self._scale, max_x=-3 * self._scale, max_y=2.2
        )

        self.box_bottomRight = self.place_entity(
            Box(color="green"),
            pos=np.array([1.5 * self._scale, 0.0, 1.5 * self._scale]),
            dir=0,
        )
        self.box_topLeft = self.place_entity(
            Box(color="yellow"),
            pos=np.array([-1.5 * self._scale, 0.0, -1.5 * self._scale]),
            dir=0,
        )

        self._possible_directions = np.asarray([0, 90, 180, 270])
        self._agent_dir_choice = np.random.choice(self._possible_directions)

        agent_dir_radians = self._agent_dir_choice * math.pi / 180.0

        self.place_agent(dir=agent_dir_radians)

        if self.coverage_plot:
            conds = []
            for room in self.rooms:
                # maybe need to print here
                conds.append(room.point_inside(self.agent.pos))
            self.valid_pos = any(conds)

    def _gen_world_with_agent_pos_dir(
        self,
        agent_pos: tuple = None,
        agent_dir: int = None,
        room_idx: int = None,
    ) -> tuple[tuple, bool, int]:
        """
        Generate the world with the given agent position, direction or room index.

        if agent_pos is None, then agent_pos is randomly generated
        if agent_dir is None, then agent_dir is randomly generated
        if room_idx is None, then room_idx is randomly generated

        Parameters
        ----------
        agent_pos: tuple
        agent_dir: int (in degrees)

        Returns
        -------
        valid_pos: bool - whether the agent position is valid or not
        room_idx: int - index of the room in which the agent is placed
        """

        # Bottom-left room
        room0 = self.add_rect_room(
            min_x=-7 * self._scale,
            max_x=-1 * self._scale,
            min_z=1 * self._scale,
            max_z=7 * self._scale,
            wall_tex="cardboard",
        )
        # Bottom-right room
        room1 = self.add_rect_room(
            min_x=1 * self._scale,
            max_x=7 * self._scale,
            min_z=1 * self._scale,
            max_z=7 * self._scale,
            wall_tex="marble",
        )
        # Top-right room
        room2 = self.add_rect_room(
            min_x=1 * self._scale,
            max_x=7 * self._scale,
            min_z=-7 * self._scale,
            max_z=-1 * self._scale,
            wall_tex="metal_grill",
        )
        # Top-left room
        room3 = self.add_rect_room(
            min_x=-7 * self._scale,
            max_x=-1 * self._scale,
            min_z=-7 * self._scale,
            max_z=-1 * self._scale,
            wall_tex="stucco",
        )

        # Add openings to connect the rooms together
        self.connect_rooms(
            room0, room1, min_z=3 * self._scale, max_z=5 * self._scale, max_y=2.2
        )
        self.connect_rooms(
            room1, room2, min_x=3 * self._scale, max_x=5 * self._scale, max_y=2.2
        )
        # self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(
            room3, room0, min_x=-5 * self._scale, max_x=-3 * self._scale, max_y=2.2
        )

        self.box_bottomRight = self.place_entity(
            Box(color="green"),
            pos=np.array([1.5 * self._scale, 0.0, 1.5 * self._scale]),
            dir=0,
        )
        self.box_topLeft = self.place_entity(
            Box(color="yellow"),
            pos=np.array([-1.5 * self._scale, 0.0, -1.5 * self._scale]),
            dir=0,
        )

        # convert agent_dir from degrees to radians
        agent_dir_radians = agent_dir * math.pi / 180.0

        if agent_pos is not None:
            (
                _,
                valid_position,
                room_idx_assigned,
            ) = self.place_agent_with_valid_position_checks(
                dir=agent_dir_radians, pos=agent_pos
            )

        elif agent_pos is None and room_idx is not None:
            (
                ent,
                valid_position,
                room_idx_assigned,
            ) = self.place_agent_randomly_in_specific_room(
                dir=agent_dir_radians, room_idx=room_idx
            )
            agent_pos = ent.pos

            assert room_idx_assigned == room_idx

        # if both agent_pos and room_idx are None, then randomly place the agent
        else:
            (
                ent,
                valid_position,
                room_idx_assigned,
            ) = self.place_agent_randomly_with_valid_position_checks(
                dir=agent_dir_radians
            )
            agent_pos = ent.pos

        if self.coverage_plot:
            conds = []
            for room in self.rooms:
                # maybe need to print here
                conds.append(room.point_inside(self.agent.pos))
            self.valid_pos = any(conds)

        return agent_pos, valid_position, room_idx_assigned

    def set_agent_pos(self, agent_pos):
        self.agent_pos = agent_pos

    def step(self, action):
        pass


class FourRoomsNoTopActions(FourRoomsNoTop):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        rand = self.rand if self.domain_rand else None
        fwd_step = self.params.sample(rand, "forward_step")
        fwd_drift = self.params.sample(rand, "forward_drift")
        turn_step = self.params.sample(rand, "turn_step")

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.turn_agent(turn_step * 2)
            self.move_agent(fwd_step, fwd_drift)

        # elif action == self.actions.move_left:
        #     self.turn_agent(turn_step)
        #     self.move_agent(fwd_step, fwd_drift)
        #
        # elif action == self.actions.move_right:
        #     self.turn_agent(-turn_step)
        #     self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        # Pick up an object
        elif action == self.actions.pickup:
            # Position at which we will test for an intersection
            test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
            ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
            if not self.agent.carrying:
                if isinstance(ent, Entity):
                    if not ent.is_static:
                        self.agent.carrying = ent

        # Drop an object being carried
        elif action == self.actions.drop:
            if self.agent.carrying:
                self.agent.carrying.pos[1] = 0
                self.agent.carrying = None

        # If we are carrying an object, update its position as we move
        if self.agent.carrying:
            ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
            self.agent.carrying.pos = ent_pos
            self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        if self.obs_view == "agent":
            obs = self.render_obs()
        else:
            obs = self.render_top_view()

        # update agent room
        self.update_agent_room()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        return obs, reward, done, {}

    def step_with_agent_pos_dir(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        agent_in_room_idx = 0

        rand = self.rand if self.domain_rand else None
        fwd_step = self.params.sample(rand, "forward_step")
        fwd_drift = self.params.sample(rand, "forward_drift")
        turn_step = self.params.sample(rand, "turn_step")

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.turn_agent(turn_step * 2)
            self.move_agent(fwd_step, fwd_drift)

        # elif action == self.actions.move_left:
        #     self.turn_agent(turn_step)
        #     self.move_agent(fwd_step, fwd_drift)
        #
        # elif action == self.actions.move_right:
        #     self.turn_agent(-turn_step)
        #     self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        # Pick up an object
        elif action == self.actions.pickup:
            # Position at which we will test for an intersection
            test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
            ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
            if not self.agent.carrying:
                if isinstance(ent, Entity):
                    if not ent.is_static:
                        self.agent.carrying = ent

        # Drop an object being carried
        elif action == self.actions.drop:
            if self.agent.carrying:
                self.agent.carrying.pos[1] = 0
                self.agent.carrying = None

        # If we are carrying an object, update its position as we move
        if self.agent.carrying:
            ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
            self.agent.carrying.pos = ent_pos
            self.agent.carrying.dir = self.agent.dir
            self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        if self.obs_view == "agent":
            obs = self.render_obs()
        else:
            obs = self.render_top_view()

        # update agent room
        self.update_agent_room()

        self.agent.dir_degrees = int(self.agent.dir * 180 / math.pi) % 360

        # if self.agent.dir_degrees is not one of the 0, 90, 180, 270, then round it to the nearest one
        if self.agent.dir_degrees not in [0, 90, 180, 270]:
            self.agent.dir_degrees = int(round(self.agent.dir_degrees / 90.0)) * 90

        for room in self.rooms:
            # maybe need to print here
            if room.point_inside(self.agent.pos):
                agent_in_room_idx = self.rooms.index(room)

        if self.near(self.box_bottomRight):
            reward = self._reward()
            done = True
            return (
                obs,
                reward,
                done,
                {
                    "agent_pos": self.agent.pos_round,
                    "agent_dir": self.agent.dir_degrees,
                    "room_idx": agent_in_room_idx,
                },
            )

        if self.near(self.box_topLeft):
            reward = -1
            done = True
            return (
                obs,
                reward,
                done,
                {
                    "agent_pos": self.agent.pos_round,
                    "agent_dir": self.agent.dir_degrees,
                    "room_idx": agent_in_room_idx,
                },
            )

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            # convert self.agent.pos to two decimal places

            return (
                obs,
                reward,
                done,
                {
                    "agent_pos": self.agent.pos_round,
                    "agent_dir": self.agent.dir_degrees,
                    "room_idx": agent_in_room_idx,
                },
            )

        reward = 0
        done = False

        return (
            obs,
            reward,
            done,
            {
                "agent_pos": self.agent.pos_round,
                "agent_dir": self.agent.dir_degrees,
                "room_idx": agent_in_room_idx,
            },
        )


class FourRoomsNoTopTwoTasks(FourRoomsNoTopActions):
    def __init__(self, task_id: int = 0, scale: float = 1.0):
        self._scale = scale
        super().__init__(max_episode_steps=4000, coverage_plot=False, scale=scale)
        self._task_id = task_id

        assert self._task_id in [0, 1]

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # update agent room
        self.update_agent_room()

        if self.near(self.box_bottomRight):
            if self._task_id == 0:
                reward += self._reward()
                done = True
            else:
                reward -= 1
                done = True

        elif self.near(self.box_topLeft):
            if self._task_id == 0:
                reward -= 1
                done = True
            else:
                reward += self._reward()
                done = True

        return obs, reward, done, info

    def step_with_agent_pos_dir(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        agent_in_room_idx = 0

        rand = self.rand if self.domain_rand else None
        fwd_step = self.params.sample(rand, "forward_step")
        fwd_drift = self.params.sample(rand, "forward_drift")
        turn_step = self.params.sample(rand, "turn_step")

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.turn_agent(turn_step * 2)
            self.move_agent(fwd_step, fwd_drift)

        # elif action == self.actions.move_left:
        #     self.turn_agent(turn_step)
        #     self.move_agent(fwd_step, fwd_drift)
        #
        # elif action == self.actions.move_right:
        #     self.turn_agent(-turn_step)
        #     self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        # Pick up an object
        elif action == self.actions.pickup:
            # Position at which we will test for an intersection
            test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
            ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
            if not self.agent.carrying:
                if isinstance(ent, Entity):
                    if not ent.is_static:
                        self.agent.carrying = ent

        # Drop an object being carried
        elif action == self.actions.drop:
            if self.agent.carrying:
                self.agent.carrying.pos[1] = 0
                self.agent.carrying = None

        # If we are carrying an object, update its position as we move
        if self.agent.carrying:
            ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
            self.agent.carrying.pos = ent_pos
            self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        if self.obs_view == "agent":
            obs = self.render_obs()
        else:
            obs = self.render_top_view()

        # update agent room
        self.update_agent_room()

        self.agent.pos_round = np.round(self.agent.pos, 2)
        self.agent.dir_degrees = int(self.agent.dir * 180 / math.pi) % 360

        # if self.agent.dir_degrees is not one of the 0, 90, 180, 270, then round it to the nearest one
        if self.agent.dir_degrees not in [0, 90, 180, 270]:
            self.agent.dir_degrees = (
                int((round(self.agent.dir_degrees / 90.0)) * 90) % 360
            )

        for room in self.rooms:
            # maybe need to print here
            if room.point_inside(self.agent.pos):
                agent_in_room_idx = self.rooms.index(room)

        if self.near(self.box_bottomRight):
            if self._task_id == 0:
                reward = self._reward()
                done = True

            else:
                reward = -1
                done = True

            return (
                obs,
                reward,
                done,
                {
                    "agent_pos": self.agent.pos_round,
                    "agent_dir": self.agent.dir_degrees,
                    "room_idx": agent_in_room_idx,
                },
            )

        if self.near(self.box_topLeft):
            if self._task_id == 0:
                reward = -1
                done = True
            else:
                reward = self._reward()
                done = True
            return (
                obs,
                reward,
                done,
                {
                    "agent_pos": self.agent.pos_round,
                    "agent_dir": self.agent.dir_degrees,
                    "room_idx": agent_in_room_idx,
                },
            )

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            # convert self.agent.pos to two decimal places

            return (
                obs,
                reward,
                done,
                {
                    "agent_pos": self.agent.pos_round,
                    "agent_dir": self.agent.dir_degrees,
                    "room_idx": agent_in_room_idx,
                },
            )

        reward = 0
        done = False

        return (
            obs,
            reward,
            done,
            {
                "agent_pos": self.agent.pos_round,
                "agent_dir": self.agent.dir_degrees,
                "room_idx": agent_in_room_idx,
            },
        )


class FourRoomsNoTopTask1(FourRoomsNoTopTwoTasks):
    def __init__(self):
        super().__init__(task_id=0)


class FourRoomsNoTopTask2(FourRoomsNoTopTwoTasks):
    def __init__(self):
        super().__init__(task_id=1)


class FourRoomsNoTopSize2Task1(FourRoomsNoTopTwoTasks):
    def __init__(self):
        super().__init__(task_id=0, scale=1.2)


class FourRoomsNoTopSize2Task2(FourRoomsNoTopTwoTasks):
    def __init__(self):
        super().__init__(task_id=1, scale=1.2)


class FourRoomsNoTopSize3Task1(FourRoomsNoTopTwoTasks):
    def __init__(self):
        super().__init__(task_id=0, scale=1.5)


class FourRoomsNoTopSize3Task2(FourRoomsNoTopTwoTasks):
    def __init__(self):
        super().__init__(task_id=1, scale=1.5)
