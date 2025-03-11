import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces

from CRLMSF.environments.gym_miniworld.miniworld import *


class TMaze(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(
        self,
        goal_pos=None,
        task_id: int = 0,
        pos_change_upon_reset: bool = True,
        **kwargs
    ):
        self.goal_pos = goal_pos
        self._pos_change_upon_reset = pos_change_upon_reset

        super().__init__(max_episode_steps=280, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        self._task_id = task_id
        assert self._task_id in [0, 1]

    def _gen_world(self):
        # default env sizes and locations
        # room1 = self.add_rect_room(min_x=-1, max_x=8, min_z=-2, max_z=2)
        # room1 = self.add_rect_room(min_x=5, max_x=8, min_z=-1.5, max_z=1.5)
        # room2 = self.add_rect_room(min_x=8, max_x=10, min_z=-5, max_z=5)
        # self.connect_rooms(room1, room2, min_z=-2, max_z=2)
        #
        # self.box_left = self.place_entity(
        #     Box(color="green"), pos=np.array([8.89, 0.0, -2]), dir=0
        # )
        # self.box_right = self.place_entity(
        #     Box(color="yellow"), pos=np.array([8.89, 0.0, 2]), dir=0
        # )

        room1 = self.add_rect_room(min_x=0, max_x=1, min_z=-1, max_z=1)
        room2 = self.add_rect_room(min_x=4, max_x=6, min_z=-4, max_z=4)
        self.connect_rooms(room1, room2, min_z=-1, max_z=1)

        self.box_left = self.place_entity(
            Box(color="green"), pos=np.array([5, 0.0, -3]), dir=0
        )
        self.box_right = self.place_entity(
            Box(color="yellow"), pos=np.array([5, 0.0, 3]), dir=0
        )

        self._possible_directions = np.asarray([0, 90, 180, 270])
        self._agent_dir_choice = np.random.choice(self._possible_directions)

        agent_dir_radians = self._agent_dir_choice * math.pi / 180.0

        if self._pos_change_upon_reset:
            self.place_agent(dir=agent_dir_radians)

        else:
            default_pos_btm_main_room = [1.1, 0.0, -0.09]
            self.place_agent(dir=agent_dir_radians, pos=default_pos_btm_main_room)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # update agent room
        self.update_agent_room()

        if self.near(self.box_right):
            if self._task_id == 0:
                reward += self._reward()
                done = True
            else:
                reward -= 1
                done = True

        elif self.near(self.box_left):
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

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)
            # self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)
            # self.move_agent(fwd_step, fwd_drift)

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
            if room.point_inside(self.agent.pos):
                agent_in_room_idx = self.rooms.index(room)

        if self.near(self.box_left):
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

        if self.near(self.box_right):
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


class TMazeTask1(TMaze):
    def __init__(self):
        super().__init__(task_id=0)


class TMazeTask2(TMaze):
    def __init__(self):
        super().__init__(task_id=1)

class TMazeSameStartPosTask1(TMaze):
    def __init__(self):
        super().__init__(task_id=0, pos_change_upon_reset=False)

class TMazeSameStartPosTask2(TMaze):
    def __init__(self):
        super().__init__(task_id=1, pos_change_upon_reset=False)
