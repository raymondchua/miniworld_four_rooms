import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..maths import gen_rot_matrix
from gym import spaces

from CRLMSF.environments.gym_miniworld.miniworld import *


class YMaze(MiniWorldEnv):
    """
    Two hallways connected in a Y-junction
    """

    def __init__(self, task_id: int = 0, goal_pos=None, **kwargs):
        self.goal_pos = goal_pos

        super().__init__(max_episode_steps=280, **kwargs)

        self._task_id = task_id

        assert self._task_id in [0, 1]

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Outline of the main (starting) arm
        main_outline = np.array(
            [
                [-9.15, 0, -2],
                [-9.15, 0, +2],
                [-1.15, 0, +2],
                [-1.15, 0, -2],
            ]
        )

        main_arm = self.add_room(outline=np.delete(main_outline, 1, 1))

        # Triangular hub room, outline of XZ points
        hub_room = self.add_room(
            outline=np.array(
                [
                    [-1.15, -2],
                    [-1.15, +2],
                    [2.31, 0],
                ]
            )
        )

        # Left arm of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), -120 * (math.pi / 180))
        left_outline = np.dot(main_outline, m)
        left_arm = self.add_room(outline=np.delete(left_outline, 1, 1))

        # Right arm of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), +120 * (math.pi / 180))
        right_outline = np.dot(main_outline, m)
        right_arm = self.add_room(outline=np.delete(right_outline, 1, 1))

        # Connect the maze arms with the hub
        self.connect_rooms(main_arm, hub_room, min_z=-2, max_z=2)
        self.connect_rooms(left_arm, hub_room, min_z=-1.995, max_z=0)
        self.connect_rooms(right_arm, hub_room, min_z=0, max_z=1.995)

        self.box_left_arm = self.place_entity(
            Box(color="green"), room=left_arm, max_z=left_arm.min_z + 2.5, dir=0
        )
        self.box_right_arm = self.place_entity(
            Box(color="yellow"), room=right_arm, min_z=right_arm.max_z - 2.5, dir=0
        )

        self._possible_directions = np.asarray([0, 90, 180, 270])
        self._agent_dir_choice = np.random.choice(self._possible_directions)

        agent_dir_radians = self._agent_dir_choice * math.pi / 180.0

        self.place_agent(dir=agent_dir_radians, room=main_arm)

    def step(self, action):
        obs, reward, done, info = super().step(action)

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

        if self.near(self.box_left_arm):
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

        if self.near(self.box_right_arm):
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


class YMazeTask1(YMaze):
    def __init__(self):
        super().__init__(task_id=0)


class YMazeTask2(YMaze):
    def __init__(self):
        super().__init__(task_id=1)
