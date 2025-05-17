import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces


class Hallway(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    at the end of a hallway
    """

    def __init__(self, length=12, agent_pos=[-6.5, 0, -6.5], **kwargs):
        assert length >= 2
        self.length = length

        super().__init__(max_episode_steps=250, **kwargs)

        # Default - Allow only movement actions (left/right/forward)
        # self.action_space = spaces.Discrete(self.actions.move_forward + 1)

        # Modified by Raymond - Allow only movement actions (left/right/forward/backward)
        self.agent_pos = agent_pos
        self._actions_name = ["Turn left", "Turn right", "Forward", "Move back"]
        self.action_space = spaces.Discrete(len(self._actions_name))  # 0-indexed

        # random positions for the agent
        # agent_x = np.random.uniform(-6.25, 6.25)
        # agent_z = np.random.uniform(-6.25, 6.25)

        # random direction for the agent
        self._possible_directions = np.asarray([0, 90, 180, 270])
        self._agent_dir_choice = np.random.choice(self._possible_directions)

        # convert agent_dir to radians
        # agent_dir_radians = self._agent_dir_choice * math.pi / 180.0

        # self.agent_pos = [agent_x, 0.0, agent_z]
        # self.agent_pos = None
        self.agent.pos = [self.agent_pos[0], 0.0, self.agent_pos[2]]

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(min_x=-1, max_x=-1 + self.length, min_z=-2, max_z=2)

        # Default: Place the box at the end of the hallway
        # self.box = self.place_entity(Box(color="red"), min_x=room.max_x - 2)

        # Default: Place the agent a random distance away from the goal
        # self.place_agent(
        #     dir=self.rand.float(-math.pi / 4, math.pi / 4), max_x=room.max_x - 2
        # )

        # Modified by Raymond - Place the box at the same position
        self.box = self.place_entity(Box(color="red"), pos=np.array([10.08, 0.0, 0.1]), dir=0)

        # Modified by Raymond - Place the agent at a specific position and direction
        self._possible_directions = np.asarray([0, 90, 180, 270])
        self._agent_dir_choice = np.random.choice(self._possible_directions)

        agent_dir_radians = self._agent_dir_choice * math.pi / 180.0

        self.place_agent(dir=agent_dir_radians)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
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

        if self.near(self.box):
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
