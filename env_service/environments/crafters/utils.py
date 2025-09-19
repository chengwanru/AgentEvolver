


from gym import spaces

# class EnvWrapper(gym.Wrapper):
#     """
#     A wrapper class for gym environments to standardize interactions across different environments.
#     It provides additional functionalities, such as handling specific observation processing,
#     managing action validity, retrieving instruction prompts, and tracking failed action candidates.
#     """
#
#     def __init__(self, env, env_name, task_name):
#         super().__init__(env)
#         self.env_name = env_name
#         self.task_name = task_name
#         self.failed_candidates = []
#
#     @property
#     def max_steps(self):
#         return self.env.max_steps
#
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return self._process_observation(obs), info
#
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         processed_obs = self._process_observation(obs)
#         return processed_obs, reward, terminated, truncated, info
#
#     def _process_observation(self, obs):
#         if self.env_name in ["nle", "minihack"]:
#             obs = obs
#         elif self.env_name == "babyai":
#             obs = obs
#         elif self.env_name == "textworld":
#             obs = obs
#         elif self.env_name == "babaisai":
#             obs = obs
#         elif self.env_name == "crafter":
#             obs = obs
#         else:
#             raise ValueError(f"Unknown environment: {self.env_name}")
#
#         return obs
#
#     @property
#     def actions(self):
#         # This property should return the list of available actions
#         return self.env.actions if hasattr(self.env, "actions") else list(range(len(self.env.action_space)))
#
#     def get_text_action(self, action):
#         return self.env.get_text_action(action)
#
#     def get_instruction_prompt(self, instructions=None):
#         if self.env_name == "nle":
#             from balrog.environments.nle import get_instruction_prompt
#
#             return get_instruction_prompt()
#         elif self.env_name == "minihack":
#             from balrog.environments.minihack import get_instruction_prompt
#
#             return get_instruction_prompt(self.env, self.task_name)
#         elif self.env_name == "babyai":
#             from balrog.environments.babyai_text import get_instruction_prompt
#
#             return get_instruction_prompt(self.env, mission=instructions)
#         elif self.env_name == "textworld":
#             from balrog.environments.textworld import get_instruction_prompt
#
#             return get_instruction_prompt(self.env, self.task_name)
#         elif self.env_name == "babaisai":
#             from balrog.environments.babaisai import get_instruction_prompt
#
#             return get_instruction_prompt(self.env, self.task_name)
#         elif self.env_name == "crafter":
#             from balrog.environments.crafter import get_instruction_prompt
#
#             return get_instruction_prompt(self.task_name)
#         else:
#             raise ValueError(f"Unknown environment: {self.env_namee}")
#
#     def check_action_validity(self, candidate_action):
#         valid_action = None
#         if candidate_action in self.env.language_action_space:
#             valid_action = candidate_action
#         else:
#             valid_action = self.env.default_action
#             self.failed_candidates.append(candidate_action)
#         return valid_action
#
#     def get_stats(self):
#         return self.env.get_stats()


class Strings(spaces.Space):
    """A custom Gym space for managing discrete string-based actions."""

    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def map(self, action):
        return self._dict[action]

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()


from collections import deque
from typing import List, Optional


# class Message:
#     """Represents a conversation message with role, content, and optional attachment."""
#
#     def __init__(self, role: str, content: str, attachment: Optional[object] = None):
#         self.role = role  # 'system', 'user', 'assistant'
#         self.content = content  # String content of the message
#         self.attachment = attachment
#
#     def __repr__(self):
#         return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"




class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning.

    Maintains a configurable history of text, images, and chain-of-thought reasoning to
    construct prompt messages for conversational agents.
    """

    def __init__(
            self,
            max_text_history: int = 16,
            max_image_history: int = 1,
            system_prompt: Optional[str] = None,
            max_cot_history: int = 1,
    ):
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.max_history = max(max_text_history, max_image_history)
        self.system_prompt = system_prompt
        self._events = deque(maxlen=self.max_history * 2)  # Stores observations and actions
        self._last_short_term_obs = None  # To store the latest short-term observation
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history, which can include text, an image, or both."""
        long_term_context = obs["text"].get("long_term_context", "")
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        text = long_term_context

        image = obs.get("image", None)

        # Add observation to events
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "image": image,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history, including reasoning if available."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_prompt(self, icl_episodes=False) -> List:
        """Generate a list of Message objects representing the prompt.

        Returns:
            List: Messages List constructed from the event history.
        """
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(
                {"role": "user", "content": self.system_prompt})



        # Determine which text observations to include
        text_needed = self.max_text_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include
        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        # determine the reasoning to include
        reasoning_needed = self.max_cot_history
        for event in reversed(self._events):
            if event["type"] == "action":
                if reasoning_needed > 0 and event.get("reasoning") is not None:
                    reasoning_needed -= 1
                else:
                    event["reasoning"] = None

        # Process events to create messages
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                message_parts = []

                if idx == len(self._events) - 1:
                    message_parts.append("Current Observation:")
                    if self._last_short_term_obs:
                        message_parts.append(self._last_short_term_obs)
                else:
                    message_parts.append("Observation:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])

                # image = None
                # if event.get("include_image", False):
                #     image = event["image"]
                #     message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message =  {"role": "user", "content": content} #Message(role="user", content=content)#, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text", "include_image"]:
                    if flag in event:
                        del event[flag]
            elif event["type"] == "action":
                if event.get("reasoning") is not None:
                    content = "Previous plan:\n" + event["reasoning"]
                else:
                    content = event["action"]
                message ={"role": "assistant", "content": content} #Message(role="assistant", content=content)
            messages.append(message)

        return messages
