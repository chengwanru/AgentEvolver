from dataclasses import dataclass

@dataclass
class EnvEntityOpt:
    name: str
    description: str

def get_crud_opts()->list[EnvEntityOpt]:
    # TODO
    pass

@dataclass
class EnvEntity:
    name: str
    description: str
    attrs: dict[str,str]
    opts: list[EnvEntityOpt]


class TaskPreference:
    def __init__(self, num_entities: int, num_opts: int, relation_difficulty: float):
        self._num_entities = num_entities
        self._num_opts = num_opts
        self._relation_difficulty = relation_difficulty
    
    # TODO implement getter for two vars
    
    @property
    def relation_difficulty(self)->str:
        # TODO implement different prompts for difficulty respectively
        pass
    

class UserProfile:
    def __init__(self, name: str, background:str, task: TaskPreference):
        self._name = name
        self._background=background
        self._entities: list[EnvEntity] = []
        self._task_preference = task
    
    def reg_entity(self, entity: EnvEntity):
        self._entities.append(entity)
    
    def reg_entities(self, entities: list[EnvEntity]):
        self._entities.extend(entities)
    
    def get_instruction(self)->str:
        # TODO generate instruction
        pass
        