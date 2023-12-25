import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from dataclasses_json import config, dataclass_json

from .types import NDArray

__all__ = [
    "SerializableConfig",
    "DynamicConfig",
    "generate_config_registration",
    "generate_optional_config_generation",
    "make_numpy_field",
    "make_optional_numpy_field",
    "generate_list_config_field",
]


TConfig = TypeVar("TConfig", bound="SerializableConfig")
TDynamicConfig = TypeVar("TDynamicConfig", bound="DynamicConfig")


@dataclass_json
@dataclasses.dataclass()
class SerializableConfig:
    def serialize(self) -> str:
        return self.to_json()  # type: ignore

    def serialize_to_dict(self) -> Dict[str, Any]:
        return self.to_dict()  # type: ignore

    @classmethod
    def deserialize(cls: Type[TConfig], serialized_config: str) -> TConfig:
        return cls.from_json(serialized_config)  # type: ignore

    @classmethod
    def deserialize_from_dict(
        cls: Type[TConfig], dict_config: Dict[str, Any]
    ) -> TConfig:
        return cls.from_dict(dict_config)  # type: ignore

    @classmethod
    def deserialize_from_file(cls: Type[TConfig], path: str) -> TConfig:
        with open(path, "r") as f:
            return cls.deserialize(f.read())


class DynamicConfig(SerializableConfig):
    @staticmethod
    def get_type() -> str:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ConfigMetadata:
    base_cls: Type[DynamicConfig]
    encoder: Callable[[DynamicConfig], Dict[str, Any]]
    decoder: Callable[
        [Dict[str, Any]], Union[DynamicConfig, Optional[DynamicConfig]]
    ]
    config_list: Dict[str, Type[DynamicConfig]]

    def add_config(self, name: str, new_config: Type[DynamicConfig]) -> None:
        assert name not in self.config_list, f"{name} is already registered"
        self.config_list[name] = new_config


CONFIG_STORAGE: Dict[Type[DynamicConfig], ConfigMetadata] = {}


def generate_config_registration(
    base_cls: Type[TDynamicConfig],
    default_factory: Optional[Callable[[], TDynamicConfig]] = None,
) -> Tuple[
    Callable[[Type[TDynamicConfig]], None], Callable[[], TDynamicConfig]
]:
    CONFIG_LIST: Dict[str, Type[TDynamicConfig]] = {}

    def register_config(cls: Type[TDynamicConfig]) -> None:
        assert issubclass(cls, base_cls)
        type_name = cls.get_type()
        is_registered = type_name in CONFIG_LIST
        assert not is_registered, f"{type_name} seems to be already registered"
        CONFIG_LIST[type_name] = cls

    def _encoder(orig_config: TDynamicConfig) -> Dict[str, Any]:
        return {
            "type": orig_config.get_type(),
            "params": orig_config.serialize_to_dict(),
        }

    def _decoder(dict_config: Dict[str, Any]) -> TDynamicConfig:
        name = dict_config["type"]
        params = dict_config["params"]
        return CONFIG_LIST[name].deserialize_from_dict(params)

    config_metadata = ConfigMetadata(
        base_cls=base_cls,
        encoder=_encoder,  # type: ignore
        decoder=_decoder,
        config_list=CONFIG_LIST,  # type: ignore
    )
    CONFIG_STORAGE[base_cls] = config_metadata

    if default_factory is None:

        def make_field() -> TDynamicConfig:
            field = cast(
                TDynamicConfig,
                dataclasses.field(
                    metadata=config(encoder=_encoder, decoder=_decoder)
                ),
            )
            return field

    else:

        def make_field() -> TDynamicConfig:
            return dataclasses.field(
                metadata=config(encoder=_encoder, decoder=_decoder),
                default_factory=default_factory,
            )

    return register_config, make_field


def generate_optional_config_generation(
    base_cls: Type[TDynamicConfig],
) -> Tuple[
    Callable[[Type[TDynamicConfig]], None],
    Callable[[], Optional[TDynamicConfig]],
]:
    CONFIG_LIST: Dict[str, Type[TDynamicConfig]] = {}

    def register_config(cls: Type[TDynamicConfig]) -> None:
        assert issubclass(cls, base_cls)
        type_name = cls.get_type()
        is_registered = type_name in CONFIG_LIST
        assert not is_registered, f"{type_name} seems to be already registered"
        CONFIG_LIST[type_name] = cls

    def _encoder(orig_config: Optional[TDynamicConfig]) -> Dict[str, Any]:
        if orig_config is None:
            return {"type": "none", "params": {}}
        return {
            "type": orig_config.get_type(),
            "params": orig_config.serialize_to_dict(),
        }

    def _decoder(dict_config: Dict[str, Any]) -> Optional[TDynamicConfig]:
        name = dict_config["type"]
        params = dict_config["params"]
        if name == "none":
            return None
        return CONFIG_LIST[name].deserialize_from_dict(params)

    config_metadata = ConfigMetadata(
        base_cls=base_cls,
        encoder=_encoder,  # type: ignore
        decoder=_decoder,
        config_list=CONFIG_LIST,  # type: ignore
    )
    CONFIG_STORAGE[base_cls] = config_metadata

    def make_field() -> Optional[TDynamicConfig]:
        return dataclasses.field(
            metadata=config(encoder=_encoder, decoder=_decoder),
            default=None,
        )

    return register_config, make_field


def generate_list_config_field(
    base_cls: Type[TDynamicConfig],
) -> Callable[[], Sequence[TDynamicConfig]]:
    assert base_cls in CONFIG_STORAGE

    config_metadata = CONFIG_STORAGE[base_cls]

    def _encoder(
        orig_config: Sequence[TDynamicConfig],
    ) -> Sequence[Dict[str, Any]]:
        return [config_metadata.encoder(config) for config in orig_config]

    def _decoder(
        dict_config: Sequence[Dict[str, Any]]
    ) -> Sequence[TDynamicConfig]:
        configs = [config_metadata.decoder(config) for config in dict_config]
        return configs  # type: ignore

    def make_field() -> Sequence[TDynamicConfig]:
        return dataclasses.field(
            metadata=config(encoder=_encoder, decoder=_decoder),
            default_factory=list,
        )

    return make_field


# setup numpy encoder/decoder
def _numpy_encoder(v: NDArray) -> Sequence[float]:
    return v.tolist()  # type: ignore


def _numpy_decoder(v: Sequence[float]) -> NDArray:
    return np.array(v)


def make_numpy_field() -> NDArray:
    return dataclasses.field(  # type: ignore
        metadata=config(encoder=_numpy_encoder, decoder=_numpy_decoder),
    )


def make_optional_numpy_field() -> Optional[NDArray]:
    return dataclasses.field(
        metadata=config(encoder=_numpy_encoder, decoder=_numpy_decoder),
        default=None,
    )
