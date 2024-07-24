from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class VoiceActivityDetectorRequest(_message.Message):
    __slots__ = ("audio_data",)
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    def __init__(self, audio_data: _Optional[bytes] = ...) -> None: ...

class VoiceActivityDetectorResponse(_message.Message):
    __slots__ = ("confidence",)
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    confidence: float
    def __init__(self, confidence: _Optional[float] = ...) -> None: ...
