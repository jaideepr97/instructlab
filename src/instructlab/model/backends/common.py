# Standard
from typing import Tuple
import contextlib
import json
import logging
import multiprocessing
import pathlib
import socket
import typing

# Local
from ...configuration import get_model_family

from instructlab.training.chat_templates import ibm_legacy_tmpl as granite_legacy
from instructlab.training.chat_templates import mistral_tmpl as mistral
from instructlab.training.chat_templates import ibm_generic_tmpl as granite

logger = logging.getLogger(__name__)

API_ROOT_WELCOME_MESSAGE = "Hello from InstructLab! Visit us at https://instructlab.ai"
CHAT_TEMPLATE_AUTO = "auto"
CHAT_TEMPLATE_TOKENIZER = "tokenizer"
LLAMA_CPP = "llama-cpp"
VLLM = "vllm"
templates = [
    {
        "model": "granite",
        "template": granite.CHAT_TEMPLATE,
        "special_tokens": granite.SPECIAL_TOKENS,
    },
    {
        "model": "granite_legacy",
        "template": granite_legacy.CHAT_TEMPLATE,
        "special_tokens": granite_legacy.SPECIAL_TOKENS,
    },
    {
        "model": "mixtral",
        "template": mistral.CHAT_TEMPLATE,
        "special_tokens": mistral.SPECIAL_TOKENS,
    },
]


class Closeable(typing.Protocol):
    def close(self) -> None: ...


class ServerException(Exception):
    """An exception raised when serving the API."""

def get_in_memory_model_template(
    model_family: str, model_path: pathlib.Path
) -> Tuple[str, str, str]:
    resolved_family = get_model_family(model_family, model_path)
    logger.debug(
        "Searching hard coded model templates for model family %s's template",
        resolved_family,
    )
    for template_dict in templates:
        if template_dict["model"] == resolved_family:
            template = template_dict["template"]
            bos_token = template_dict["special_tokens"].bos.token
            eos_token = template_dict["special_tokens"].eos.token

    return template, eos_token, bos_token


def format_template(template: str, bos_token: str, eos_token: str) -> str:
    prefix = ""
    if eos_token:
        prefix = '{{% set eos_token = "{}" %}}\n'.format(eos_token)
    if bos_token:
        prefix = '{}{{% set bos_token = "{}" %}}\n'.format(prefix, bos_token)

    return prefix + template


def get_model_template(
    model_family: str, model_path: pathlib.Path
) -> Tuple[str, str, str]:
    """
    Read the chat template from the model's tokenizer config if available. If not,
    fallback to in-memory templates

    args
        model_path (str): Path to the model, used to read the tokenizer config if available
        model_family (str): model family used to map in-memory template if needed
    returns
        template (str): resolved chat template
        bos_token (str): Beginning of sentence token
        eos_token (str): End of sentence token
    """

    template = eos_token = bos_token = ""
    try:
        with open(
            pathlib.Path(model_path) / "tokenizer_config.json",
            "r",
            encoding="utf-8",
        ) as f:
            tcfg = json.load(f)
        template = tcfg["chat_template"]
        bos_token = tcfg["bos_token"]
        eos_token = tcfg["eos_token"]

        # Handle edge case: some 7b models (llama architecture) may contain a sub-optimal chat template in their
        # tokenizer configs. Check if the template accounts for generation prompt addition
        # If this is not the case we must patch the chat template to make sure the model does not lose performance.
        with open(pathlib.Path(model_path) / "config.json", "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        if model_cfg["model_type"].lower() == "llama":
            if "add_generation_prompt" not in template:
                template, eos_token, bos_token = get_in_memory_model_template(
                    "granite_legacy", model_path
                )

    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.warning(
            f"Unable to read tokenizer config for model: {model_path}: {e}. Falling back to in-memory chat template mapping"
        )
        template, eos_token, bos_token = get_in_memory_model_template(
            model_family, model_path
        )
        if template == "":
            raise ValueError(
                "Unable to find an appropriate chat template for supplied model"
            )

    # if present, replace token placeholders with actual tokens in the chat template
    template = format_template(
        template=template, bos_token=bos_token, eos_token=eos_token
    )

    return template, eos_token, bos_token

def is_temp_server_running():
    """Check if the temp server is running."""
    return multiprocessing.current_process().name != "MainProcess"


def verify_template_exists(path):
    if not path.exists():
        raise FileNotFoundError("Chat template file does not exist: {}".format(path))

    if not path.is_file():
        raise IsADirectoryError(
            "Chat templates paths must point to a file: {}".format(path)
        )


def free_tcp_ipv4_port(host: str) -> int:
    """Ask the OS for a random, ephemeral, and bindable TCP/IPv4 port

    Note: The idea of finding a free port is bad design and subject to
    race conditions. Instead vLLM and llama-cpp should accept port 0 and
    have an API to return the actual listening port. Or they should be able
    to use an existing socket like a systemd socket activation service.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[-1])


def safe_close_all(resources: typing.Iterable[Closeable]):
    for resource in resources:
        with contextlib.suppress(Exception):
            resource.close()
