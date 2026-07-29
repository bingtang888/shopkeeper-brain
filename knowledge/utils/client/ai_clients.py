import json
import os
import threading
from pathlib import Path
from typing import Optional
from langchain_openai import ChatOpenAI

from openai import OpenAI
import httpx
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from FlagEmbedding import FlagReranker

from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_client_json: Optional[ChatOpenAI] = None
    _openai_llm_lock_json = threading.Lock()

    _openai_llm_client_text: Optional[ChatOpenAI] = None
    _openai_llm_lock_text = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    # ── VLM ──

    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            http_client = httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(60.0, connect=15.0)
            )
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client
            )
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e


    # -- LLM --
    @classmethod
    def get_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        """获取 LLM 客户端
        Args:
            response_format: True=JSON输出模式, False=普通文本输出模式
        """
        if response_format:
            return cls._get_or_create("_openai_llm_client_json", cls._openai_llm_lock_json,
                                      lambda: cls._create_llm_client(response_format=True))
        else:
            return cls._get_or_create("_openai_llm_client_text", cls._openai_llm_lock_text,
                                      lambda: cls._create_llm_client(response_format=False))

    @classmethod
    def _create_llm_client(cls, response_format:bool = True) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            model_kwargs = {}
            if response_format:
                model_kwargs["response_format"] = {"type": "json_object"}

            http_client = httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(60.0, connect=15.0)
            )
            llm_client = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key,
                base_url=base_url,
                model_kwargs=model_kwargs,
                http_client=http_client,
                timeout=60.0,
                max_retries=0
            )
            logger.info(f"OpenAI LLM 客户端初始化成功")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI LLM 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}")

    @classmethod
    def get_bge_m3_client(cls):
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        """
        创建bge_m3 客户端
        Returns:

        """
        try:
            # 1.获取环境变量
            model_name = cls._require_env("BGE_M3_PATH")

            # 2、创建
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,  # Specify the model name
                device='cpu',
                use_fp16=False
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败：{e}") from e

    # ── BGE-M3 Reranker ──

    @classmethod
    def get_bge_m3_rerank_client(cls) -> FlagReranker:
        return cls._get_or_create("_bge_m3_rerank_client", cls._bge_m3_rerank_lock, cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls) -> FlagReranker:
        """
        创建 BGE-Reranker-Large 客户端
        """
        try:
            model_name = cls._require_env("BGE_RERANKER_LARGE")
            device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
            use_fp16 = os.getenv("BGE_RERANKER_FP16", "0") == "1"

            reranker = FlagReranker(
                model_name_or_path=model_name,
                device=device,
                use_fp16=use_fp16
            )
            logger.info(f"BGE-Reranker 客户端初始化成功 (device={device}, fp16={use_fp16})")
            return reranker
        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"BGE-Reranker 客户端创建失败：{e}") from e


if __name__ == '__main__':
    llm_client: ChatOpenAI = AIClients.get_llm_client(False)

    llm_response = llm_client.invoke("请您给我讲一个笑话，要求输出格式是一个json")

    llm_result = llm_response.content

    result = json.loads(llm_result)
    print(llm_result)
