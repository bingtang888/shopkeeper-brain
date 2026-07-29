"""
AnswerOutPutNode 单元测试

测试范围：
1. _format_retrieval_context — 检索上下文格式化 + 元数据拼接 + 长度截断
2. _format_chat_history — 历史对话格式化 + 角色映射 + 长度截断
3. _build_prompt — 提示词组装（4个占位符填充）
4. _push_exist_answer — 已有答案推送（非流式写入任务队列）
5. process — 已有答案场景（跳过 LLM）
"""

import os
import sys

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutPutNode
from knowledge.utils.task_util import get_task_result, set_task_result, _tasks_result


def test_format_retrieval_context():
    """测试 _format_retrieval_context：元数据拼接 + 长度控制"""
    print("=" * 60)
    print("测试 1: _format_retrieval_context")
    print("=" * 60)

    node = AnswerOutPutNode()

    # ===== 用例 1：正常文档，含完整元数据 =====
    docs = [
        {
            "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。",
            "chunk_id": "local_1",
            "title": "主板维修手册",
            "source": "local",
            "url": "",
            "score": 0.9937,
        },
        {
            "content": "今天中午去吃猪脚饭吧。",
            "chunk_id": "local_2",
            "title": "闲聊",
            "source": "local",
            "url": "",
            "score": 0.0086,
        },
        {
            "content": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。",
            "chunk_id": None,
            "title": "短路查修指南",
            "source": "web",
            "url": "https://example.com/repair",
            "score": 0.9814,
        },
    ]

    formatted, remaining = node._format_retrieval_context(docs, max_context_chars=12000)

    print("\n【格式化后的上下文】:")
    print(formatted)
    print(f"\n【剩余可用字符数】: {remaining}")

    # 验证
    assert "[文档:1]" in formatted, "应包含文档编号 [文档:1]"
    assert "[chunk_id=local_1]" in formatted, "应包含 chunk_id"
    assert "[title=主板维修手册]" in formatted, "应包含 title"
    assert "[source=local]" in formatted, "应包含 source"
    assert "[score=0.993700]" in formatted, "应包含 6 位小数 score"
    assert "https://example.com/repair" in formatted, "应包含 url"
    assert "\n\n" in formatted, "文档之间应用 \\n\\n 分隔"
    assert remaining == 12000 - len(formatted), "剩余字符数应等于总额减去已用"

    # ===== 用例 2：content 为空的文档应被跳过 =====
    docs_with_empty = [
        {"content": "", "source": "local"},
        {"content": "有效内容", "source": "local", "score": 0.9},
    ]
    formatted2, remaining2 = node._format_retrieval_context(docs_with_empty, max_context_chars=12000)
    assert "[文档:1]" not in formatted2, "空 content 文档应被跳过，编号从 2 开始"
    assert "[文档:2]" in formatted2, "第二个文档编号应为 2"
    print("\n【用例 2 - 空 content 跳过】:")
    print(formatted2)

    # ===== 用例 3：超长截断 =====
    long_docs = [
        {"content": "A" * 100, "source": "local"},
        {"content": "B" * 100, "source": "local"},
        {"content": "C" * 100, "source": "local"},
    ]
    formatted3, remaining3 = node._format_retrieval_context(long_docs, max_context_chars=150)
    # 150 字符只能放下第 1 篇（100 + 元数据），第 2 篇放不下
    assert "A" * 100 in formatted3, "第 1 篇应保留"
    assert "B" * 100 not in formatted3, "第 2 篇应被截断"
    print("\n【用例 3 - 超长截断】:")
    print(f"保留内容长度: {len(formatted3)}, 剩余: {remaining3}")

    print("\n✅ test_format_retrieval_context 通过\n")


def test_format_chat_history():
    """测试 _format_chat_history：角色映射 + 长度控制"""
    print("=" * 60)
    print("测试 2: _format_chat_history")
    print("=" * 60)

    node = AnswerOutPutNode()

    # ===== 用例 1：正常历史对话 =====
    history = [
        {"role": "user", "text": "万用表怎么用？"},
        {"role": "assistant", "text": "万用表可以测量电压、电流、电阻。"},
        {"role": "user", "text": "怎么测这块主板的短路问题？"},
    ]

    formatted = node._format_chat_history(history, usage_chars=12000)

    print("\n【格式化后的历史对话】:")
    print(formatted)

    assert "用户: 万用表怎么用？" in formatted, "user 应映射为 用户"
    assert "助手: 万用表可以测量电压、电流、电阻。" in formatted, "assistant 应映射为 助手"
    assert "\n" in formatted, "消息之间应用 \\n 分隔"

    # ===== 用例 2：空 text 和未知 role 应被跳过 =====
    history_with_invalid = [
        {"role": "user", "text": ""},                    # text 为空 → 跳过
        {"role": "system", "text": "系统提示"},           # role 不在映射中 → 跳过
        {"role": "assistant", "text": "有效回复"},        # 正常
    ]
    formatted2 = node._format_chat_history(history_with_invalid, usage_chars=12000)
    assert "系统提示" not in formatted2, "system role 应被跳过"
    assert "助手: 有效回复" in formatted2, "assistant 应保留"
    print("\n【用例 2 - 无效消息跳过】:")
    print(formatted2)

    # ===== 用例 3：长度截断 =====
    long_history = [
        {"role": "user", "text": "A" * 30},
        {"role": "assistant", "text": "B" * 30},
        {"role": "user", "text": "C" * 30},
    ]
    formatted3 = node._format_chat_history(long_history, usage_chars=50)
    # 50 字符只能放下第 1 条（30 + "用户: " = 35），第 2 条放不下
    assert "A" * 30 in formatted3, "第 1 条应保留"
    assert "B" * 30 not in formatted3, "第 2 条应被截断"
    print("\n【用例 3 - 长度截断】:")
    print(f"保留内容: {repr(formatted3)}")

    print("\n✅ test_format_chat_history 通过\n")


def test_build_prompt():
    """测试 _build_prompt：4 个占位符填充"""
    print("=" * 60)
    print("测试 3: _build_prompt")
    print("=" * 60)

    node = AnswerOutPutNode()

    state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "item_names": ["万用表", "主板"],
        "reranked_docs": [
            {
                "content": "主板短路通常表现为通电后风扇转一下就停。",
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "source": "local",
                "score": 0.9937,
            },
        ],
        "history": [
            {"role": "user", "text": "万用表怎么用？"},
            {"role": "assistant", "text": "万用表可以测量电压、电流、电阻。"},
        ],
    }

    prompt = node._build_prompt(state)

    print("\n【组装后的提示词】:")
    print(prompt)

    # 验证 4 个占位符都被填充
    assert "暂无检索到上下文" not in prompt, "context 应有内容"
    assert "暂无历史上下文" not in prompt, "history 应有内容"
    assert "万用表,主板" in prompt, "item_names 应被逗号拼接"
    assert "怎么测这块主板的短路问题？" in prompt, "question 应填充"
    assert "【参考内容】" in prompt, "应包含模板标签"
    assert "【历史对话】" in prompt
    assert "【相关商品/实体】" in prompt
    assert "【用户问题】" in prompt

    print("\n✅ test_build_prompt 通过\n")


def test_build_prompt_empty():
    """测试 _build_prompt：无检索结果 + 无历史对话的降级"""
    print("=" * 60)
    print("测试 4: _build_prompt（空数据降级）")
    print("=" * 60)

    node = AnswerOutPutNode()

    state = {
        "rewritten_query": "你好",
        "item_names": [],
        "reranked_docs": [],
        "history": [],
    }

    prompt = node._build_prompt(state)

    print("\n【降级提示词】:")
    print(prompt)

    assert "暂无检索到上下文" in prompt, "无检索结果应降级"
    assert "暂无历史上下文" in prompt, "无历史对话应降级"

    print("\n✅ test_build_prompt_empty 通过\n")


def test_push_exist_answer_non_stream():
    """测试 _push_exist_answer：非流式模式写入任务队列"""
    print("=" * 60)
    print("测试 5: _push_exist_answer（非流式）")
    print("=" * 60)

    node = AnswerOutPutNode()

    task_id = "test_task_001"
    test_answer = "我不确定您指的是哪款产品。您是在询问以下产品吗：RS-12、RS-13？"

    state = {
        "is_stream": False,
        "task_id": task_id,
        "answer": test_answer,
    }

    # 调用
    node._push_exist_answer(task_id=task_id, is_stream=False, state=state)

    # 验证任务队列
    result = get_task_result(task_id=task_id, key="answer")
    print(f"\n【任务队列中的答案】: {result}")

    assert result == test_answer, "非流式模式应写入任务队列"

    # 清理
    if task_id in _tasks_result:
        del _tasks_result[task_id]

    print("\n✅ test_push_exist_answer_non_stream 通过\n")


def test_process_with_exist_answer():
    """测试 process：已有 answer 场景（跳过 LLM）"""
    print("=" * 60)
    print("测试 6: process（已有 answer，非流式）")
    print("=" * 60)

    node = AnswerOutPutNode()

    task_id = "test_task_002"
    exist_answer = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"

    state = {
        "is_stream": False,
        "task_id": task_id,
        "session_id": "test_sess_001",
        "original_query": "那个东西怎么用？",
        "rewritten_query": "那个东西怎么用？",
        "item_names": [],
        "answer": exist_answer,
    }

    # 调用 process（会走 _push_exist_answer + save_history）
    # 注意：save_history 会尝试写 MongoDB，如果连不上会记日志但不报错
    result = node.process(state)

    print(f"\n【state['answer']】: {result.get('answer')}")
    print(f"【state['prompt']】: {result.get('prompt', '(无)')}")

    # 验证
    assert result.get("answer") == exist_answer, "answer 应保持不变"
    assert "prompt" not in result, "已有 answer 时不应生成 prompt"

    # 任务队列应有答案
    task_result = get_task_result(task_id=task_id, key="answer")
    assert task_result == exist_answer, "非流式应写入任务队列"
    print(f"【任务队列答案】: {task_result}")

    # 清理
    if task_id in _tasks_result:
        del _tasks_result[task_id]

    print("\n✅ test_process_with_exist_answer 通过\n")


def test_format_retrieval_context_score_none():
    """测试 _format_retrieval_context：score 为 None 时不拼接"""
    print("=" * 60)
    print("测试 7: _format_retrieval_context（score=None）")
    print("=" * 60)

    node = AnswerOutPutNode()

    docs = [
        {"content": "测试内容", "source": "local", "score": None},
    ]

    formatted, remaining = node._format_retrieval_context(docs, max_context_chars=12000)

    print(f"\n【格式化结果】:\n{formatted}")

    assert "[score=" not in formatted, "score 为 None 时不应拼接 score 元数据"
    assert "测试内容" in formatted, "content 应保留"

    print("\n✅ test_format_retrieval_context_score_none 通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AnswerOutPutNode 单元测试")
    print("=" * 60 + "\n")

    test_format_retrieval_context()
    test_format_chat_history()
    test_build_prompt()
    test_build_prompt_empty()
    test_push_exist_answer_non_stream()
    test_process_with_exist_answer()
    test_format_retrieval_context_score_none()

    print("=" * 60)
    print("🎉 全部测试通过！")
    print("=" * 60)
