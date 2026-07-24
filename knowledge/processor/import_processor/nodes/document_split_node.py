import json
import os
import re
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):

    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分的核心入口逻辑
        (原文档打散) -- (组装)
        Args:
            state:

        Returns:

        """
        config = self.config
        # 1、参数校验
        md_content, file_title,max_content_length,min_content_length = self._validate_state(state, config)

        # 2、切分(一级策略：根据md文档中的标题来切分)多个章节(章节：标题之间的内容)
        sections:List[Dict[str,Any]] = self._split_by_headings(md_content,file_title)

        # 3、二次切分或者合并
        final_sections = self._split_and_merge(
            sections,
            config.max_content_length,
            config.min_content_length
        )
        # 4、组装成chunk对象
        final_sections = self._assemble_chunks(final_sections)
        # 5、备份
        self._backup_chunks(final_sections, state)
        # 6、更新state(chunks)
        state["chunks"] = final_sections

        # 7、返回
        return state

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:

        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')
        if not isinstance(md_content, str) or not md_content.strip():
            raise ValueError("md_content 不能为空")

        # 2. 统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title') or ""

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content:str, file_title:str) -> List[Dict[str,Any]]:

        """
        parent_title: 封装的原因主要是为了后面短section在合并的时候有一个判断标准(同源：同一个父标题)
        根据标题来切分(# {1,6}都有可能)
        Args:
            md_content: 切分的md
            file_title: 上传文档的标题

        Returns:
            List[Dict[str,Any]]:切分后的多个章节
        """

        in_fence = False  # 是否在代码块内
        body_lines = []
        sections = [] # 最终收集到的章节对象
        current_title = ""
        hierarchy = [""] * 7 # (数组)存储所有标题内容(作为section的副标题使用)  标题层级追踪搜索
        current_level = 0

        def _flush() -> List[Dict[str,Any]]:
            """
            打包section
            {
              "body":"收集到的所有行"
              "title":"当前内容的标题"
              "parent_title":"当前内容的父标题"(最麻烦)
              "file_title":文档标题(最简单)
            }
            Returns:
            如果current_title没有，body有  能进入打包section,也有意义
            如果current_title有，body没有，也打包成section：在合并阶段可以保留上【可选，建议留下来】  在后续合并阶段没有任何影响
            如果current_title有,body也有 能进入打包成section【一定留】
            如果current_title没有，body也没有，不会进入(不能打包)
            """
            # 1、处理内容行
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,  # 内容标题
                    "parent_title": parent_title,  # 内容父标题
                    "file_title": file_title
                })
                body_lines.clear()



        # 1、根据\n切分md_content
        md_lines = md_content.split("\n")

        # 2、定义正则(正则的规则是从MD中找标题#{1,6}) ():捕获组：产生三个group(0):全拿   group(1):几个#号  group(2):标题的内容
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3、遍历切分后md_lines
        for md_line in md_lines:
            # 3.1 检测代码围栏边界(```或~~~)
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence  # 不要用固定true or false

            # 3.2 判断是否要走正则
            match = heading_re.match(md_line) if not in_fence else None

            # 3.3 判断match 是否有
            # 代表匹配到了标题而且一定是非代码块中的 # 标题
            if match:

                # 将 body_lines中收集到的行封装到section对象
                _flush()

                current_title = match.group(2).strip()  #当前标题
                level = len(match.group(1))  #当前标题的层级(# {1,6})
                current_level = level
                hierarchy[level] = current_title  #写入操作

                for i in range(level +1, 7):
                    hierarchy[i] = ""
            # 没有匹配到标题[普通行] 或者是代码块(加入)
            else:
                body_lines.append(md_line)

        _flush()

        return sections

    def _split_and_merge(self, sections:List[Dict[str,Any]], max_content_length:int, min_content_length:int) ->List[Dict[str,Any]]:
        """
        切分较大的章节(section)以及合并较小章节(section)
        Args:
            sections: 所有经过一级切分后的章节
            max_content_length: 最大内容长度(content:title+\n\n+body) 如果section中的内容长度超过阈值max_content_length,就需要进行切割，反之不需要进行切割(尽量保证不要太多的section进行二次切割)
            min_content_length: 最小内容长度：如果section中的内容长度不足阈值min_content_length,就需要对该section进行合并。同源机制合并(section相同的父标题才合，尽量保证确实比较小的内容才合并，不要把大多数的section都合并)


        Returns:
            先切后合
        """

        # 1、切分
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length))

        # 2、合并
        final_sections = self._merger_short_section(current_sections, min_content_length, max_content_length)
        return final_sections

    def _split_long_section(self, section:Dict[str,Any], max_content_length:int) ->List[Dict[str,Any]]:
        """
        切分的章节内容
        Args:
            section: 当前章节
            max_content_length: 最大长度阈值

        Returns:
            List[Dict[str,Any]]
        """

        # 1、获取section对象中的属性
        body = section.get("body") or ""          #行内容
        title = section.get("title") or ""        #标题
        parent_title = section.get("parent_title") or ""  #父标题
        file_title = section.get("file_title") or ""    # 文档标题

        #
        if len(title) > 80:
            title = title[:80]  # 防御性编程：title 本身就超长的极端情况

        processed_body = MarkdownTableLinearizer.process(body)
        if processed_body != body:
            self.logger.info("检查到 section 中有表格，已完成线性化处理")
        body = processed_body
        section["body"] = body
        section["title"] = title

        # 2、获取标题前缀
        title_prefix = f"{title}\n\n"

        # 3、获取总长度(标题(前缀)+body]
        total_length = len(title_prefix) + len(body)

        # 4、判断总长度是否超过阈值
        if total_length <= max_content_length:
            return [section]

        # 5、能切分的内容长度计算出来(body)
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]  # 防御性编程：title 本身就超长的极端情况

        # 6、切分(6.1 用谁切：LangChain的切分器：递归切分器  6.2：去切谁：body)
        # 6.1 切分器对象  # 1、chunk_size:chunk块的大小  2、chunk_overlap块与块之间的重叠的字符数  3、separators:分割符【"\n\n","\n"," ",""】
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n","\n","。","？","！","；",".",",","?","!",";"," ",""],
            keep_separator=True)

        # 6.2 切分器对象切分
        sections = text_spliter.split_text(body)

        # 6.3判断
        if len(sections) == 1:
            return [section]

        # 6.4 遍历
        sub_sections = []
        for index, section in enumerate(sections):
            sub_sections.append({
                "body":section,
                "title":f"{title}_{index+1}",
                "parent_title":parent_title,
                "file_title":file_title
            })

        # 7.返回
        return sub_sections

    def _merger_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int, max_content_length: int) -> List[Dict[str, Any]]:
        """
        合并短的章节：
        短章节来源：
        来源1：原本根据一级标题切分之后内容就很短
        来源2：(LangChain递归切分器)二次切分之后可能有很短的内容
        合并策略：
        条件1：section很短比最小阈值还小
        条件2：同源(父标题相同)
        Args:
            current_sections: 二次切分后的所有section对象
            min_content_length: 每一个section的内容最小的长度

        Returns:
            合并之后的section对象

            贪心累加算法
        """

        # 1. 初始化
        if not current_sections:
            return []

        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (
                current_section.get('parent_title', '')
                == next_section.get('parent_title', '')
            )
            current_body = current_section.get('body') or ''
            next_body = next_section.get('body') or ''

            merged_body = current_body.rstrip() + "\n\n" + next_body.lstrip()
            merged_title = current_section.get('title') or ''
            next_title = next_section.get('title') or ''
            candidate_title = merged_title
            if merged_title and next_title and next_title not in merged_title:
                candidate_title = f"{merged_title}、{next_title}"

            merged_content_length = len(f"{candidate_title}\n\n{merged_body}")
            merged_too_long = merged_content_length > max_content_length

            if same_parent and len(current_body) < min_content_length and not merged_too_long:
                # 合并 body
                current_section['body'] = merged_body
                # 标题保留原始标题，用逗号拼接被合并的标题
                current_section['title'] = candidate_title
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section

        # 最后一个封箱
        final_sections.append(current_section)

        return final_sections

    def _assemble_chunks(self, final_sections:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        """
        组装最后的chunks
        Args:
            final_sections:

        Returns:

        """
        final_chunks = []
        for section in final_sections:
            
            body = section.get("body") or ""
            title = section.get("title") or ""
            file_title = section.get("file_title") or ""
            parent_title = section.get("parent_title") or ""
            content = f"{title}\n\n{body}"
            final_chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title
            })
        self.logger.info(f"最终切割后能够进入到嵌入节点的chunk个数：{len(final_chunks)}")
        return final_chunks

    def _backup_chunks(self, final_chunks, state:ImportGraphState):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        try:
            os.makedirs(local_dir, exist_ok=True)  # 如果目录存在，不报错
            output_path = os.path.join(local_dir, "chunks.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == "__main__":
    document_split_node = DocumentSplitNode()

    md_path = r"D:\forAI\project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\auto\万用表的使用_new.md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_tables = MarkdownTableLinearizer.HTML_TABLE_PATTERN.findall(md_content)
    markdown_tables = MarkdownTableLinearizer.MD_TABLE_PATTERN.findall(md_content)
    linearized_md_content = MarkdownTableLinearizer.process(md_content)

    print(json.dumps({
        "source": md_path,
        "html_table_count": len(html_tables),
        "markdown_table_count": len(markdown_tables),
        "linearized_document_changed": linearized_md_content != md_content,
    }, ensure_ascii=False))

    source_tables = [("html", table) for table in html_tables]
    source_tables.extend(("markdown", table) for table in markdown_tables)
    for index, (table_type, table_content) in enumerate(source_tables, start=1):
        linearized_table = MarkdownTableLinearizer.process(table_content)
        print(json.dumps({
            "table_index": index,
            "type": table_type,
            "before": table_content.strip(),
            "after": linearized_table.strip(),
            "changed": linearized_table != table_content,
        }, ensure_ascii=False))

    init_state = {
        "md_content": md_content,
        "file_title": "万用表的使用"
    }
    result = document_split_node.process(init_state)
    sections = result.get("chunks", [])
    final_content = "\n".join(section.get("content", "") for section in sections)

    print(f"\n切分完成，共 {len(sections)} 个 section")
    for index, section in enumerate(sections, start=1):
        content_preview = section["content"].replace("\n", " ").strip()
        if len(content_preview) > 120:
            content_preview = content_preview[:120] + "..."
        print(json.dumps({
            "index": index,
            "title": section["title"],
            "parent_title": section["parent_title"],
            "content_length": len(section["content"]),
            "content_preview": content_preview,
        }, ensure_ascii=False))

    for index, (table_type, table_content) in enumerate(source_tables, start=1):
        linearized_table = MarkdownTableLinearizer.process(table_content).strip()
        output_lines = [line for line in linearized_table.splitlines() if line.strip()]
        matched_lines = sum(1 for line in output_lines if line in final_content)
        print(json.dumps({
            "table_index": index,
            "type": table_type,
            "output_lines": len(output_lines),
            "matched_lines_in_final_chunks": matched_lines,
            "content_in_final_chunks": bool(output_lines) and matched_lines == len(output_lines),
        }, ensure_ascii=False))
