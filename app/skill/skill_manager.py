import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from app.log.logger import LOG


class SkillManager:
    """
    Skill 管理器 - 管理项目中的 SKILL 文件

    Skill 是一种特殊的知识文件，以 YAML frontmatter + Markdown 格式存储在 SKILL.md 文件中。

    Skill 文件格式示例：
    ```
    ---
    name: translation
    description: 翻译技能
    ---

    ## 翻译指南
    ...
    ```

    Attributes:
        skills: List[Dict], 缓存的 skill 信息列表
        skill_paths: List[Path], SKILL 目录路径列表（从环境变量 SKILL_PATH 读取，支持逗号分隔）

    Note:
        SKILL_PATH 环境变量支持多个路径，用逗号分隔
    """

    def __init__(self):
        self.skills: List[Dict] = []
        # 支持多个路径，用逗号分隔
        skill_path_env = os.getenv("SKILL_PATH", "")
        self.skill_paths = [Path(p.strip()) for p in skill_path_env.split(",") if p.strip()]

    def _parse_frontmatter(self, content: str) -> Tuple[Dict, str]:
        """
        解析 YAML frontmatter

        从 MARKDOWN 内容中提取 frontmatter 元数据和正文

        Args:
            content: SKILL.md 文件内容

        Returns:
            Tuple[Dict, str]: (metadata dict, body content)
                - metadata: 解析后的 YAML 键值对
                - body: MARKDOWN 正文（不含 frontmatter）

        Example:
            >>> content = "---\\nname: test\\n---\\n# Body"
            >>> _parse_frontmatter(content)
            ({"name": "test"}, "# Body")

        Note:
            如果没有 frontmatter，返回空字典和完整内容
        """
        match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if not match:
            return {}, content
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def _read_skill(self, skill_path: Path) -> Optional[Dict]:
        """
        读取单个 skill 目录

        读取指定目录下的 SKILL.md 文件，解析 frontmatter 和正文

        Args:
            skill_path: Skill 目录路径

        Returns:
            Dict: 成功时返回 skill 信息字典
                {
                    "path": str,           # skill 目录完整路径
                    "name": str,           # skill 名称（来自 frontmatter）
                    "description": str,    # skill 描述（来自 frontmatter）
                    "body": str            # skill 正文（不含 frontmatter）
                }
            None: 如果 SKILL.md 不存在
            Dict: 错误信息字典，如果读取失败

        Note:
            自动处理编码为 UTF-8
        """
        skill_md = skill_path / "SKILL.md"

        if not skill_md.exists():
            return None

        try:
            content = skill_md.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(content)

            return {
                "path": str(skill_path),
                "name": meta.get("name"),
                "description": meta.get("description"),
                "body": body.strip() if body else "",
            }
        except Exception as e:
            return {"read skill error, path": str(skill_path), "error": str(e)}

    def _scan_skills(self) -> None:
        """
        递归扫描所有路径下的 SKILL.md 文件

        扫描 self.skill_paths 中指定的所有目录，递归查找 SKILL.md 文件
        找到的 skill 信息保存到 self.skills 列表中

        Note:
            - 使用 set 避免重复扫描同一目录
            - 忽略不存在的路径（仅记录日志）
            - 扫描结果缓存到 self.skills，不返回
        """
        scanned_dirs = set()

        for skill_path in self.skill_paths:
            if not skill_path.exists():
                LOG.info(f"Skill path {skill_path} does not exist")
                continue

            # 递归查找所有 SKILL.md 文件
            for skill_md in skill_path.rglob("SKILL.md"):
                skill_dir = skill_md.parent

                # 避免重复扫描同一目录
                if str(skill_dir) in scanned_dirs:
                    continue
                scanned_dirs.add(str(skill_dir))

                skill_info = self._read_skill(skill_dir)
                if skill_info:
                    self.skills.append(skill_info)

    def get_skill_content(self, name: str) -> Optional[str]:
        """
        获取指定 skill 的正文内容

        Args:
            name: Skill 名称

        Returns:
            str: Skill 正文内容（不含 frontmatter）
            None: 如果 skill 不存在
        """
        for skill in self.get_all_skils():
            if name == skill['name']:
                return skill['body']

    def get_all_skils(self) -> List[Dict]:
        """
        获取所有已加载的 skill 列表

        如果尚未扫描，会自动执行扫描

        Returns:
            List[Dict]: skill 信息字典列表
                每个字典包含：path, name, description, body
        """
        if len(self.skills) == 0:
            self._scan_skills()
        return self.skills

    def get_skill(self, skill_name: str) -> Optional[Dict]:
        """
        获取指定名称的 skill 信息

        Args:
            skill_name: Skill 名称

        Returns:
            Dict: skill 信息字典
                {
                    "path": str,
                    "name": str,
                    "description": str,
                    "body": str
                }
            None: 如果 skill 不存在
        """
        for skill in self.get_all_skils():
            if skill_name == skill['name']:
                return skill
        return None

    def get_all_skill_desc(self) -> str:
        """
        获取所有 skill 的描述文本

        格式：
        - skill_name: description
        - skill_name: description
        ...

        Returns:
            str: 格式化后的 skill 描述列表
        """
        skill_descs = []
        for skill in self.get_all_skils():
            skill_descs.append(f"- {skill['name']}: {skill['description']}")
        return "\n".join(skill_descs)


# 全局单例实例
SKILL_MANAGER = SkillManager()

if __name__ == "__main__":
    skill_manager = SkillManager()
    skill_manager._scan_skills()
    for skill in skill_manager.skills:
        print(f"path: {skill['path']}")
        print(f"name: {skill['name']}")
        print(f"description: {skill['description']}")
        print(f"body: {skill['body']}")
