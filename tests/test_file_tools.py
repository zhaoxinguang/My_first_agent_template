"""文件工具：只读、只 .txt、禁止逃出 documents/。"""

import pytest

from tools import file_tools


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """把 BASE_DIR 指向临时目录，测试不依赖仓库内 documents/ 的内容。"""
    monkeypatch.setattr(file_tools, "BASE_DIR", tmp_path)
    return tmp_path


def test_list_files_only_txt(docs):
    (docs / "a.txt").write_text("甲\n", encoding="utf-8")
    (docs / "b.md").write_text("乙\n", encoding="utf-8")
    (docs / "c.TXT").write_text("丙\n", encoding="utf-8")

    result = file_tools.list_files()

    assert "a.txt" in result
    assert "c.TXT" in result  # 后缀大小写不敏感
    assert "b.md" not in result


def test_list_files_when_empty(docs):
    assert "No .txt files" in file_tools.list_files()


def test_list_files_when_dir_missing(docs, monkeypatch, tmp_path):
    monkeypatch.setattr(file_tools, "BASE_DIR", tmp_path / "not_there")
    assert "does not exist" in file_tools.list_files()


def test_read_whole_file(docs):
    (docs / "a.txt").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")

    result = file_tools.read_text_file("a.txt")

    assert "第一行" in result and "第三行" in result
    assert "Lines: 1-3" in result


def test_read_line_range(docs):
    (docs / "a.txt").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")

    result = file_tools.read_text_file("a.txt", start_line=2, end_line=3)

    assert "Lines: 2-3" in result
    assert "2\n" in result and "3\n" in result
    assert "1\n" not in result.replace("Lines: 2-3", "")


def test_read_end_line_beyond_file_is_clamped(docs):
    (docs / "a.txt").write_text("1\n2\n", encoding="utf-8")

    result = file_tools.read_text_file("a.txt", start_line=1, end_line=99)

    assert "Lines: 1-2" in result


def test_rejects_non_txt_extension(docs):
    (docs / "a.md").write_text("内容", encoding="utf-8")

    assert "Only .txt" in file_tools.read_text_file("a.md")


def test_rejects_path_traversal(docs):
    """../../secret.txt 必须被拦截，不能靠后缀检查放过。"""
    assert "outside the documents directory" in file_tools.read_text_file(
        "../secret.txt"
    )
    assert "outside the documents directory" in file_tools.read_text_file(
        "../../etc/passwd.txt"
    )


def test_rejects_subdirectory(docs):
    (docs / "sub").mkdir()
    (docs / "sub" / "a.txt").write_text("内容", encoding="utf-8")

    assert "outside the documents directory" in file_tools.read_text_file(
        "sub/a.txt"
    )


def test_missing_file(docs):
    assert "File not found" in file_tools.read_text_file("nope.txt")


def test_start_line_validation(docs):
    (docs / "a.txt").write_text("1\n", encoding="utf-8")

    assert "start_line" in file_tools.read_text_file("a.txt", start_line=0)
    assert "end_line" in file_tools.read_text_file(
        "a.txt", start_line=3, end_line=1
    )


def test_start_line_beyond_file(docs):
    (docs / "a.txt").write_text("1\n2\n", encoding="utf-8")

    assert "only 2 lines" in file_tools.read_text_file("a.txt", start_line=9)


def test_non_utf8_file(docs):
    (docs / "gbk.txt").write_bytes("中文".encode("gbk"))

    assert "UTF-8" in file_tools.read_text_file("gbk.txt")


def test_tools_are_callable_from_agent_side(docs):
    """CodeAgent 眼里工具就是可调用函数，这里验证 __call__ 通路。"""
    (docs / "a.txt").write_text("内容\n", encoding="utf-8")

    assert "内容" in file_tools.read_text_file("a.txt")
    assert "a.txt" in file_tools.list_files()
