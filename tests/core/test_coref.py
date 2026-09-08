from app.core.coref import clean_rewrite_output


def test_normal_rewrite_is_unchanged():
    assert clean_rewrite_output("订单 1001 到哪了", "原问题") == "订单 1001 到哪了"


def test_complete_think_block_is_removed():
    text = "<think>用户在问商品是否销售</think>有汽水吗"
    assert clean_rewrite_output(text, "原问题") == "有汽水吗"


def test_orphan_closing_think_tag_keeps_final_answer():
    text = "用户在问商品是否销售</think>有汽水吗"
    assert clean_rewrite_output(text, "原问题") == "有汽水吗"


def test_only_think_markup_falls_back_to_original_query():
    assert clean_rewrite_output("<think>分析中</think>", "有汽水吗") == "有汽水吗"
