from parallax.grouped_gemm import GroupedGEMMDispatcher


def test_grouped_gemm_dispatcher_init() -> None:
    dispatcher = GroupedGEMMDispatcher(num_experts=8)
    assert dispatcher.num_experts == 8
