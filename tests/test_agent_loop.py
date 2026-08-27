"""
Day22 Stage5

Multi Tool Agent Loop Test
"""


from __future__ import annotations

import json
import sys
from pathlib import Path


# ======================================================
# 添加项目根目录到 Python Path
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from agent.agent_loop import (
    MultiToolAgent,
)



# ==========================================================
# Mock LLM Decision
# ==========================================================


class MockAgent(MultiToolAgent):


    def __init__(self):

        super().__init__()

        self.counter = 0



    def _llm_decision(
        self,
        messages,
    ):


        self.counter += 1



        # Step 1

        if self.counter == 1:

            return {

                "type":
                    "tool_call",

                "tool_name":
                    "get_device_status",

                "arguments":
                    {
                        "device_id":
                            "DEVICE-001"
                    }
            }



        # Step 2

        if self.counter == 2:

            return {

                "type":
                    "tool_call",

                "tool_name":
                    "query_maintenance_history",

                "arguments":
                    {
                        "device_id":
                            "DEVICE-001"
                    }
            }



        # Step 3

        return {

            "type":
                "final_answer",

            "content":
                (
                    "DEVICE-001当前状态"
                    "与历史维护信息已经获取。"
                )
        }



# ==========================================================
# Test
# ==========================================================


def test_multi_tool_agent_loop():


    agent = MockAgent()



    result = agent.run(

        "DEVICE-001现在有异常吗？"
        "它过去有没有发生过类似问题？"

    )


    print()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )



    assert (
        result["status"]
        ==
        "completed"
    )


    steps = result["steps"]



    # ------------------------------------------------------
    # Step1 Tool
    # ------------------------------------------------------

    assert (
        steps[0]["tool_name"]
        ==
        "get_device_status"
    )



    # ------------------------------------------------------
    # Step2 Tool
    # ------------------------------------------------------

    assert (
        steps[1]["tool_name"]
        ==
        "query_maintenance_history"
    )



    # ------------------------------------------------------
    # Step3 Final
    # ------------------------------------------------------

    assert (
        steps[2]["type"]
        ==
        "final_answer"
    )



def test_direct_answer_without_tool():


    class DirectAgent(
        MultiToolAgent
    ):


        def _llm_decision(
            self,
            messages,
        ):

            return {

                "type":
                    "final_answer",

                "content":
                    "你好"
            }



    agent = DirectAgent()



    result = agent.run(
        "你好"
    )


    assert (
        result["steps"][0]["type"]
        ==
        "final_answer"
    )



def test_max_step_protection():


    class InfiniteAgent(
        MultiToolAgent
    ):


        def _llm_decision(
            self,
            messages,
        ):


            return {

                "type":
                    "tool_call",

                "tool_name":
                    "get_device_status",

                "arguments":
                    {
                        "device_id":
                            "DEVICE-001"
                    }
            }



    agent = InfiniteAgent()

    agent.max_steps = 2



    result = agent.run(
        "查询设备"
    )


    assert (
        result["status"]
        ==
        "max_steps_exceeded"
    )



if __name__ == "__main__":


    test_multi_tool_agent_loop()

    test_direct_answer_without_tool()

    test_max_step_protection()


    print(
        "\nStage5 Agent Loop Test PASS"
    )
