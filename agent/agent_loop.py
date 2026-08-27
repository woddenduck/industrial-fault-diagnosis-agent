from __future__ import annotations

import json
from typing import Any, Optional

from agent.structured_output import (
    generate_decision,
    LLMConfig,
)

from agent.validation import (
    validate_agent_output,
    execute_tool_call,
)


class AgentLoopError(Exception):
    pass


class MultiToolAgent:

    """
    Stage6 Three Tool Industrial Diagnostic Agent

    Flow:

    User
      |
      v
    LLM
      |
      v
    Tool Call
      |
      v
    Validation
      |
      v
    Tool Execute
      |
      v
    Tool Result
      |
      v
    Agent State
      |
      v
    Next LLM

    """


    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        max_steps: int = 6,
    ):

        self.llm_config = llm_config
        self.max_steps = max_steps


    def run(
        self,
        question: str,
    ) -> dict[str, Any]:


        messages = [
            {
                "role": "user",
                "content": question
            }
        ]


        steps = []

        tool_trace = []

        side_effects = {}


        for step in range(
            1,
            self.max_steps + 1
        ):


            print("=" * 70)

            print(
                f"Agent Loop Step {step}"
            )


            decision = self._llm_decision(
                messages
            )


            print(
                json.dumps(
                    decision,
                    ensure_ascii=False,
                    indent=2
                )
            )


            steps.append(
                {
                    "step": step,
                    **decision
                }
            )


            #
            # Final Answer
            #
            if decision["type"] == "final_answer":

                return {

                    "status":
                        "completed",

                    "answer":
                        decision.get(
                            "content",
                            ""
                        ),

                    "steps":
                        steps,

                    "tool_trace":
                        tool_trace,

                    "side_effects":
                        side_effects,

                    "messages":
                        messages
                }


            #
            # Tool Call
            #
            if decision["type"] != "tool_call":

                raise AgentLoopError(
                    "unknown decision type"
                )


            #
            # Validation
            #
            validation = (
                validate_agent_output(
                    decision
                )
            )


            if not validation["valid"]:

                return {

                    "status":
                        "validation_failed",

                    "error":
                        validation
                }



            tool_name = (
                decision["tool_name"]
            )


            tool_trace.append(
                tool_name
            )


            #
            # Execute Tool
            #
            tool_result = (
                execute_tool_call(
                    decision
                )
            )


            #
            # Action Tool
            #
            if tool_name == "create_diagnostic_report":
                report_data = (
                    tool_result
                    .get("data", {})
                )

                report_id = (
                    report_data
                    .get("report_id")
                )

                side_effects[
                    "report_id"
                ] = report_id



            #
            # update state
            #

            messages.append(
                {
                    "role":
                        "assistant",

                    "content":
                        json.dumps(
                            decision,
                            ensure_ascii=False
                        )
                }
            )


            messages.append(
                {
                    "role":
                        "tool",

                    "content":
                        json.dumps(
                            tool_result,
                            ensure_ascii=False
                        )
                }
            )


        return {

            "status":
                "max_steps_exceeded",

            "steps":
                steps,

            "tool_trace":
                tool_trace
        }



    def _llm_decision(
        self,
        messages
    ):


        prompt = f"""

你是工业设备诊断 Agent。

你的任务：

根据当前 Agent State，
决定下一步：

1.
调用 Tool

或者

2.
输出最终答案。


可用 Tool：

1.
get_device_status

用途:
查询设备当前状态


2.
query_maintenance_history

用途:
查询维修历史


3.
create_diagnostic_report

用途:
创建诊断报告，
产生 report_id


严格要求：

- 只输出 JSON
- 不执行 Tool
- 不伪造 Tool Result


当前状态:

{json.dumps(
    messages,
    ensure_ascii=False,
    indent=2
)}

"""


        #
        # 这里是真实调用 Qwen3-8B
        #
        decision = generate_decision(
            user_message=prompt,
            config=self.llm_config
        )


        return decision



def run_agent(question):

    agent = MultiToolAgent()

    return agent.run(
        question
    )



if __name__ == "__main__":

    result = run_agent(
        """
检查DEVICE-001当前运行情况，
结合维修记录判断问题，
并生成诊断报告。
"""
    )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )