"""
Day23 Stage1

模拟 LangGraph 中:

State
 ↓
Node
 ↓
State Update

当前阶段:
不使用 LangGraph。

通过普通 Python 理解 State 生命周期。
"""


import re

from agent.state import AgentState



def create_initial_state(
    user_query: str
) -> AgentState:
    """
    创建 Agent 初始状态。

    所有 Agent 都应该从统一入口创建 State。
    """

    return {

        "user_query": user_query,

        "intent": "",

        "device_id": "",

        "retrieved_documents": [],

        "device_status": {},

        "maintenance_history": [],

        "diagnosis": "",

        "risk_level": "",

        "next_action": "",

        "error": "",
    }



# ===============================
# Node 1
# intent_node
# ===============================

def intent_node(
    state: AgentState
) -> dict:
    """
    根据用户问题判断任务类型。

    Read:
        user_query

    Update:
        intent
    """

    query = state["user_query"]


    if (
        "异常" in query
        or "故障" in query
        or "检查" in query
    ):

        return {
            "intent": "diagnosis"
        }


    return {
        "intent": "unknown"
    }



# ===============================
# Node 2
# device_parse_node
# ===============================

def device_parse_node(
    state: AgentState
) -> dict:
    """
    从用户问题中提取设备编号。

    Read:

        user_query


    Update:

        device_id
        error

    """

    query = state["user_query"]


    result = re.search(
        r"DEVICE-\d+",
        query
    )


    if not result:

        return {

            "device_id": "",

            "error": "DEVICE_ID_NOT_FOUND"
        }


    return {

        "device_id": result.group()

    }



# ===============================
# Node 3
# status_node
# ===============================

def status_node(
    state: AgentState
) -> dict:
    """
    查询设备状态。

    当前使用模拟数据。

    Read:

        device_id


    Update:

        device_status

    """

    device_id = state["device_id"]


    mock_devices = {


        "DEVICE-001": {

            "status": "warning",

            "temperature": 85,

            "current": 12.5
        },


        "DEVICE-002": {

            "status": "normal",

            "temperature": 42,

            "current": 8.2
        }

    }



    if device_id not in mock_devices:

        return {

            "device_status": {},

            "error": "DEVICE_NOT_FOUND"

        }


    return {


        "device_status":
            mock_devices[device_id]

    }




# ===============================
# Node 4
# risk_node
# ===============================

def risk_node(
    state: AgentState
) -> dict:
    """
    根据设备状态判断风险。

    Read:

        device_status


    Update:

        risk_level

    """

    status = state["device_status"]


    temperature = status.get(
        "temperature"
    )


    if temperature is None:

        return {

            "risk_level": "unknown"

        }


    if temperature >= 80:

        return {

            "risk_level": "high"

        }


    elif temperature >= 60:

        return {

            "risk_level": "medium"

        }


    else:

        return {

            "risk_level": "low"

        }




# ===============================
# State 更新模拟
# ===============================

def apply_update(
    state: AgentState,
    update: dict
) -> AgentState:
    """
    模拟 LangGraph:

        Node 返回部分 State

        ↓

        Graph 合并


    """

    state.update(update)

    return state




# ===============================
# Demo流程
# ===============================


def run_demo():

    state = create_initial_state(
        "DEVICE-001温度异常，请检查"
    )


    print("\n===== Initial State =====")
    print(state)



    nodes = [

        intent_node,

        device_parse_node,

        status_node,

        risk_node

    ]


    for node in nodes:


        update = node(state)


        state = apply_update(
            state,
            update
        )


        print(
            f"\n===== After {node.__name__} ====="
        )

        print(state)



    return state




if __name__ == "__main__":

    run_demo()