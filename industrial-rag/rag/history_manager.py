"""
Day18 Stage7
History Manager

功能:
1. 小历史保持原样
2. 长历史结构化压缩
3. 保留最近3轮
"""


import re


MAX_ROUNDS = 5
RECENT_ROUNDS = 3



def extract_fault_codes(text):

    return re.findall(
        r"F\d{5}",
        text
    )



def summarize_history(history):

    summary = {

        "device_model":None,

        "fault_codes":[],

        "observed_symptoms":[],

        "completed_steps":[],

        "risk_status":[],

        "unresolved_questions":[]

    }


    for item in history:


        content=item["content"]


        # 型号

        models=re.findall(
            r"G\d+[A-Z]?",
            content
        )

        if models:
            summary["device_model"]=models[0]


        # 故障码

        faults=extract_fault_codes(content)

        for f in faults:

            if f not in summary["fault_codes"]:
                summary["fault_codes"].append(f)



        # 已执行步骤

        keywords=[
            "检查",
            "重新上电",
            "更换",
            "测试"
        ]

        for k in keywords:

            if k in content:

                summary["completed_steps"].append(
                    content
                )



    return summary



def truncate_history(history):


    rounds=len(history)//2


    if rounds<=MAX_ROUNDS:

        return {

            "summary":None,

            "recent_history":history

        }



    old_history=history[:-RECENT_ROUNDS*2]


    recent_history=history[-RECENT_ROUNDS*2:]


    summary=summarize_history(
        old_history
    )


    return {

        "summary":summary,

        "recent_history":recent_history

    }



def build_history_context(history):


    return truncate_history(history)