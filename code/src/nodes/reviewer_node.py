# reviewer_node.py
from pydantic import BaseModel, Field
from typing import List
from services.review import review_error_logs, generate_rewrite_plan


def reviewer_node(state):
    """
    Reviewer node: Reviews the error logs and provides analysis and suggestions
    for fixing the errors. This node only focuses on analysis, not file modification.
    """
    print(f"============================== Reviewer Analysis ==============================")
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        return state
    
    # Stateless review via service
    history_text = state.get("history_text") or []
    loop_count = state.get("loop_count", 0)
    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get('tutorial_reference', ''),
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs'),
        user_requirement=state.get('user_requirement', ''),
        similar_case_advice=state.get('similar_case_advice'),
        history_text=history_text,
        loop_count=loop_count,
    )

    print(review_content)

    rewrite_plan = generate_rewrite_plan(
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs', []),
        review_analysis=review_content,
        user_requirement=state.get('user_requirement', ''),
        loop_count=loop_count,
    )
    print(f"Rewrite plan: {rewrite_plan}")

    return {
        "history_text": updated_history,
        "review_analysis": review_content,
        "rewrite_plan": rewrite_plan,
        "loop_count": state.get("loop_count", 0) + 1,
        "input_writer_mode": "rewrite",
    }
