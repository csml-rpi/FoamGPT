from dataclasses import dataclass, field
from typing import List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import argparse
from pathlib import Path
from utils import LLMService, GraphState

from config import Config
from services import set_global_llm_service
from nodes.planner_node import planner_node
from nodes.meshing_node import meshing_node
from nodes.input_writer_node import input_writer_node
from nodes.local_runner_node import local_runner_node
from nodes.reviewer_node import reviewer_node
from nodes.visualization_node import visualization_node
from nodes.hpc_runner_node import hpc_runner_node
from router_func import (
    route_after_planner, 
    route_after_input_writer, 
    route_after_runner, 
    route_after_reviewer
)
import json

def create_foam_agent_graph() -> StateGraph:
    """Create the OpenFOAM agent workflow graph."""
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("meshing", meshing_node)
    workflow.add_node("input_writer", input_writer_node)
    workflow.add_node("local_runner", local_runner_node)
    workflow.add_node("hpc_runner", hpc_runner_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("visualization", visualization_node)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", route_after_planner)
    workflow.add_edge("meshing", "input_writer")
    workflow.add_conditional_edges("input_writer", route_after_input_writer)
    workflow.add_conditional_edges("hpc_runner", route_after_runner)
    workflow.add_conditional_edges("local_runner", route_after_runner)
    workflow.add_conditional_edges("reviewer", route_after_reviewer)
    workflow.add_edge("visualization", END)
    
    return workflow

def initialize_state(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None) -> GraphState:
    case_stats = json.load(open(f"{config.database_path}/raw/openfoam_case_stats.json", "r"))
    # mesh_type = "custom_mesh" if custom_mesh_path else "standard_mesh"
    llm_service = LLMService(config)
    # Ensure service-layer modules reuse the same LLMService instance (avoid double-loading models).
    set_global_llm_service(llm_service)

    state = GraphState(
        user_requirement=user_requirement,
        config=config,
        case_dir="",
        tutorial="",
        case_name="",
        subtasks=[],
        current_subtask_index=0,
        error_command=None,
        error_content=None,
        loop_count=0,
        llm_service=llm_service,
        case_stats=case_stats,
        tutorial_reference=None,
        case_path_reference=None,
        dir_structure_reference=None,
        case_info=None,
        allrun_reference=None,
        dir_structure=None,
        commands=None,
        foamfiles=None,
        error_logs=None,
        history_text=None,
        case_domain=None,
        case_category=None,
        case_solver=None,
        mesh_info=None,
        mesh_commands=None,
        custom_mesh_used=None,
        mesh_type=None,
        custom_mesh_path=custom_mesh_path,
        review_analysis=None,
        rewrite_plan=None,
        input_writer_mode="initial",
        requires_hpc=None,
        requires_visualization=None,
        job_id=None,
        cluster_info=None,
        slurm_script_path=None,
        termination_reason=None
    )
    if custom_mesh_path:
        print(f"Custom mesh path: {custom_mesh_path}")
    else:
        print("No custom mesh path provided.")
    return state

def _exit_without_teardown(rc: int) -> None:
    """FoamGPT kit_e46: flush, terminate this process's children (SIGTERM, then SIGKILL after 30 s), os._exit(rc)."""
    import os
    import sys
    import psutil
    sys.stdout.flush()
    sys.stderr.flush()
    kids = psutil.Process().children(recursive=True)
    for k in kids:
        try:
            k.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(kids, timeout=30)
    for k in alive:
        try:
            k.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=10)
    print(f"[kit_e46] exit {rc}: terminated {len(kids)} child process(es), {len(alive)} needed SIGKILL", flush=True)
    os._exit(rc)


def main(user_requirement: str, config: Config, custom_mesh_path: Optional[str] = None):
    """Main function to run the OpenFOAM workflow."""
    
    # Create and compile the graph
    workflow = create_foam_agent_graph()
    app = workflow.compile()
    
    # Initialize the state
    initial_state = initialize_state(user_requirement, config, custom_mesh_path)
    
    print("Starting Foam-Agent...")
    
    # Invoke the graph
    try:
        result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

        termination_reason = result.get("termination_reason")
        if termination_reason == "max_review_loop_reached":
            print("Workflow finished after reaching the maximum review loop limit.")
        else:
            print("Workflow completed successfully!")

        # Print final statistics
        if result.get("llm_service"):
            result["llm_service"].print_statistics()
        
        # print(f"Final state: {result}")
        
    except Exception as e:
        print(f"Workflow failed with error: {e}")
        raise

if __name__ == "__main__":
    # python main.py
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM workflow"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=f"{Path(__file__).parent.parent}/user_requirement.txt",
        help="User requirement file path for the workflow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for the workflow.",
    )
    parser.add_argument(
        "--custom_mesh_path",
        type=str,
        default=None,
        help="Path to custom mesh file (e.g., .msh, .stl, .obj). If not provided, no custom mesh will be used.",
    )
    parser.add_argument(
        "--reuse_generated_dir",
        type=str,
        default="",
        help=(
            "Path to a directory containing previously generated OpenFOAM files. "
            "If a file exists at <reuse_generated_dir>/<folder>/<file>, Foam-Agent will copy it into the current output and skip generation for that file."
        ),
    )
    parser.add_argument(
        "--dataset_log_path",
        type=str,
        default="",
        help="Path to dataset.jsonl for fine-tuning data extraction. Empty = disabled.",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default="",
        help="Case identifier for dataset logging, e.g. 'Basic/Cavity/1'.",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="",
        help="Path to JSON file with 'models' and/or 'substep_model_overrides' keys.",
    )

    args = parser.parse_args()
    print(f"args: {args}")
    
    # Initialize configuration.
    config = Config()

    print(f"config: {config}")

    if args.output_dir != "":
        config.case_dir = args.output_dir

    if args.reuse_generated_dir:
        config.reuse_generated_dir = args.reuse_generated_dir

    if args.dataset_log_path:
        config.dataset_log_path = args.dataset_log_path
    if args.case_id:
        config.case_id = args.case_id

    if args.model_config_path:
        import json as _json
        with open(args.model_config_path) as f:
            mcfg = _json.load(f)
        if "models" in mcfg:
            config.models.update(mcfg["models"])
        if "substep_model_overrides" in mcfg:
            config.substep_model_overrides = mcfg["substep_model_overrides"]

    with open(args.prompt_path, 'r') as f:
        user_requirement = f.read()
    
    # FoamGPT kit_e46 (E46-E48): with vLLM 0.31 this process did not exit after the workflow ended (E48 smoke,
    # Cavity/5: still alive 90 min after "Workflow completed successfully!" and the statistics; under vLLM 0.20.1 it
    # exited). Once main() has returned or raised, everything the run produces is written: exit with the status Python
    # would give (traceback printed, 1, on an exception), after terminating this process's own children (the vLLM
    # engine core and its GPU workers) and without the interpreter's exit handlers.
    _rc = 0
    try:
        main(user_requirement, config, args.custom_mesh_path)
    except SystemExit as _e:
        _rc = _e.code if isinstance(_e.code, int) else (0 if _e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _rc = 1
    finally:
        _exit_without_teardown(_rc)
